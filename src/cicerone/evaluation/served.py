"""Production replay of served recommendation lists."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from rectools.metrics import (
    MAP,
    MRR,
    NDCG,
    AvgRecPopularity,
    CatalogCoverage,
    HitRate,
    MeanInvUserFreq,
    Precision,
    Recall,
    calc_metrics,
)

from cicerone.blending import COLD_START_USER_ID
from cicerone.config.constants import TRACK_KIND_IMPRESSION
from cicerone.evaluation.metrics import OCCURRED_AT, _frame, _ratio
from cicerone.io.recommendation_schema import (
    ITEM_COLUMN,
    RANK_COLUMN,
    SCORE_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
    VARIANT_COLUMN,
)

logger = logging.getLogger(__name__)
_RECTOOLS_ERRORS = (ValueError, TypeError, LookupError)


@dataclass(frozen=True)
class ServedEvalReport:
    n_users: int
    n_users_with_events: int
    metrics: dict[str, float]
    by_source: dict[str, dict[str, float]]
    generated_at: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_users": self.n_users,
            "n_users_with_events": self.n_users_with_events,
            "metrics": dict(self.metrics),
            "by_source": {key: dict(value) for key, value in sorted(self.by_source.items())},
            "generated_at": self.generated_at,
        }


def replay_ks(configured: Sequence[int], *, top_k: int) -> tuple[int, ...]:
    if configured:
        return tuple(sorted({int(value) for value in configured if 1 <= int(value) <= top_k})) or (top_k,)
    values = {top_k, *[k for k in (5, 10) if k <= top_k]}
    return tuple(sorted(values))


def filter_recs_to_assigned(
    recommendations: pd.DataFrame,
    assigned: Mapping[str, str] | None,
) -> pd.DataFrame:
    if recommendations.empty or not assigned or VARIANT_COLUMN not in recommendations.columns:
        return recommendations
    expected = recommendations[USER_COLUMN].astype(str).map(assigned)
    return recommendations.loc[expected.isna() | (recommendations[VARIANT_COLUMN].astype(str) == expected)]


def filter_events_to_recommended(
    events: pd.DataFrame,
    recommendations: pd.DataFrame,
    *,
    assigned: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    if events.empty or recommendations.empty:
        return events.iloc[0:0].copy() if not events.empty else events
    recs = recommendations.copy()
    recs[USER_COLUMN] = recs[USER_COLUMN].astype(str)
    recs[ITEM_COLUMN] = recs[ITEM_COLUMN].astype(str)
    recs = recs[recs[USER_COLUMN] != COLD_START_USER_ID]
    recs = filter_recs_to_assigned(recs, assigned)
    keys = recs.loc[:, [USER_COLUMN, ITEM_COLUMN]].drop_duplicates()
    frame = events.copy()
    frame[USER_COLUMN] = frame[USER_COLUMN].astype(str)
    frame[ITEM_COLUMN] = frame[ITEM_COLUMN].astype(str)
    return frame.merge(keys, on=[USER_COLUMN, ITEM_COLUMN], how="inner")


def _hit_rate(reco: pd.DataFrame, relevant: pd.DataFrame, *, k: int) -> float:
    if reco.empty or relevant.empty:
        return 0.0
    truth = relevant.groupby(USER_COLUMN)[ITEM_COLUMN].agg(set)
    hits = 0
    scored = 0
    top = reco[reco[RANK_COLUMN] <= k] if RANK_COLUMN in reco.columns else reco
    for user_id, group in top.groupby(USER_COLUMN):
        items = truth.get(str(user_id))
        if items is None:
            continue
        scored += 1
        if set(group[ITEM_COLUMN].astype(str)) & set(items):
            hits += 1
    return _ratio(hits, scored)


def _ranking_metric_defs(k: int) -> dict[str, object]:
    return {
        f"HitRate@{k}": HitRate(k=k),
        f"MAP@{k}": MAP(k=k),
        f"NDCG@{k}": NDCG(k=k),
        f"Recall@{k}": Recall(k=k),
        f"MRR@{k}": MRR(k=k),
        f"Precision@{k}": Precision(k=k),
    }


def _catalog_metric_defs(k: int, *, with_prev: bool) -> dict[str, object]:
    metrics: dict[str, object] = {f"CatalogCoverage@{k}": CatalogCoverage(k=k, normalize=True)}
    if with_prev:
        metrics[f"MeanInvUserFreq@{k}"] = MeanInvUserFreq(k=k)
        metrics[f"AvgRecPopularity@{k}"] = AvgRecPopularity(k=k)
    return metrics


def _impression_matches_generated_at(stamp: object, generated_at: str) -> bool:
    raw = str(stamp or "").strip()
    if not raw:
        return False
    left = pd.to_datetime(raw, utc=True, errors="coerce")
    right = pd.to_datetime(generated_at, utc=True, errors="coerce")
    if pd.isna(left) or pd.isna(right):
        return False
    return bool(left == right)


def recs_from_impressions(
    track_rows: Sequence[Mapping[str, Any]],
    *,
    generated_at: str | None = None,
    recommendations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build reco rows from impression events (optionally stamped to one job)."""
    rows: list[dict[str, Any]] = []
    for raw in track_rows:
        if str(raw.get("kind") or "") != TRACK_KIND_IMPRESSION:
            continue
        user_id = str(raw.get(USER_COLUMN) or "")
        item_id = str(raw.get(ITEM_COLUMN) or "")
        if not user_id or not item_id:
            continue
        if generated_at and not _impression_matches_generated_at(raw.get("generated_at"), generated_at):
            continue
        raw_rank = raw.get(RANK_COLUMN)
        if isinstance(raw_rank, (int, float, str)):
            try:
                rank = int(raw_rank)
            except (TypeError, ValueError):
                rank = 1
        else:
            rank = 1
        row: dict[str, Any] = {
            USER_COLUMN: user_id,
            ITEM_COLUMN: item_id,
            RANK_COLUMN: rank if rank >= 1 else 1,
        }
        source = raw.get(SOURCE_COLUMN)
        if source:
            row[SOURCE_COLUMN] = str(source)
        variant = raw.get(VARIANT_COLUMN)
        if variant:
            row[VARIANT_COLUMN] = str(variant)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame = frame.sort_values([USER_COLUMN, RANK_COLUMN], kind="mergesort")
    frame = frame.drop_duplicates(subset=[USER_COLUMN, ITEM_COLUMN], keep="first")
    if recommendations is not None and not recommendations.empty:
        keys = [USER_COLUMN, ITEM_COLUMN]
        extra = [
            column
            for column in (SOURCE_COLUMN, SCORE_COLUMN, VARIANT_COLUMN)
            if column in recommendations.columns
        ]
        if extra:
            lookup = recommendations.loc[:, [*keys, *extra]].copy()
            lookup[USER_COLUMN] = lookup[USER_COLUMN].astype(str)
            lookup[ITEM_COLUMN] = lookup[ITEM_COLUMN].astype(str)
            lookup = lookup.drop_duplicates(subset=keys, keep="first")
            frame = frame.merge(lookup, on=keys, how="left", suffixes=("", "_job"))
            for column in extra:
                job_column = f"{column}_job"
                if job_column in frame.columns:
                    frame[column] = frame[column].where(frame[column].notna(), frame[job_column])
                    frame = frame.drop(columns=[job_column])
    if SCORE_COLUMN not in frame.columns:
        top = int(frame[RANK_COLUMN].max()) if RANK_COLUMN in frame.columns and not frame.empty else 1
        frame[SCORE_COLUMN] = (top + 1 - frame[RANK_COLUMN]).astype(float)
    else:
        frame[SCORE_COLUMN] = pd.to_numeric(frame[SCORE_COLUMN], errors="coerce").fillna(0.0)
    return frame.reset_index(drop=True)


def _unseen_relevant(window: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    if window.empty or USER_COLUMN not in window.columns or ITEM_COLUMN not in window.columns:
        return window.iloc[0:0]
    relevant = window.loc[:, [USER_COLUMN, ITEM_COLUMN]].drop_duplicates()
    if relevant.empty or prev.empty:
        return relevant
    seen = prev.loc[:, [USER_COLUMN, ITEM_COLUMN]].drop_duplicates()
    merged = relevant.merge(seen, on=[USER_COLUMN, ITEM_COLUMN], how="left", indicator=True)
    return merged.loc[merged["_merge"] == "left_only", [USER_COLUMN, ITEM_COLUMN]].reset_index(drop=True)


def _prev_interactions(events: pd.DataFrame, generated_at: str | None) -> pd.DataFrame:
    if events.empty or generated_at is None or OCCURRED_AT not in events.columns:
        return events.iloc[0:0]
    start = pd.to_datetime(generated_at, utc=True, errors="coerce")
    if pd.isna(start):
        return events.iloc[0:0]
    frame = events.loc[events[OCCURRED_AT] <= start, [USER_COLUMN, ITEM_COLUMN]].drop_duplicates()
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["weight"] = 1.0
    return frame


def _served_catalog(
    catalog: pd.DataFrame | Sequence[object] | None,
    recs: pd.DataFrame,
    events: pd.DataFrame,
) -> list[str]:
    if isinstance(catalog, pd.DataFrame) and not catalog.empty and ITEM_COLUMN in catalog.columns:
        return list(dict.fromkeys(catalog[ITEM_COLUMN].astype(str)))
    if catalog is not None and not isinstance(catalog, pd.DataFrame):
        return list(dict.fromkeys(str(item) for item in catalog))
    seen: list[str] = []
    for frame in (events, recs):
        if frame is not None and not frame.empty and ITEM_COLUMN in frame.columns:
            seen.extend(frame[ITEM_COLUMN].astype(str))
    return list(dict.fromkeys(seen))


def evaluate_served(
    recommendations: pd.DataFrame,
    events: pd.DataFrame,
    *,
    generated_at: str | None,
    ks: Sequence[int],
    event_types: Sequence[str],
    history: pd.DataFrame | None = None,
    catalog: pd.DataFrame | Sequence[object] | None = None,
    assigned: Mapping[str, str] | None = None,
    impressions: pd.DataFrame | None = None,
) -> ServedEvalReport | None:
    if recommendations is None or recommendations.empty:
        return None
    recs = recommendations.copy()
    recs[USER_COLUMN] = recs[USER_COLUMN].astype(str)
    recs[ITEM_COLUMN] = recs[ITEM_COLUMN].astype(str)
    recs = recs[recs[USER_COLUMN] != COLD_START_USER_ID]
    recs = filter_recs_to_assigned(recs, assigned)
    if recs.empty:
        return None
    used_impressions = False
    if impressions is not None and not impressions.empty:
        recs = impressions.copy()
        recs[USER_COLUMN] = recs[USER_COLUMN].astype(str)
        recs[ITEM_COLUMN] = recs[ITEM_COLUMN].astype(str)
        recs = recs[recs[USER_COLUMN] != COLD_START_USER_ID]
        recs = filter_recs_to_assigned(recs, assigned)
        if recs.empty:
            return None
        used_impressions = True
    all_events = _frame(events)
    if all_events.empty:
        return ServedEvalReport(
            n_users=int(recs[USER_COLUMN].nunique()),
            n_users_with_events=0,
            metrics={},
            by_source={},
            generated_at=generated_at,
        )
    window_events = all_events
    if generated_at and OCCURRED_AT in window_events.columns:
        start = pd.to_datetime(generated_at, utc=True, errors="coerce")
        if pd.notna(start):
            window_events = window_events[window_events[OCCURRED_AT] > start]
    if event_types and "event_type" in window_events.columns:
        window_events = window_events[window_events["event_type"].astype(str).isin(set(event_types))]
    if (
        not used_impressions
        and history is not None
        and not history.empty
        and OCCURRED_AT in window_events.columns
    ):
        live = recs.copy()
        if generated_at:
            live["generated_at"] = generated_at
            combined = pd.concat([history, live], ignore_index=True)
        else:
            combined = history
        hist_recs = _recs_from_history(combined, window_events)
        if not hist_recs.empty:
            hist_recs = hist_recs[hist_recs[USER_COLUMN] != COLD_START_USER_ID]
            hist_recs = filter_recs_to_assigned(hist_recs, assigned)
        if not hist_recs.empty:
            recs = hist_recs
    prev = _prev_interactions(all_events, generated_at)
    relevant = _unseen_relevant(window_events, prev)
    if used_impressions and not recs.empty and not relevant.empty:
        served_users = set(recs[USER_COLUMN].astype(str))
        relevant = relevant.loc[relevant[USER_COLUMN].isin(served_users)]
    n_users = int(recs[USER_COLUMN].nunique())
    n_with_events = int(relevant[USER_COLUMN].nunique()) if not relevant.empty else 0
    catalog_ids = _served_catalog(catalog, recs, all_events)
    metrics: dict[str, float] = {}
    for k in ks:
        if not relevant.empty and not recs.empty:
            reco_k = recs[recs[RANK_COLUMN] <= k] if RANK_COLUMN in recs.columns else recs
            interactions = relevant.copy()
            interactions["weight"] = 1.0
            extra: dict[str, object] = {}
            extra.update(_catalog_metric_defs(k, with_prev=not prev.empty))
            try:
                computed = calc_metrics(
                    {**_ranking_metric_defs(k), **extra},
                    reco=reco_k,
                    interactions=interactions,
                    catalog=catalog_ids,
                    prev_interactions=prev if not prev.empty else None,
                )
                metrics.update({key: float(value) for key, value in computed.items()})
            except _RECTOOLS_ERRORS as exc:
                logger.exception(
                    "RecTools calc_metrics failed for k=%s (%s: %s)",
                    k,
                    type(exc).__name__,
                    exc,
                )
                metrics[f"HitRate@{k}"] = _hit_rate(recs, relevant, k=k)
        else:
            metrics[f"HitRate@{k}"] = _hit_rate(recs, relevant, k=k)
    by_source: dict[str, dict[str, float]] = {}
    if SOURCE_COLUMN in recs.columns:
        for source, group in recs.groupby(SOURCE_COLUMN, dropna=True):
            source_metrics: dict[str, float] = {}
            for k in ks:
                if not relevant.empty and not group.empty:
                    reco_k = group[group[RANK_COLUMN] <= k] if RANK_COLUMN in group.columns else group
                    interactions = relevant.copy()
                    interactions["weight"] = 1.0
                    try:
                        computed = calc_metrics(
                            {f"HitRate@{k}": HitRate(k=k)},
                            reco=reco_k,
                            interactions=interactions,
                        )
                        source_metrics.update({key: float(value) for key, value in computed.items()})
                    except _RECTOOLS_ERRORS as exc:
                        logger.exception(
                            "RecTools HitRate failed for source=%s k=%s (%s: %s)",
                            source,
                            k,
                            type(exc).__name__,
                            exc,
                        )
                        source_metrics[f"HitRate@{k}"] = _hit_rate(group, relevant, k=k)
                else:
                    source_metrics[f"HitRate@{k}"] = _hit_rate(group, relevant, k=k)
            by_source[str(source)] = source_metrics
    return ServedEvalReport(
        n_users=n_users,
        n_users_with_events=n_with_events,
        metrics=metrics,
        by_source=by_source,
        generated_at=generated_at,
    )


def _recs_from_history(history: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Use each user's newest snapshot with generated_at <= that user's first event."""
    hist = history.copy()
    hist[USER_COLUMN] = hist[USER_COLUMN].astype(str)
    hist[ITEM_COLUMN] = hist[ITEM_COLUMN].astype(str)
    hist["generated_at"] = pd.to_datetime(hist["generated_at"], utc=True, errors="coerce")
    hist = hist.dropna(subset=["generated_at"])
    hist = hist.drop_duplicates(subset=[USER_COLUMN, ITEM_COLUMN, "generated_at"], keep="last")
    if hist.empty or events.empty or OCCURRED_AT not in events.columns:
        return hist.iloc[0:0]
    first = events.groupby(USER_COLUMN, sort=False)[OCCURRED_AT].min().reset_index()
    first[USER_COLUMN] = first[USER_COLUMN].astype(str)
    snaps = (
        hist.loc[:, [USER_COLUMN, "generated_at"]]
        .drop_duplicates()
        .sort_values("generated_at")
        .rename(columns={"generated_at": "snap_at"})
    )
    first = first.sort_values(OCCURRED_AT).rename(columns={OCCURRED_AT: "event_at"})
    chosen = pd.merge_asof(
        first,
        snaps,
        by=USER_COLUMN,
        left_on="event_at",
        right_on="snap_at",
        direction="backward",
    )
    chosen = chosen.dropna(subset=["snap_at"])
    if chosen.empty:
        return hist.iloc[0:0]
    picked = chosen.loc[:, [USER_COLUMN, "snap_at"]].rename(columns={"snap_at": "generated_at"})
    return hist.merge(picked, on=[USER_COLUMN, "generated_at"], how="inner")
