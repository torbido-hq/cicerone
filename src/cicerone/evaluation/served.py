"""Production replay of served recommendation lists."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
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
from cicerone.errors import OPTIONAL_EVAL_ERRORS, log_caught
from cicerone.evaluation.metrics import OCCURRED_AT, normalized_frame, safe_ratio, with_str_ids
from cicerone.evaluation.tracking import filter_events_by_types, variant_blank
from cicerone.io.recommendation_schema import (
    ITEM_COLUMN,
    RANK_COLUMN,
    SCORE_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
    VARIANT_COLUMN,
)
from cicerone.ranking import drop_seen_pairs

logger = logging.getLogger(__name__)


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


def _served_recs(recommendations: pd.DataFrame, assigned: Mapping[str, str] | None) -> pd.DataFrame:
    recs = with_str_ids(recommendations)
    recs = recs[recs[USER_COLUMN] != COLD_START_USER_ID]
    return filter_recs_to_assigned(recs, assigned)


def filter_events_to_recommended(
    events: pd.DataFrame,
    recommendations: pd.DataFrame,
    *,
    assigned: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    if events.empty or recommendations.empty:
        return events.iloc[0:0].copy() if not events.empty else events
    keys = _served_recs(recommendations, assigned).loc[:, [USER_COLUMN, ITEM_COLUMN]].drop_duplicates()
    return with_str_ids(events).merge(keys, on=[USER_COLUMN, ITEM_COLUMN], how="inner")


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
    return safe_ratio(hits, scored)


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


def _attach_recommendation_fields(
    frame: pd.DataFrame,
    recommendations: pd.DataFrame | None,
    *,
    on_variant: bool,
) -> pd.DataFrame:
    if frame.empty or recommendations is None or recommendations.empty:
        return frame
    keys = [USER_COLUMN, ITEM_COLUMN]
    extra = [
        column
        for column in (SOURCE_COLUMN, SCORE_COLUMN, VARIANT_COLUMN)
        if column in recommendations.columns
    ]
    if on_variant and VARIANT_COLUMN in extra:
        keys.append(VARIANT_COLUMN)
    elif VARIANT_COLUMN in extra:
        extra = [column for column in extra if column != VARIANT_COLUMN]
    if not extra:
        return frame
    columns = [*keys, *[column for column in extra if column not in keys]]
    if "generated_at" in recommendations.columns:
        columns.append("generated_at")
    lookup = with_str_ids(recommendations.loc[:, columns])
    if on_variant and VARIANT_COLUMN in lookup.columns:
        lookup = lookup.loc[~variant_blank(lookup[VARIANT_COLUMN])]
    if "generated_at" in lookup.columns:
        lookup["generated_at"] = pd.to_datetime(lookup["generated_at"], utc=True, errors="coerce")
        lookup = lookup.sort_values("generated_at", kind="mergesort", na_position="first")
        lookup = lookup.drop(columns=["generated_at"]).drop_duplicates(subset=keys, keep="last")
    else:
        lookup = lookup.drop_duplicates(subset=keys, keep="first")
    frame = frame.merge(lookup, on=keys, how="left", suffixes=("", "_job"))
    for column in extra:
        job_column = f"{column}_job"
        if job_column in frame.columns:
            frame[column] = frame[column].where(frame[column].notna(), frame[job_column])
            frame = frame.drop(columns=[job_column])
    return frame


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
    if recommendations is not None and not recommendations.empty and VARIANT_COLUMN in frame.columns:
        blank = variant_blank(frame[VARIANT_COLUMN])
        stamped = frame.loc[~blank].drop_duplicates(
            subset=[USER_COLUMN, ITEM_COLUMN, VARIANT_COLUMN], keep="first"
        )
        unstamped = frame.loc[blank].drop_duplicates(subset=[USER_COLUMN, ITEM_COLUMN], keep="first")
        stamped = _attach_recommendation_fields(stamped, recommendations, on_variant=True)
        unstamped = _attach_recommendation_fields(unstamped, recommendations, on_variant=False)
        frame = pd.concat([stamped, unstamped], ignore_index=True)
    else:
        frame = frame.drop_duplicates(subset=[USER_COLUMN, ITEM_COLUMN], keep="first")
        frame = _attach_recommendation_fields(frame, recommendations, on_variant=False)
    if SCORE_COLUMN not in frame.columns:
        top = int(frame[RANK_COLUMN].max()) if RANK_COLUMN in frame.columns and not frame.empty else 1
        frame[SCORE_COLUMN] = (top + 1 - frame[RANK_COLUMN]).astype(float)
    else:
        frame[SCORE_COLUMN] = pd.to_numeric(frame[SCORE_COLUMN], errors="coerce").fillna(0.0)
    return frame.reset_index(drop=True)


def _unseen_relevant(window: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    if window.empty or USER_COLUMN not in window.columns or ITEM_COLUMN not in window.columns:
        return window.iloc[0:0]
    return drop_seen_pairs(window.loc[:, [USER_COLUMN, ITEM_COLUMN]].drop_duplicates(), prev)


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


def _window_events(
    events: pd.DataFrame, generated_at: str | None, event_types: Sequence[str]
) -> pd.DataFrame:
    window = events
    if generated_at and OCCURRED_AT in window.columns:
        start = pd.to_datetime(generated_at, utc=True, errors="coerce")
        if pd.notna(start):
            window = window[window[OCCURRED_AT] > start]
    return filter_events_by_types(window, event_types)


def _recs_at_first_event(
    recs: pd.DataFrame,
    history: pd.DataFrame,
    window_events: pd.DataFrame,
    *,
    generated_at: str | None,
    assigned: Mapping[str, str] | None,
) -> pd.DataFrame:
    combined = history
    if generated_at:
        combined = pd.concat([history, recs.assign(generated_at=generated_at)], ignore_index=True)
    hist_recs = recs_from_history(combined, window_events)
    if not hist_recs.empty:
        hist_recs = _served_recs(hist_recs, assigned)
    return recs if hist_recs.empty else hist_recs


def _metrics_at_ks(
    reco: pd.DataFrame,
    relevant: pd.DataFrame,
    ks: Sequence[int],
    metric_defs: Callable[[int], dict[str, object]],
    failure: str,
    **calc_kwargs: Any,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        if relevant.empty or reco.empty:
            metrics[f"HitRate@{k}"] = _hit_rate(reco, relevant, k=k)
            continue
        reco_k = reco[reco[RANK_COLUMN] <= k] if RANK_COLUMN in reco.columns else reco
        try:
            computed = calc_metrics(
                metric_defs(k),
                reco=reco_k,
                interactions=relevant.assign(weight=1.0),
                **calc_kwargs,
            )
            metrics.update({key: float(value) for key, value in computed.items()})
        except OPTIONAL_EVAL_ERRORS as exc:
            log_caught(f"{failure} k={k}", exc, log=logger)
            metrics[f"HitRate@{k}"] = _hit_rate(reco, relevant, k=k)
    return metrics


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
    recs = _served_recs(recommendations, assigned)
    if recs.empty:
        return None
    used_impressions = impressions is not None and not impressions.empty
    if used_impressions:
        recs = _served_recs(impressions, assigned)
        if recs.empty:
            return None
    all_events = normalized_frame(events)
    if all_events.empty:
        return ServedEvalReport(
            n_users=int(recs[USER_COLUMN].nunique()),
            n_users_with_events=0,
            metrics={},
            by_source={},
            generated_at=generated_at,
        )
    window_events = _window_events(all_events, generated_at, event_types)
    if (
        not used_impressions
        and history is not None
        and not history.empty
        and OCCURRED_AT in window_events.columns
    ):
        recs = _recs_at_first_event(
            recs, history, window_events, generated_at=generated_at, assigned=assigned
        )
    prev = _prev_interactions(all_events, generated_at)
    relevant = _unseen_relevant(window_events, prev)
    if used_impressions and not relevant.empty:
        relevant = relevant.loc[relevant[USER_COLUMN].isin(set(recs[USER_COLUMN]))]
    prev_interactions = None if prev.empty else prev
    metrics = _metrics_at_ks(
        recs,
        relevant,
        ks,
        lambda k: {
            **_ranking_metric_defs(k),
            **_catalog_metric_defs(k, with_prev=prev_interactions is not None),
        },
        "RecTools calc_metrics failed for",
        catalog=_served_catalog(catalog, recs, all_events),
        prev_interactions=prev_interactions,
    )
    by_source: dict[str, dict[str, float]] = {}
    if SOURCE_COLUMN in recs.columns:
        by_source = {
            str(source): _metrics_at_ks(
                group,
                relevant,
                ks,
                lambda k: {f"HitRate@{k}": HitRate(k=k)},
                f"RecTools HitRate failed for source={source}",
            )
            for source, group in recs.groupby(SOURCE_COLUMN, dropna=True)
        }
    return ServedEvalReport(
        n_users=int(recs[USER_COLUMN].nunique()),
        n_users_with_events=int(relevant[USER_COLUMN].nunique()) if not relevant.empty else 0,
        metrics=metrics,
        by_source=by_source,
        generated_at=generated_at,
    )


def recs_from_history(history: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Use each user's newest snapshot with generated_at <= that user's first event."""
    hist = with_str_ids(history)
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
