"""Production replay of served recommendation lists."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from rectools.metrics import MAP, NDCG, Recall, calc_metrics

from cicerone.blending import COLD_START_USER_ID
from cicerone.evaluation.metrics import OCCURRED_AT, _frame, _ratio
from cicerone.io.recommendation_schema import (
    ITEM_COLUMN,
    RANK_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
    VARIANT_COLUMN,
)

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
    if assigned and VARIANT_COLUMN in recs.columns:
        expected = recs[USER_COLUMN].map(assigned)
        recs = recs.loc[expected.isna() | (recs[VARIANT_COLUMN].astype(str) == expected)]
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


def evaluate_served(
    recommendations: pd.DataFrame,
    events: pd.DataFrame,
    *,
    generated_at: str | None,
    ks: Sequence[int],
    event_types: Sequence[str],
    history: pd.DataFrame | None = None,
) -> ServedEvalReport | None:
    if recommendations is None or recommendations.empty:
        return None
    recs = recommendations.copy()
    recs[USER_COLUMN] = recs[USER_COLUMN].astype(str)
    recs[ITEM_COLUMN] = recs[ITEM_COLUMN].astype(str)
    recs = recs[recs[USER_COLUMN] != COLD_START_USER_ID]
    if recs.empty:
        return None
    window_events = _frame(events)
    if window_events.empty:
        return ServedEvalReport(
            n_users=int(recs[USER_COLUMN].nunique()),
            n_users_with_events=0,
            metrics={},
            by_source={},
            generated_at=generated_at,
        )
    if generated_at and OCCURRED_AT in window_events.columns:
        start = pd.to_datetime(generated_at, utc=True, errors="coerce")
        if pd.notna(start):
            window_events = window_events[window_events[OCCURRED_AT] > start]
    if event_types and "event_type" in window_events.columns:
        window_events = window_events[window_events["event_type"].astype(str).isin(set(event_types))]
    if history is not None and not history.empty and OCCURRED_AT in window_events.columns:
        live = recs.copy()
        if generated_at:
            live["generated_at"] = generated_at
            combined = pd.concat([history, live], ignore_index=True)
        else:
            combined = history
        hist_recs = _recs_from_history(combined, window_events)
        if not hist_recs.empty:
            hist_recs = hist_recs[hist_recs[USER_COLUMN] != COLD_START_USER_ID]
        if not hist_recs.empty:
            recs = hist_recs
    relevant = window_events.loc[:, [USER_COLUMN, ITEM_COLUMN]].drop_duplicates()
    n_users = int(recs[USER_COLUMN].nunique())
    n_with_events = int(relevant[USER_COLUMN].nunique()) if not relevant.empty else 0
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"HitRate@{k}"] = _hit_rate(recs, relevant, k=k)
        if not relevant.empty and not recs.empty:
            reco_k = recs[recs[RANK_COLUMN] <= k] if RANK_COLUMN in recs.columns else recs
            interactions = relevant.copy()
            interactions["weight"] = 1.0
            try:
                computed = calc_metrics(
                    {
                        f"MAP@{k}": MAP(k=k),
                        f"NDCG@{k}": NDCG(k=k),
                        f"Recall@{k}": Recall(k=k),
                    },
                    reco=reco_k,
                    interactions=interactions,
                )
                metrics.update({key: float(value) for key, value in computed.items()})
            except Exception:
                logger.exception("RecTools calc_metrics failed for k=%s", k)
    by_source: dict[str, dict[str, float]] = {}
    if SOURCE_COLUMN in recs.columns:
        for source, group in recs.groupby(SOURCE_COLUMN, dropna=True):
            source_metrics: dict[str, float] = {}
            for k in ks:
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
