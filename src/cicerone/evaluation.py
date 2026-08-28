"""CTR/CVR attribution and production replay of served lists."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd
from rectools.metrics import MAP, NDCG, Recall, calc_metrics

from cicerone.blending import COLD_START_USER_ID
from cicerone.config.constants import TRACK_KIND_CLICK, TRACK_KIND_IMPRESSION
from cicerone.io.recommendation_schema import (
    ITEM_COLUMN,
    RANK_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
    VARIANT_COLUMN,
)

logger = logging.getLogger(__name__)

OCCURRED_AT = "occurred_at"
DEFAULT_CONVERSION_TYPE = "purchase"


@dataclass(frozen=True)
class SliceMetrics:
    n_impressions: int
    n_clicks: int
    n_conversions_click: int
    n_conversions_view: int
    ctr: float
    cvr_click: float
    cvr_view: float
    n_users: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_impressions": self.n_impressions,
            "n_clicks": self.n_clicks,
            "n_conversions_click": self.n_conversions_click,
            "n_conversions_view": self.n_conversions_view,
            "ctr": self.ctr,
            "cvr_click": self.cvr_click,
            "cvr_view": self.cvr_view,
            "n_users": self.n_users,
        }


@dataclass(frozen=True)
class TrackEvalReport:
    overall: SliceMetrics
    by_rank: dict[str, SliceMetrics] = field(default_factory=dict)
    by_source: dict[str, SliceMetrics] = field(default_factory=dict)
    by_variant: dict[str, SliceMetrics] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.as_dict(),
            "by_rank": {key: value.as_dict() for key, value in sorted(self.by_rank.items())},
            "by_source": {key: value.as_dict() for key, value in sorted(self.by_source.items())},
            "by_variant": {key: value.as_dict() for key, value in sorted(self.by_variant.items())},
        }


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


def conversion_event_types(
    configured: Sequence[str],
    *,
    primary_metric: str,
) -> tuple[str, ...]:
    if configured:
        return tuple(configured)
    if primary_metric in {"weighted", "ctr", "conversion"}:
        return (DEFAULT_CONVERSION_TYPE,)
    return (primary_metric,)


def replay_ks(configured: Sequence[int], *, top_k: int) -> tuple[int, ...]:
    if configured:
        return tuple(sorted({int(value) for value in configured if 1 <= int(value) <= top_k})) or (top_k,)
    values = {top_k, *[k for k in (5, 10) if k <= top_k]}
    return tuple(sorted(values))


def _frame(rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame()
    if USER_COLUMN in frame.columns:
        frame[USER_COLUMN] = frame[USER_COLUMN].astype(str)
    if ITEM_COLUMN in frame.columns:
        frame[ITEM_COLUMN] = frame[ITEM_COLUMN].astype(str)
    if OCCURRED_AT in frame.columns:
        frame[OCCURRED_AT] = pd.to_datetime(frame[OCCURRED_AT], utc=True, errors="coerce")
    return frame


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _slice_metrics(
    impressions: pd.DataFrame,
    matched_clicks: pd.DataFrame,
    view_conversions: pd.DataFrame,
    click_conversions: pd.DataFrame,
) -> SliceMetrics:
    n_impressions = int(len(impressions))
    n_clicks = int(len(matched_clicks))
    n_view = int(len(view_conversions))
    n_click = int(len(click_conversions))
    users = set()
    if not impressions.empty and USER_COLUMN in impressions.columns:
        users.update(impressions[USER_COLUMN].astype(str))
    return SliceMetrics(
        n_impressions=n_impressions,
        n_clicks=n_clicks,
        n_conversions_click=n_click,
        n_conversions_view=n_view,
        ctr=_ratio(n_clicks, n_impressions),
        cvr_click=_ratio(n_click, n_impressions),
        cvr_view=_ratio(n_view, n_impressions),
        n_users=len(users),
    )


def _merge_asof_events(
    later: pd.DataFrame,
    earlier: pd.DataFrame,
    *,
    window: timedelta,
) -> pd.DataFrame:
    if later.empty or earlier.empty:
        return later.iloc[0:0].copy()
    left = later.copy()
    keep = [USER_COLUMN, ITEM_COLUMN, OCCURRED_AT]
    if "event_id" in earlier.columns:
        keep.append("event_id")
    right = earlier.loc[:, [column for column in keep if column in earlier.columns]].copy()
    if "event_id" not in right.columns:
        right["event_id"] = [f"prior-{i}" for i in range(len(right))]
    right = right.rename(columns={OCCURRED_AT: "prior_at", "event_id": "prior_event_id"})
    left = left.sort_values(OCCURRED_AT)
    right = right.sort_values("prior_at")
    merged = pd.merge_asof(
        left,
        right,
        by=[USER_COLUMN, ITEM_COLUMN],
        left_on=OCCURRED_AT,
        right_on="prior_at",
        direction="backward",
    )
    if "prior_at" not in merged.columns:
        return later.iloc[0:0].copy()
    delta = merged[OCCURRED_AT] - merged["prior_at"]
    matched = merged["prior_event_id"].notna() & delta.notna()
    matched &= delta >= pd.Timedelta(0)
    matched &= delta <= window
    return merged.loc[matched]


def _annotate_source(impressions: pd.DataFrame, recommendations: pd.DataFrame | None) -> pd.DataFrame:
    if impressions.empty:
        return impressions
    frame = impressions.copy()
    if recommendations is None or recommendations.empty or SOURCE_COLUMN not in recommendations.columns:
        if SOURCE_COLUMN not in frame.columns:
            frame[SOURCE_COLUMN] = None
        return frame
    recs = recommendations.copy()
    recs[USER_COLUMN] = recs[USER_COLUMN].astype(str)
    recs[ITEM_COLUMN] = recs[ITEM_COLUMN].astype(str)
    keep = [USER_COLUMN, ITEM_COLUMN, SOURCE_COLUMN]
    if VARIANT_COLUMN in recs.columns:
        keep.append(VARIANT_COLUMN)
    recs = recs.loc[:, [column for column in keep if column in recs.columns]].drop_duplicates(
        subset=[USER_COLUMN, ITEM_COLUMN], keep="first"
    )
    merged = frame.merge(recs, on=[USER_COLUMN, ITEM_COLUMN], how="left", suffixes=("", "_rec"))
    if SOURCE_COLUMN not in merged.columns and f"{SOURCE_COLUMN}_rec" in merged.columns:
        merged[SOURCE_COLUMN] = merged[f"{SOURCE_COLUMN}_rec"]
    elif f"{SOURCE_COLUMN}_rec" in merged.columns:
        merged[SOURCE_COLUMN] = merged[SOURCE_COLUMN].fillna(merged[f"{SOURCE_COLUMN}_rec"])
    return merged


def evaluate_tracking(
    *,
    track_rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    conversions: pd.DataFrame,
    recommendations: pd.DataFrame | None = None,
    window_hours: float = 24.0,
) -> TrackEvalReport:
    rows = _frame(track_rows)
    window = timedelta(hours=float(window_hours))
    empty = SliceMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, 0)
    if rows.empty or "kind" not in rows.columns:
        return TrackEvalReport(overall=empty)
    impressions = rows[rows["kind"].astype(str) == TRACK_KIND_IMPRESSION].copy()
    clicks = rows[rows["kind"].astype(str) == TRACK_KIND_CLICK].copy()
    if OCCURRED_AT not in impressions.columns:
        return TrackEvalReport(overall=empty)
    impressions = impressions.dropna(subset=[OCCURRED_AT, USER_COLUMN, ITEM_COLUMN])
    clicks = clicks.dropna(subset=[OCCURRED_AT, USER_COLUMN, ITEM_COLUMN]) if not clicks.empty else clicks
    impressions = _annotate_source(impressions, recommendations)
    impressions = impressions.copy()
    if "event_id" not in impressions.columns:
        impressions["event_id"] = [f"imp-{i}" for i in range(len(impressions))]
    if not clicks.empty and "event_id" not in clicks.columns:
        clicks = clicks.copy()
        clicks["event_id"] = [f"clk-{i}" for i in range(len(clicks))]
    matched_clicks = _merge_asof_events(clicks, impressions, window=window) if not clicks.empty else clicks
    conv = _frame(conversions)
    if not conv.empty and "event_type" in conv.columns:
        conv = conv.dropna(subset=[OCCURRED_AT, USER_COLUMN, ITEM_COLUMN])
    else:
        conv = pd.DataFrame()
    view_conv = _merge_asof_events(conv, impressions, window=window) if not conv.empty else conv
    click_base = matched_clicks if not matched_clicks.empty else clicks
    if not conv.empty and not click_base.empty:
        click_conv = _merge_asof_events(conv, click_base, window=window)
    else:
        click_conv = conv.iloc[0:0]
    overall = _slice_metrics(impressions, matched_clicks, view_conv, click_conv)
    by_rank: dict[str, SliceMetrics] = {}
    if RANK_COLUMN in impressions.columns:
        for rank, group in impressions.groupby(RANK_COLUMN, dropna=True):
            key = str(int(rank)) if float(rank).is_integer() else str(rank)
            by_rank[key] = _metrics_for_impression_slice(group, matched_clicks, view_conv, click_conv)
    by_source: dict[str, SliceMetrics] = {}
    if SOURCE_COLUMN in impressions.columns:
        for source, group in impressions.groupby(SOURCE_COLUMN, dropna=True):
            by_source[str(source)] = _metrics_for_impression_slice(
                group, matched_clicks, view_conv, click_conv
            )
    by_variant: dict[str, SliceMetrics] = {}
    if VARIANT_COLUMN in impressions.columns:
        for variant, group in impressions.groupby(VARIANT_COLUMN, dropna=True):
            by_variant[str(variant)] = _metrics_for_impression_slice(
                group, matched_clicks, view_conv, click_conv
            )
    return TrackEvalReport(overall=overall, by_rank=by_rank, by_source=by_source, by_variant=by_variant)


def _metrics_for_impression_slice(
    impressions: pd.DataFrame,
    matched_clicks: pd.DataFrame,
    view_conv: pd.DataFrame,
    click_conv: pd.DataFrame,
) -> SliceMetrics:
    keys = impressions.loc[:, [USER_COLUMN, ITEM_COLUMN]].drop_duplicates()

    def _subset(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or USER_COLUMN not in frame.columns or ITEM_COLUMN not in frame.columns:
            return frame
        return frame.merge(keys, on=[USER_COLUMN, ITEM_COLUMN], how="inner")

    return _slice_metrics(impressions, _subset(matched_clicks), _subset(view_conv), _subset(click_conv))


def user_track_outcomes(
    *,
    track_rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    conversions: pd.DataFrame,
    primary_metric: str,
    attribution: str,
    window_hours: float,
) -> dict[str, float]:
    """Per-user CTR or attributed conversion counts for experiment assignment."""
    report_users: dict[str, float] = {}
    rows = _frame(track_rows)
    if rows.empty or "kind" not in rows.columns:
        return report_users
    impressions = rows[rows["kind"].astype(str) == TRACK_KIND_IMPRESSION]
    clicks = rows[rows["kind"].astype(str) == TRACK_KIND_CLICK]
    window = timedelta(hours=float(window_hours))
    if impressions.empty:
        return report_users
    impressions = impressions.dropna(subset=[OCCURRED_AT, USER_COLUMN, ITEM_COLUMN])
    if "event_id" not in impressions.columns:
        impressions = impressions.copy()
        impressions["event_id"] = [f"imp-{i}" for i in range(len(impressions))]
    if not clicks.empty and "event_id" not in clicks.columns:
        clicks = clicks.copy()
        clicks["event_id"] = [f"clk-{i}" for i in range(len(clicks))]
    matched_clicks = (
        _merge_asof_events(clicks.dropna(subset=[OCCURRED_AT]), impressions, window=window)
        if not clicks.empty
        else clicks
    )
    conv = _frame(conversions)
    if attribution == "click":
        attributed = (
            _merge_asof_events(conv.dropna(subset=[OCCURRED_AT]), matched_clicks, window=window)
            if not conv.empty and not matched_clicks.empty
            else conv.iloc[0:0]
        )
    else:
        attributed = (
            _merge_asof_events(conv.dropna(subset=[OCCURRED_AT]), impressions, window=window)
            if not conv.empty
            else conv.iloc[0:0]
        )
    impression_counts = impressions.groupby(USER_COLUMN).size()
    click_counts = (
        matched_clicks.groupby(USER_COLUMN).size() if not matched_clicks.empty else pd.Series(dtype=int)
    )
    conversion_counts = (
        attributed.groupby(USER_COLUMN).size() if not attributed.empty else pd.Series(dtype=int)
    )
    for user_id, n_imp in impression_counts.items():
        user = str(user_id)
        n_clicks = float(click_counts.get(user, 0)) if not click_counts.empty else 0.0
        n_conv = float(conversion_counts.get(user, 0)) if not conversion_counts.empty else 0.0
        if primary_metric == "ctr":
            report_users[user] = n_clicks / float(n_imp) if n_imp else 0.0
        else:
            report_users[user] = n_conv
    return report_users


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
