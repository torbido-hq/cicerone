"""CTR/CVR attribution from impression and click tracks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd

from cicerone.config.constants import TRACK_KIND_CLICK, TRACK_KIND_IMPRESSION
from cicerone.evaluation.metrics import (
    SliceMetrics,
    _coalesce_column,
    _frame,
    _merge_asof_events,
    _metrics_for_impression_slice,
    _with_join_keys,
)
from cicerone.io.recommendation_schema import (
    ITEM_COLUMN,
    RANK_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
    VARIANT_COLUMN,
)

DEFAULT_CONVERSION_TYPE = "purchase"


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


def filter_events_by_types(frame: pd.DataFrame, event_types: Sequence[str] | None) -> pd.DataFrame:
    if not event_types or frame.empty or "event_type" not in frame.columns:
        return frame
    return frame[frame["event_type"].astype(str).isin(set(event_types))]


def conversion_events(
    events: pd.DataFrame,
    configured: Sequence[str],
    *,
    primary_metric: str,
) -> pd.DataFrame:
    return filter_events_by_types(events, conversion_event_types(configured, primary_metric=primary_metric))


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
    if "generated_at" in recs.columns:
        keep.append("generated_at")
        recs["generated_at"] = pd.to_datetime(recs["generated_at"], utc=True, errors="coerce")
    recs = recs.loc[:, [column for column in keep if column in recs.columns]]
    if "generated_at" in recs.columns:
        recs = recs.sort_values("generated_at", kind="mergesort", na_position="first")
    latest = recs.drop_duplicates(subset=[USER_COLUMN, ITEM_COLUMN], keep="last")
    if "generated_at" in recs.columns and "generated_at" in frame.columns:
        frame["generated_at"] = pd.to_datetime(frame["generated_at"], utc=True, errors="coerce")
        snap = recs.dropna(subset=["generated_at"]).drop_duplicates(
            subset=[USER_COLUMN, ITEM_COLUMN, "generated_at"], keep="last"
        )
        merged = frame.merge(
            snap, on=[USER_COLUMN, ITEM_COLUMN, "generated_at"], how="left", suffixes=("", "_rec")
        )
        _coalesce_column(merged, SOURCE_COLUMN)
        if VARIANT_COLUMN in recs.columns:
            _coalesce_column(merged, VARIANT_COLUMN)
        need_fill = merged["generated_at"].isna()
        if need_fill.any():
            fill = latest.drop(columns=["generated_at"], errors="ignore")
            filled = merged.loc[need_fill].merge(
                fill, on=[USER_COLUMN, ITEM_COLUMN], how="left", suffixes=("", "_latest")
            )
            _coalesce_column(filled, SOURCE_COLUMN)
            if VARIANT_COLUMN in recs.columns:
                _coalesce_column(filled, VARIANT_COLUMN)
            for column in (SOURCE_COLUMN, VARIANT_COLUMN):
                if column in filled.columns:
                    merged.loc[need_fill, column] = filled[column].to_numpy()
        return merged
    merged = frame.merge(latest, on=[USER_COLUMN, ITEM_COLUMN], how="left", suffixes=("", "_rec"))
    _coalesce_column(merged, SOURCE_COLUMN)
    if VARIANT_COLUMN in recs.columns:
        _coalesce_column(merged, VARIANT_COLUMN)
    return merged


@dataclass(frozen=True)
class _ClickFrames:
    impressions: pd.DataFrame
    clicks: pd.DataFrame
    matched_clicks: pd.DataFrame
    window: timedelta


def _prepare_click_frames(
    track_rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    window_hours: float,
    recommendations: pd.DataFrame | None = None,
    annotate_source: bool = False,
) -> _ClickFrames | None:
    rows = _frame(track_rows)
    window = timedelta(hours=float(window_hours))
    if rows.empty or "kind" not in rows.columns:
        return None
    impressions = rows[rows["kind"].astype(str) == TRACK_KIND_IMPRESSION].copy()
    clicks = rows[rows["kind"].astype(str) == TRACK_KIND_CLICK].copy()
    impressions = _with_join_keys(impressions)
    if impressions.empty:
        return None
    clicks = _with_join_keys(clicks)
    if annotate_source:
        impressions = _annotate_source(impressions, recommendations)
        impressions = impressions.copy()
    if "event_id" not in impressions.columns:
        impressions = impressions.copy()
        impressions["event_id"] = [f"imp-{i}" for i in range(len(impressions))]
    if not clicks.empty and "event_id" not in clicks.columns:
        clicks = clicks.copy()
        clicks["event_id"] = [f"clk-{i}" for i in range(len(clicks))]
    matched_clicks = _merge_asof_events(clicks, impressions, window=window) if not clicks.empty else clicks
    return _ClickFrames(impressions, clicks, matched_clicks, window)


def evaluate_tracking(
    *,
    track_rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    conversions: pd.DataFrame,
    recommendations: pd.DataFrame | None = None,
    window_hours: float = 24.0,
) -> TrackEvalReport:
    empty = SliceMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, 0)
    prepared = _prepare_click_frames(
        track_rows,
        window_hours=window_hours,
        recommendations=recommendations,
        annotate_source=True,
    )
    if prepared is None:
        return TrackEvalReport(overall=empty)
    impressions = prepared.impressions
    matched_clicks = prepared.matched_clicks
    window = prepared.window
    conv = _frame(conversions)
    conv = _with_join_keys(conv) if not conv.empty and "event_type" in conv.columns else pd.DataFrame()
    view_conv = _merge_asof_events(conv, impressions, window=window) if not conv.empty else conv
    if not conv.empty and not matched_clicks.empty:
        click_conv = _merge_asof_events(conv, matched_clicks, window=window)
    else:
        click_conv = conv.iloc[0:0]
    overall = _metrics_for_impression_slice(impressions, matched_clicks, view_conv, click_conv)
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


def user_track_outcomes(
    *,
    track_rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    conversions: pd.DataFrame,
    primary_metric: str,
    attribution: str,
    window_hours: float,
) -> dict[str, float]:
    """Per-user CTR or attributed CVR (conversions / impressions) for experiments."""
    report_users: dict[str, float] = {}
    prepared = _prepare_click_frames(track_rows, window_hours=window_hours)
    if prepared is None:
        return report_users
    impressions = prepared.impressions
    matched_clicks = prepared.matched_clicks
    window = prepared.window
    conv = _with_join_keys(_frame(conversions))
    if attribution == "click":
        attributed = (
            _merge_asof_events(conv, matched_clicks, window=window)
            if not conv.empty and not matched_clicks.empty
            else conv.iloc[0:0]
        )
    else:
        attributed = (
            _merge_asof_events(conv, impressions, window=window) if not conv.empty else conv.iloc[0:0]
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
            capped = min(n_conv, float(n_imp))
            report_users[user] = capped / float(n_imp) if n_imp else 0.0
    return report_users
