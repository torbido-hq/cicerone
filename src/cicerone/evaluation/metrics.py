"""Shared CTR/CVR metric helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

from cicerone.io.recommendation_schema import ITEM_COLUMN, USER_COLUMN

OCCURRED_AT = "occurred_at"


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


_JOIN_KEYS = (OCCURRED_AT, USER_COLUMN, ITEM_COLUMN)


def _with_join_keys(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or any(column not in frame.columns for column in _JOIN_KEYS):
        return frame.iloc[0:0]
    return frame.dropna(subset=list(_JOIN_KEYS))


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
    n_view = min(int(len(view_conversions)), n_impressions)
    n_click = min(int(len(click_conversions)), n_impressions)
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


def _coalesce_column(frame: pd.DataFrame, name: str) -> None:
    extras = (f"{name}_rec", f"{name}_latest")
    if name not in frame.columns:
        for extra in extras:
            if extra in frame.columns:
                frame[name] = frame[extra]
                return
        return
    for extra in extras:
        if extra in frame.columns:
            frame[name] = frame[name].where(frame[name].notna(), frame[extra])


def _column_ids(frame: pd.DataFrame, column: str) -> set[str]:
    if frame.empty or column not in frame.columns:
        return set()
    return {str(value) for value in frame[column].dropna()}


def _slice_later_events(frame: pd.DataFrame, keys: pd.DataFrame, prior_ids: set[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    if prior_ids and "prior_event_id" in frame.columns:
        matched = frame["prior_event_id"].notna() & frame["prior_event_id"].astype(str).isin(prior_ids)
        return frame.loc[matched]
    if USER_COLUMN not in frame.columns or ITEM_COLUMN not in frame.columns:
        return frame
    pair = keys.loc[:, [USER_COLUMN, ITEM_COLUMN]].drop_duplicates()
    return frame.merge(pair, on=[USER_COLUMN, ITEM_COLUMN], how="inner")


def _metrics_for_impression_slice(
    impressions: pd.DataFrame,
    matched_clicks: pd.DataFrame,
    view_conv: pd.DataFrame,
    click_conv: pd.DataFrame,
) -> SliceMetrics:
    impression_ids = _column_ids(impressions, "event_id")
    clicks = _slice_later_events(matched_clicks, impressions, impression_ids)
    views = _slice_later_events(view_conv, impressions, impression_ids)
    attributed = _slice_later_events(click_conv, clicks, _column_ids(clicks, "event_id"))
    return _slice_metrics(impressions, clicks, views, attributed)
