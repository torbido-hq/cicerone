"""Catalog-wide popular / latest scores for search-index weights."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from rectools import Columns

from cicerone.dataset import build_interactions
from cicerone.feature_config import FeatureConfig
from cicerone.io.recommendation_schema import ITEM_COLUMN
from cicerone.model_config import LATEST_WINDOW_DAYS

POPULAR_SCORE_COLUMN = "popular_score"
LATEST_SCORE_COLUMN = "latest_score"
N_USERS_COLUMN = "n_users"
ITEM_SCORES_COLUMNS: tuple[str, ...] = (
    ITEM_COLUMN,
    POPULAR_SCORE_COLUMN,
    LATEST_SCORE_COLUMN,
    N_USERS_COLUMN,
)
ITEM_SCORES_FILENAME = "item_scores.parquet"


def empty_item_scores() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ITEM_SCORES_COLUMNS))


def _item_weight_sum(interactions: pd.DataFrame) -> pd.Series:
    if interactions.empty or Columns.Item not in interactions.columns:
        return pd.Series(dtype="float64")
    weights = interactions[Columns.Weight] if Columns.Weight in interactions.columns else 0.0
    return interactions.assign(_w=weights).groupby(Columns.Item, sort=False)["_w"].sum()


def _item_n_users(interactions: pd.DataFrame) -> pd.Series:
    if interactions.empty or Columns.Item not in interactions.columns:
        return pd.Series(dtype="int64")
    if Columns.User not in interactions.columns:
        return pd.Series(0, index=_item_weight_sum(interactions).index, dtype="int64")
    return interactions.groupby(Columns.Item, sort=False)[Columns.User].nunique()


def build_item_scores(
    events: pd.DataFrame,
    items: pd.DataFrame | None,
    config: FeatureConfig,
    half_life_days: float,
    *,
    interactions: pd.DataFrame | None = None,
    latest_window_days: int = LATEST_WINDOW_DAYS,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Full-catalog scores from the same weighted interactions as training."""
    popular = interactions if interactions is not None else build_interactions(events, config, half_life_days)
    stamp = now if now is not None else datetime.now(UTC)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    window_start = pd.Timestamp(stamp) - pd.Timedelta(days=latest_window_days)
    if events.empty or "occurred_at" not in events.columns:
        latest = popular.iloc[0:0]
    else:
        occurred = pd.to_datetime(events["occurred_at"], utc=True)
        latest_events = events.loc[occurred >= window_start]
        latest = (
            build_interactions(latest_events, config, half_life_days)
            if not latest_events.empty
            else popular.iloc[0:0]
        )

    popular_sum = _item_weight_sum(popular)
    latest_sum = _item_weight_sum(latest)
    n_users = _item_n_users(popular)
    popular_sum.index = popular_sum.index.astype(str)
    latest_sum.index = latest_sum.index.astype(str)
    n_users.index = n_users.index.astype(str)

    catalog: set[str] = set()
    if items is not None and not items.empty and ITEM_COLUMN in items.columns:
        catalog.update(items[ITEM_COLUMN].dropna().astype(str))
    catalog.update(popular_sum.index.astype(str))
    catalog.update(latest_sum.index.astype(str))
    if not catalog:
        return empty_item_scores()

    ids = sorted(catalog)
    frame = pd.DataFrame({ITEM_COLUMN: ids})
    frame[POPULAR_SCORE_COLUMN] = frame[ITEM_COLUMN].map(popular_sum.astype(float)).fillna(0.0)
    frame[LATEST_SCORE_COLUMN] = frame[ITEM_COLUMN].map(latest_sum.astype(float)).fillna(0.0)
    frame[N_USERS_COLUMN] = frame[ITEM_COLUMN].map(n_users).fillna(0).astype(int)
    return frame.loc[:, list(ITEM_SCORES_COLUMNS)].reset_index(drop=True)


def page_item_scores(
    frame: pd.DataFrame,
    *,
    limit: int,
    cursor: str | None = None,
    item_id: str | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Seek pagination on ``item_id``. ``cursor`` is the last returned id."""
    if frame.empty:
        return empty_item_scores(), None
    rows = frame.copy()
    rows[ITEM_COLUMN] = rows[ITEM_COLUMN].astype(str)
    if item_id is not None:
        matched = rows.loc[rows[ITEM_COLUMN] == str(item_id), list(ITEM_SCORES_COLUMNS)]
        return matched.reset_index(drop=True), None
    rows = rows.sort_values(ITEM_COLUMN, kind="mergesort")
    if cursor:
        rows = rows.loc[rows[ITEM_COLUMN] > str(cursor)]
    window = rows.head(limit + 1)
    page = window.head(limit)
    next_cursor = None
    if len(window) > limit and not page.empty:
        next_cursor = str(page.iloc[-1][ITEM_COLUMN])
    return page.reset_index(drop=True), next_cursor
