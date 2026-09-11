"""Catalog-wide popular / latest scores for search-index weights."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from rectools import Columns

from cicerone.dataset import build_interactions
from cicerone.feature_config import FeatureConfig
from cicerone.io.recommendation_schema import ITEM_COLUMN
from cicerone.model_config import LATEST_WINDOW_DAYS
from cicerone.values import is_missing

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


def normalize_item_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate, stringify ids, and sort. Raises ``ValueError`` on bad rows."""
    missing = [column for column in ITEM_SCORES_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"item_scores missing columns {missing}")
    if frame.empty:
        return empty_item_scores()
    out = frame.loc[:, list(ITEM_SCORES_COLUMNS)].copy()
    item_ids = out[ITEM_COLUMN].map(lambda value: "" if is_missing(value) else str(value).strip())
    if bool(item_ids.eq("").any()):
        raise ValueError("item_scores has missing or blank item_id")
    if bool(item_ids.duplicated().any()):
        raise ValueError("item_scores has duplicate item_id")
    out[ITEM_COLUMN] = item_ids
    popular = pd.to_numeric(out[POPULAR_SCORE_COLUMN], errors="coerce")
    latest = pd.to_numeric(out[LATEST_SCORE_COLUMN], errors="coerce")
    n_users = pd.to_numeric(out[N_USERS_COLUMN], errors="coerce")
    if (
        not np.isfinite(popular.to_numpy(dtype="float64")).all()
        or not np.isfinite(latest.to_numpy(dtype="float64")).all()
        or not np.isfinite(n_users.to_numpy(dtype="float64")).all()
        or bool((n_users < 0).any())
        or bool((n_users % 1 != 0).any())
    ):
        raise ValueError("item_scores has non-finite scores, negative n_users, or non-integral n_users")
    out[POPULAR_SCORE_COLUMN] = popular.astype(float)
    out[LATEST_SCORE_COLUMN] = latest.astype(float)
    out[N_USERS_COLUMN] = n_users.astype(int)
    return out.sort_values(ITEM_COLUMN, kind="mergesort").reset_index(drop=True)


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
    """Seek pagination on ``item_id``. ``frame`` must already be id-sorted."""
    if frame.empty:
        return empty_item_scores(), None
    ids = frame[ITEM_COLUMN].astype(str)
    if item_id is not None:
        key = str(item_id)
        start = int(ids.searchsorted(key, side="left"))
        if start < len(ids) and ids.iloc[start] == key:
            return frame.iloc[start : start + 1].reset_index(drop=True), None
        return empty_item_scores(), None
    start = 0 if cursor is None else int(ids.searchsorted(str(cursor), side="right"))
    end = start + limit
    page = frame.iloc[start:end]
    next_cursor = None
    if end < len(frame) and not page.empty:
        next_cursor = str(page.iloc[-1][ITEM_COLUMN])
    return page.reset_index(drop=True), next_cursor
