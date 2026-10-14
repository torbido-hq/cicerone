"""Read-only access to precomputed recommendations for serve mode."""

from __future__ import annotations

from cicerone.io.dataset_recommendation_reader import DatasetRecommendationReader
from cicerone.io.db_recommendation_reader import DbRecommendationReader
from cicerone.io.recommendation_reader_common import (
    FALLBACK_SOURCE_PRIORITY,
    FALLBACK_SOURCES,
    ITEM_COLUMN,
    ITEMS_SNAPSHOT_FILENAME,
    RANK_COLUMN,
    RECOMMENDATION_COLUMNS,
    SCORE_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
    VARIANT_COLUMN,
    ItemFilterMixin,
    best_fallback_user_id,
    index_recommendations_by_user,
    normalize_items_snapshot,
    pick_fallback_user,
    resolve_fallback_user_id,
    select_cold_start_fallback,
)

__all__ = [
    "ITEM_COLUMN",
    "ITEMS_SNAPSHOT_FILENAME",
    "RANK_COLUMN",
    "RECOMMENDATION_COLUMNS",
    "SCORE_COLUMN",
    "SOURCE_COLUMN",
    "USER_COLUMN",
    "VARIANT_COLUMN",
    "DatasetRecommendationReader",
    "DbRecommendationReader",
    "FALLBACK_SOURCE_PRIORITY",
    "FALLBACK_SOURCES",
    "ItemFilterMixin",
    "best_fallback_user_id",
    "index_recommendations_by_user",
    "pick_fallback_user",
    "resolve_fallback_user_id",
    "normalize_items_snapshot",
    "select_cold_start_fallback",
]
