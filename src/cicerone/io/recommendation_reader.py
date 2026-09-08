"""Read-only access to precomputed recommendations for serve mode."""

from __future__ import annotations

from cicerone.io.dataset_recommendation_reader import DatasetRecommendationReader
from cicerone.io.db_recommendation_reader import DbRecommendationReader
from cicerone.io.recommendation_reader_common import (
    _FALLBACK_SOURCE_PRIORITY,
    _FALLBACK_SOURCES,
    ITEM_COLUMN,
    ITEMS_SNAPSHOT_FILENAME,
    RANK_COLUMN,
    RECOMMENDATION_COLUMNS,
    SCORE_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
    VARIANT_COLUMN,
    _best_fallback_user_id,
    _index_recommendations_by_user,
    _ItemFilterMixin,
    _pick_fallback_user,
    _resolve_fallback_user_id,
    normalize_items_snapshot,
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
    "_FALLBACK_SOURCE_PRIORITY",
    "_FALLBACK_SOURCES",
    "_ItemFilterMixin",
    "_best_fallback_user_id",
    "_index_recommendations_by_user",
    "_pick_fallback_user",
    "_resolve_fallback_user_id",
    "normalize_items_snapshot",
    "select_cold_start_fallback",
]
