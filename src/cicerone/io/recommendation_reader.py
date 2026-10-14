"""Read-only access to precomputed recommendations for serve mode."""

from __future__ import annotations

from typing import Any

from cicerone.deprecation import renamed_attribute
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


_RENAMED = {
    "_FALLBACK_SOURCE_PRIORITY": "cicerone.io.recommendation_reader.FALLBACK_SOURCE_PRIORITY",
    "_FALLBACK_SOURCES": "cicerone.io.recommendation_reader.FALLBACK_SOURCES",
    "_ItemFilterMixin": "cicerone.io.recommendation_reader.ItemFilterMixin",
    "_best_fallback_user_id": "cicerone.io.recommendation_reader.best_fallback_user_id",
    "_index_recommendations_by_user": "cicerone.io.recommendation_reader.index_recommendations_by_user",
    "_pick_fallback_user": "cicerone.io.recommendation_reader.pick_fallback_user",
    "_resolve_fallback_user_id": "cicerone.io.recommendation_reader.resolve_fallback_user_id",
}
__all__ += list(_RENAMED)


def __getattr__(name: str) -> Any:
    return renamed_attribute(__name__, _RENAMED, name)
