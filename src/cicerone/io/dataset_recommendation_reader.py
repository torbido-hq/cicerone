"""Dataset (parquet / S3) recommendation reader."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from cicerone.blending import COLD_START_USER_ID
from cicerone.io import recommendation_schema as _rec
from cicerone.io.base import BaseRecommendationReader
from cicerone.io.options import (
    is_s3_not_found,
    read_parquet,
    require_option,
    validate_storage_options,
)
from cicerone.io.recommendation_reader_common import (
    ITEMS_SNAPSHOT_FILENAME,
    RANK_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
    VARIANT_COLUMN,
    _index_recommendations_by_user,
    _ItemFilterMixin,
    _resolve_fallback_user_id,
    normalize_items_snapshot,
    select_cold_start_fallback,
)
from cicerone.serve.metrics import observe_cache_refresh, record_cache_hit, record_cache_miss

logger = logging.getLogger(__name__)


class DatasetRecommendationReader(_ItemFilterMixin, BaseRecommendationReader):
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._backend = validate_storage_options(options)
        self._cache = pd.DataFrame(columns=[USER_COLUMN, RANK_COLUMN, SOURCE_COLUMN])
        self._by_user: dict[str, pd.DataFrame] = {}
        self._fallback_user_id: str | None = None
        self._variant_names: tuple[str, ...] = ()
        self._init_item_filter_state()
        self.refresh()

    def _read_recommendations(self) -> pd.DataFrame:
        return read_parquet(self._options, "recommendations.parquet")

    def _read_items_snapshot(self) -> pd.DataFrame | None:
        try:
            if self._backend == "local":
                path = Path(require_option(self._options, "path", "local")) / ITEMS_SNAPSHOT_FILENAME
                if not path.exists():
                    return None
            return read_parquet(self._options, ITEMS_SNAPSHOT_FILENAME)
        except FileNotFoundError:
            return None
        except Exception as exc:
            if is_s3_not_found(exc):
                return None
            logger.exception("Failed to load items snapshot; continuing without item filters")
            return None

    def refresh(self) -> None:
        started = time.perf_counter()
        recommendations_ok = False
        try:
            cache = self._read_recommendations()
            by_user = _index_recommendations_by_user(cache)
            fallback_user_id = None
            if COLD_START_USER_ID not in by_user:
                fallback_user_id = _resolve_fallback_user_id(by_user)
            variant_names: tuple[str, ...] = ()
            if VARIANT_COLUMN in cache.columns and not cache.empty:
                variant_names = tuple(
                    sorted(
                        {
                            str(name)
                            for name in cache[VARIANT_COLUMN].tolist()
                            if not pd.isna(name) and str(name)
                        }
                    )
                )
            with self._lock:
                self._cache = cache
                self._by_user = by_user
                self._fallback_user_id = fallback_user_id
                self._variant_names = variant_names
            recommendations_ok = True
        except Exception:
            logger.exception("Failed to refresh recommendations cache; keeping previous data")
        try:
            items = normalize_items_snapshot(
                self._read_items_snapshot(),
                category_column=self._category_column,
                availability_filters=self._availability_filters,
            )
            with self._lock:
                self._items = items
                self._items_version += 1
        except Exception:
            logger.exception("Failed to refresh items snapshot; keeping previous data")
        observe_cache_refresh(duration_seconds=time.perf_counter() - started, success=recommendations_ok)

    def get_recommendations(self, user_id: str, k: int, *, variant: str | None = None) -> pd.DataFrame:
        with self._lock:
            rows = self._by_user.get(str(user_id))
            if rows is None:
                record_cache_miss()
                return self._cache.iloc[0:0]
            record_cache_hit()
            return _rec.filter_variant_rows(rows, variant).head(k).reset_index(drop=True)

    def present_variant_names(self) -> tuple[str, ...]:
        with self._lock:
            return self._variant_names

    def get_cold_start_fallback(self, k: int, *, variant: str | None = None) -> pd.DataFrame:
        with self._lock:
            sentinel = self._by_user.get(COLD_START_USER_ID)
            fallback_id = self._fallback_user_id
            fallback_rows = self._by_user.get(fallback_id) if fallback_id is not None else None
            cache = self._cache
        if sentinel is not None:
            filtered = _rec.filter_variant_rows(sentinel, variant)
            if not filtered.empty:
                return filtered.head(k).reset_index(drop=True)
        if fallback_rows is not None and not fallback_rows.empty:
            filtered = _rec.filter_variant_rows(fallback_rows, variant)
            if not filtered.empty:
                return filtered.head(k).reset_index(drop=True)
        return select_cold_start_fallback(
            _rec.filter_variant_rows(cache, variant),
            k,
            sentinel=_rec.filter_variant_rows(sentinel, variant) if sentinel is not None else None,
        )
