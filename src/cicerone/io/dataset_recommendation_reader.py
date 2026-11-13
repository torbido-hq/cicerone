"""Dataset (parquet / S3) recommendation reader."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from pyarrow.lib import ArrowInvalid, ArrowIOError

from cicerone.blending import COLD_START_USER_ID
from cicerone.io import recommendation_schema as _rec
from cicerone.io.base import BaseRecommendationReader
from cicerone.io.blob import S3_READ_ERRORS, STORAGE_READ_ERRORS, is_storage_not_found
from cicerone.io.filenames import RECOMMENDATIONS_FILENAME
from cicerone.io.options import (
    local_path,
    read_parquet,
    validate_storage_options,
)
from cicerone.io.recommendation_reader_common import (
    ITEMS_SNAPSHOT_FILENAME,
    ItemFilterMixin,
    index_recommendations_by_user,
    item_snapshot_columns,
    narrow_items_frame,
    normalize_items_snapshot,
    resolve_fallback_user_id,
    select_cold_start_fallback,
)
from cicerone.io.recommendation_schema import RANK_COLUMN, SOURCE_COLUMN, USER_COLUMN, VARIANT_COLUMN
from cicerone.item_scores import ITEM_SCORES_FILENAME, normalize_item_scores
from cicerone.metrics import observe_cache_refresh, record_cache_hit, record_cache_miss

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StagedItems:
    items: pd.DataFrame | None
    category: str | None
    availability: tuple[str, ...]
    version: int


_CACHE_IO_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ArrowInvalid,
    ArrowIOError,
    *S3_READ_ERRORS,
)


class DatasetRecommendationReader(ItemFilterMixin, BaseRecommendationReader):
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._backend = validate_storage_options(options)
        self._cache = pd.DataFrame(columns=[USER_COLUMN, RANK_COLUMN, SOURCE_COLUMN])
        self._by_user: dict[str, pd.DataFrame] = {}
        self._fallback_user_id: str | None = None
        self._variant_names: tuple[str, ...] = ()
        self._recommendations_loaded = False
        self._init_item_filter_state()
        self.refresh()

    def _items_columns(
        self,
        *,
        category_column: str | None,
        availability_filters: Sequence[str],
    ) -> list[str] | None:
        wanted = item_snapshot_columns(
            category_column=category_column,
            availability_filters=availability_filters,
        )
        if wanted is None or self._backend != "local":
            return None
        path = local_path(self._options, ITEMS_SNAPSHOT_FILENAME)
        if not path.exists():
            return None
        present = set(pq.read_schema(path).names)
        chosen = [name for name in wanted if name in present]
        return chosen or None

    def _read_recommendations(self) -> pd.DataFrame:
        return read_parquet(self._options, RECOMMENDATIONS_FILENAME)

    def _read_item_scores(self) -> pd.DataFrame:
        if self._backend == "local":
            path = local_path(self._options, ITEM_SCORES_FILENAME)
            if not path.exists():
                raise FileNotFoundError(path)
        return normalize_item_scores(read_parquet(self._options, ITEM_SCORES_FILENAME))

    def _refresh_item_scores(self) -> None:
        try:
            scores = self._read_item_scores()
        except _CACHE_IO_ERRORS + (ValueError,) as exc:
            if is_storage_not_found(exc):
                logger.debug("item_scores not found; keeping previous data")
            else:
                logger.exception("Failed to refresh item scores; keeping previous data")
            return
        with self._lock:
            self._set_item_scores(scores)

    def _read_items_snapshot(
        self,
        *,
        category_column: str | None,
        availability_filters: Sequence[str],
    ) -> pd.DataFrame | None:
        try:
            if self._backend == "local":
                path = local_path(self._options, ITEMS_SNAPSHOT_FILENAME)
                if not path.exists():
                    return None
            frame = read_parquet(
                self._options,
                ITEMS_SNAPSHOT_FILENAME,
                columns=self._items_columns(
                    category_column=category_column,
                    availability_filters=availability_filters,
                ),
            )
            return narrow_items_frame(
                frame,
                item_snapshot_columns(
                    category_column=category_column,
                    availability_filters=availability_filters,
                ),
            )
        except STORAGE_READ_ERRORS as exc:
            if is_storage_not_found(exc):
                return None
            raise

    def _reload_items(self) -> None:
        staged = self._load_items_snapshot()
        with self._lock:
            self._publish_items_locked(staged)

    def _load_items_snapshot(self) -> _StagedItems:
        with self._lock:
            category = self._category_column
            availability = tuple(self._availability_filters)
            version = self._items_version
        items = normalize_items_snapshot(
            self._read_items_snapshot(category_column=category, availability_filters=availability),
            category_column=category,
            availability_filters=availability,
        )
        return _StagedItems(items, category, availability, version)

    def _publish_items_locked(self, staged: _StagedItems) -> None:
        if (
            self._items_version != staged.version
            or self._category_column != staged.category
            or tuple(self._availability_filters) != staged.availability
        ):
            return
        self._items = staged.items
        self._items_version += 1

    def refresh(self) -> bool:
        """Swap recommendations and the items snapshot together; False keeps the previous pair."""
        started = time.perf_counter()
        self._refresh_item_scores()
        try:
            cache = self._read_recommendations()
            by_user = index_recommendations_by_user(cache)
        except _CACHE_IO_ERRORS:
            logger.exception("Failed to refresh recommendations cache; keeping previous data")
            if not self._recommendations_loaded:
                # No recommendations yet, so there is no generation to mix with.
                self._reload_items()
            observe_cache_refresh(duration_seconds=time.perf_counter() - started, success=False)
            return False
        fallback_user_id = None
        if COLD_START_USER_ID not in by_user:
            fallback_user_id = resolve_fallback_user_id(by_user)
        variant_names: tuple[str, ...] = ()
        if VARIANT_COLUMN in cache.columns and not cache.empty:
            names = {str(name) for name in cache[VARIANT_COLUMN].tolist() if not pd.isna(name) and str(name)}
            variant_names = tuple(sorted(names))
        # A hard items error raises here, before either cache changes.
        items = self._load_items_snapshot()
        with self._lock:
            self._cache = cache
            self._by_user = by_user
            self._fallback_user_id = fallback_user_id
            self._variant_names = variant_names
            self._publish_items_locked(items)
            self._recommendations_loaded = True
        observe_cache_refresh(duration_seconds=time.perf_counter() - started, success=True)
        return True

    def get_recommendations(self, user_id: str, k: int, *, variant: str | None = None) -> pd.DataFrame:
        with self._lock:
            rows = self._by_user.get(str(user_id))
            if rows is None:
                record_cache_miss()
                return self._cache.iloc[0:0]
            record_cache_hit()
            return _rec.filter_variant_rows(rows, variant).head(k).reset_index(drop=True)

    def get_recommendations_for_users(
        self, user_ids: Sequence[str], k: int, *, variant: str | None = None
    ) -> dict[str, pd.DataFrame]:
        ids = [str(user_id) for user_id in user_ids]
        with self._lock:
            empty = self._cache.iloc[0:0]
            by_user = self._by_user
            out: dict[str, pd.DataFrame] = {}
            for user_id in ids:
                rows = by_user.get(user_id)
                if rows is None:
                    record_cache_miss()
                    out[user_id] = empty.copy()
                    continue
                record_cache_hit()
                out[user_id] = _rec.filter_variant_rows(rows, variant).head(k).reset_index(drop=True)
        return out

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
