"""Read-only access to precomputed recommendations for serve mode."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from cicerone.blending import COLD_START_USER_ID, LATEST_SOURCE, POPULAR_SOURCE
from cicerone.io import recommendation_schema as _rec
from cicerone.io.base import BaseRecommendationReader
from cicerone.io.db_store import (
    DEFAULT_RECOMMENDATION_ITEMS_TABLE,
    DEFAULT_RECOMMENDATIONS_TABLE,
    MISSING_TABLE_ERRORS,
)
from cicerone.io.options import (
    build_s3_client,
    is_s3_not_found,
    read_parquet,
    require_option,
    sql_identifier,
    validate_storage_options,
)
from cicerone.serve.metrics import observe_cache_refresh, record_cache_hit, record_cache_miss
from cicerone.values import item_true_mask

logger = logging.getLogger(__name__)

USER_COLUMN = _rec.USER_COLUMN
ITEM_COLUMN = _rec.ITEM_COLUMN
RANK_COLUMN = _rec.RANK_COLUMN
SCORE_COLUMN = _rec.SCORE_COLUMN
SOURCE_COLUMN = _rec.SOURCE_COLUMN
RECOMMENDATION_COLUMNS = _rec.RECOMMENDATION_COLUMNS
ITEMS_SNAPSHOT_FILENAME = "items_snapshot.parquet"

# Cold-start without __cold_start__: popular/latest only (never warm "blended"),
# prefer popular → latest → min user_id. Missing-table: ProgrammingError / OperationalError.
_FALLBACK_SOURCES = frozenset({POPULAR_SOURCE, LATEST_SOURCE})
_FALLBACK_SOURCE_PRIORITY = {POPULAR_SOURCE: 0, LATEST_SOURCE: 1}


def normalize_items_snapshot(
    items: pd.DataFrame | None,
    *,
    category_column: str | None = None,
    availability_filters: Sequence[str] = (),
) -> pd.DataFrame | None:
    """Cast filter columns once so serve requests can reuse the frame as-is."""
    if items is None or items.empty:
        return items
    out = items.copy()
    if ITEM_COLUMN not in out.columns:
        return out
    out[ITEM_COLUMN] = out[ITEM_COLUMN].astype(str)
    if category_column and category_column in out.columns:
        out[category_column] = out[category_column].astype(str)
    for column in availability_filters:
        if column in out.columns:
            out[column] = item_true_mask(out[column])
    return out


def _best_fallback_user_id(priorities: dict[str, int]) -> str | None:
    """Lowest source priority wins; ties break on lexicographically smallest user id."""
    if not priorities:
        return None
    return min(priorities, key=lambda user_id: (priorities[user_id], user_id))


def _pick_fallback_user(candidates: pd.DataFrame) -> str | None:
    """Stable fallback user: prefer popular_fallback, then latest, then min user_id."""
    if candidates.empty or USER_COLUMN not in candidates.columns:
        return None
    frame = candidates[[USER_COLUMN]].copy()
    frame["_user"] = frame[USER_COLUMN].astype(str)
    if SOURCE_COLUMN in candidates.columns:
        frame["_src_pri"] = candidates[SOURCE_COLUMN].map(_FALLBACK_SOURCE_PRIORITY).fillna(99)
    else:
        frame["_src_pri"] = 99
    priorities = frame.groupby("_user", sort=False)["_src_pri"].min().astype(int).to_dict()
    return _best_fallback_user_id(priorities)


def select_cold_start_fallback(
    recommendations: pd.DataFrame,
    k: int,
    *,
    sentinel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pick ``__cold_start__`` rows, else one popular/latest user's top-K.

    Fallback user selection is deterministic across dataset and DB backends:
    prefer ``popular_fallback``, then ``latest``, then lexicographically smallest
    ``user_id``.

    ``sentinel`` may be a pre-fetched ``__cold_start__`` slice (e.g. from SQL
    or the per-user index); when omitted, it is derived from ``recommendations``.
    """
    empty = recommendations.iloc[0:0]
    if k < 1:
        return empty

    if sentinel is None:
        if recommendations.empty or USER_COLUMN not in recommendations.columns:
            return empty
        sentinel = (
            recommendations[recommendations[USER_COLUMN].astype(str) == COLD_START_USER_ID]
            .sort_values(RANK_COLUMN)
            .head(k)
            .reset_index(drop=True)
        )
    elif not sentinel.empty:
        sentinel = sentinel.sort_values(RANK_COLUMN).head(k).reset_index(drop=True)

    if not sentinel.empty:
        return sentinel

    if recommendations.empty or SOURCE_COLUMN not in recommendations.columns:
        return empty
    candidates = recommendations[recommendations[SOURCE_COLUMN].isin(_FALLBACK_SOURCES)]
    if candidates.empty:
        return empty
    sample_user = _pick_fallback_user(candidates)
    if sample_user is None:
        return empty
    rows = candidates[candidates[USER_COLUMN].astype(str) == sample_user].sort_values(RANK_COLUMN)
    return rows.head(k).reset_index(drop=True)


def _index_recommendations_by_user(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Normalize user ids once and build an O(1) lookup for serve requests."""
    if frame.empty or USER_COLUMN not in frame.columns:
        return {}
    indexed = frame.copy()
    indexed[USER_COLUMN] = indexed[USER_COLUMN].astype(str)
    if RANK_COLUMN in indexed.columns:
        indexed = indexed.sort_values(RANK_COLUMN)
    return {
        user_id: group.reset_index(drop=True) for user_id, group in indexed.groupby(USER_COLUMN, sort=False)
    }


def _resolve_fallback_user_id(by_user: dict[str, pd.DataFrame]) -> str | None:
    """Pick popular/latest fallback user from the per-user index (no full-frame scan)."""
    priorities: dict[str, int] = {}
    for user_id, rows in by_user.items():
        if user_id == COLD_START_USER_ID or rows.empty or SOURCE_COLUMN not in rows.columns:
            continue
        sources = set(rows[SOURCE_COLUMN].astype(str))
        if not sources & _FALLBACK_SOURCES:
            continue
        priorities[user_id] = min(_FALLBACK_SOURCE_PRIORITY.get(src, 99) for src in sources)
    return _best_fallback_user_id(priorities)


class _ItemFilterMixin:
    """Shared items-snapshot filter configuration for recommendation readers.

    Call ``_init_item_filter_state()`` from subclass ``__init__`` before
    ``refresh`` / mixin methods. Methods also lazy-init if that was skipped.
    """

    _items: pd.DataFrame | None
    _items_version: int
    _category_column: str | None
    _availability_filters: list[str]
    _lock: threading.RLock

    def _init_item_filter_state(self) -> None:
        self._items = None
        self._items_version = 0
        self._category_column = None
        self._availability_filters = []
        self._lock = threading.RLock()

    def _ensure_item_filter_state(self) -> None:
        if getattr(self, "_lock", None) is None:
            self._init_item_filter_state()

    def configure_item_filters(
        self,
        *,
        category_column: str | None = None,
        availability_filters: Sequence[str] = (),
    ) -> None:
        self._ensure_item_filter_state()
        with self._lock:
            self._category_column = category_column
            self._availability_filters = list(availability_filters)
            self._items = normalize_items_snapshot(
                self._items,
                category_column=self._category_column,
                availability_filters=self._availability_filters,
            )
            self._items_version += 1

    def items_version(self) -> int:
        self._ensure_item_filter_state()
        with self._lock:
            return self._items_version

    def get_items(self) -> pd.DataFrame | None:
        self._ensure_item_filter_state()
        with self._lock:
            return self._items


class DatasetRecommendationReader(_ItemFilterMixin, BaseRecommendationReader):
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._backend = validate_storage_options(options)
        self._cache = pd.DataFrame(columns=[USER_COLUMN, RANK_COLUMN, SOURCE_COLUMN])
        self._by_user: dict[str, pd.DataFrame] = {}
        self._fallback_user_id: str | None = None
        self._init_item_filter_state()
        self._s3_client = None
        self.refresh()

    def _get_s3_client(self):
        if self._s3_client is None:
            self._s3_client = build_s3_client(self._options)
        return self._s3_client

    def _read_recommendations(self) -> pd.DataFrame:
        return read_parquet(
            self._options,
            "recommendations.parquet",
            s3_client=self._get_s3_client() if self._backend == "s3" else None,
        )

    def _read_items_snapshot(self) -> pd.DataFrame | None:
        try:
            if self._backend == "local":
                path = Path(require_option(self._options, "path", "local")) / ITEMS_SNAPSHOT_FILENAME
                if not path.exists():
                    return None
            return read_parquet(
                self._options,
                ITEMS_SNAPSHOT_FILENAME,
                s3_client=self._get_s3_client() if self._backend == "s3" else None,
            )
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
            with self._lock:
                self._cache = cache
                self._by_user = by_user
                self._fallback_user_id = fallback_user_id
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

    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame:
        with self._lock:
            rows = self._by_user.get(str(user_id))
            if rows is None:
                record_cache_miss()
                return self._cache.iloc[0:0]
            record_cache_hit()
            return rows.head(k).reset_index(drop=True)

    def get_cold_start_fallback(self, k: int) -> pd.DataFrame:
        with self._lock:
            sentinel = self._by_user.get(COLD_START_USER_ID)
            if sentinel is not None and not sentinel.empty:
                return sentinel.head(k).reset_index(drop=True)
            if self._fallback_user_id is not None:
                rows = self._by_user.get(self._fallback_user_id)
                if rows is not None and not rows.empty:
                    return rows.head(k).reset_index(drop=True)
            return select_cold_start_fallback(self._cache, k, sentinel=sentinel)


class DbRecommendationReader(_ItemFilterMixin, BaseRecommendationReader):
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._table = sql_identifier(
            options.get("recommendations_table", DEFAULT_RECOMMENDATIONS_TABLE),
            option="recommendations_table",
        )
        self._items_table = sql_identifier(
            options.get("recommendation_items_table", DEFAULT_RECOMMENDATION_ITEMS_TABLE),
            option="recommendation_items_table",
        )
        self._engine = create_engine(require_option(options, "database_url", "db"), pool_pre_ping=True)
        self._init_item_filter_state()
        self.refresh()

    def refresh(self) -> None:
        started = time.perf_counter()
        items_ok = False
        try:
            frame = pd.read_sql(text(f'SELECT * FROM "{self._items_table}"'), self._engine)
            items = normalize_items_snapshot(
                frame,
                category_column=self._category_column,
                availability_filters=self._availability_filters,
            )
            with self._lock:
                self._items = items
                self._items_version += 1
            items_ok = True
        except MISSING_TABLE_ERRORS:
            logger.debug(
                "recommendation items table %r not present; continuing without it",
                self._items_table,
            )
            with self._lock:
                self._items = None
                self._items_version += 1
            items_ok = True
        except Exception:
            logger.exception("Failed to refresh recommendation items snapshot; keeping previous data")
        observe_cache_refresh(duration_seconds=time.perf_counter() - started, success=items_ok)

    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame:
        sql = text(
            f'SELECT * FROM "{self._table}" WHERE "{USER_COLUMN}" = :user_id '
            f'ORDER BY "{RANK_COLUMN}" ASC LIMIT :k'
        )
        rows = pd.read_sql(sql, self._engine, params={"user_id": user_id, "k": k})
        if rows.empty:
            record_cache_miss()
        else:
            record_cache_hit()
        return rows

    def get_cold_start_fallback(self, k: int) -> pd.DataFrame:
        sentinel = self.get_recommendations(COLD_START_USER_ID, k)
        if not sentinel.empty:
            return sentinel
        # Same popular→latest→user_id priority as the in-memory path; full top-k fetch.
        pick_sql = text(
            f'SELECT "{USER_COLUMN}", "{SOURCE_COLUMN}" FROM "{self._table}" '
            f'WHERE "{SOURCE_COLUMN}" IN (:popular, :latest) '
            f"ORDER BY "
            f'CASE "{SOURCE_COLUMN}" '
            f"WHEN :popular THEN 0 WHEN :latest THEN 1 ELSE 99 END, "
            f'"{USER_COLUMN}" ASC '
            f"LIMIT 1"
        )
        try:
            picked = pd.read_sql(
                pick_sql,
                self._engine,
                params={"popular": POPULAR_SOURCE, "latest": LATEST_SOURCE},
            )
        except MISSING_TABLE_ERRORS:
            # Missing table/source column → empty fallback candidates.
            return sentinel
        if picked.empty:
            return sentinel
        sample_user = str(picked.iloc[0][USER_COLUMN])
        return self.get_recommendations(sample_user, k)
