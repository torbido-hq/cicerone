"""SQL recommendation reader."""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from cicerone.blending import COLD_START_USER_ID, LATEST_SOURCE, POPULAR_SOURCE
from cicerone.io import recommendation_schema as _rec
from cicerone.io.base import BaseRecommendationReader
from cicerone.io.db_errors import db_error_message, is_missing_column_error
from cicerone.io.db_store import (
    DEFAULT_RECOMMENDATION_ITEMS_TABLE,
    DEFAULT_RECOMMENDATIONS_TABLE,
    MISSING_TABLE_ERRORS,
)
from cicerone.io.options import require_option, sql_identifier
from cicerone.io.recommendation_reader_common import (
    RANK_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
    VARIANT_COLUMN,
    _ItemFilterMixin,
    normalize_items_snapshot,
)
from cicerone.serve.metrics import observe_cache_refresh, record_cache_hit, record_cache_miss

logger = logging.getLogger(__name__)


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
        self._variant_supported: bool | None = None
        self._init_item_filter_state()
        self.refresh()

    def refresh(self) -> None:
        self._variant_supported = None
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

    def _supports_variant_column(self) -> bool:
        cached = self._variant_supported
        if cached is not None:
            return cached
        try:
            columns = {col["name"] for col in inspect(self._engine).get_columns(self._table)}
        except Exception:
            logger.exception("Failed to inspect recommendations table %r for variant column", self._table)
            return False
        supported = VARIANT_COLUMN in columns
        self._variant_supported = supported
        return supported

    def _fallback_variant(self, user_id: str) -> str | None:
        sql = text(
            f'SELECT DISTINCT "{VARIANT_COLUMN}" FROM "{self._table}" WHERE "{USER_COLUMN}" = :user_id'
        )
        try:
            frame = pd.read_sql(sql, self._engine, params={"user_id": user_id})
        except Exception as exc:
            if self._remember_missing_variant_column(exc):
                return None
            logger.exception("Failed to list recommendation variants for user_id=%r", user_id)
            return None
        if frame.empty:
            return None
        return _rec.pick_fallback_variant(frame.iloc[:, 0].tolist())

    def _remember_missing_variant_column(self, exc: BaseException) -> bool:
        message = db_error_message(exc)
        if VARIANT_COLUMN not in message:
            return False
        if not is_missing_column_error(exc) and "no such column" not in message:
            return False
        self._variant_supported = False
        return True

    def get_recommendations(self, user_id: str, k: int, *, variant: str | None = None) -> pd.DataFrame:
        if variant is not None and not self._supports_variant_column():
            variant = None
        elif variant is None and self._supports_variant_column():
            variant = self._fallback_variant(user_id)
        if variant is None:
            sql = text(
                f'SELECT * FROM "{self._table}" WHERE "{USER_COLUMN}" = :user_id '
                f'ORDER BY "{RANK_COLUMN}" ASC LIMIT :k'
            )
            params: dict[str, Any] = {"user_id": user_id, "k": k}
        else:
            sql = text(
                f'SELECT * FROM "{self._table}" WHERE "{USER_COLUMN}" = :user_id '
                f'AND "{VARIANT_COLUMN}" = :variant '
                f'ORDER BY "{RANK_COLUMN}" ASC LIMIT :k'
            )
            params = {"user_id": user_id, "k": k, "variant": variant}
        try:
            rows = pd.read_sql(sql, self._engine, params=params)
        except Exception as exc:
            if variant is None:
                raise
            if not self._remember_missing_variant_column(exc):
                raise
            return self.get_recommendations(user_id, k)
        if rows.empty:
            record_cache_miss()
        else:
            record_cache_hit()
        return rows

    def get_cold_start_fallback(self, k: int, *, variant: str | None = None) -> pd.DataFrame:
        if variant is not None and not self._supports_variant_column():
            variant = None
        elif variant is None and self._supports_variant_column():
            variant = self._fallback_variant(COLD_START_USER_ID)
        sentinel = self.get_recommendations(COLD_START_USER_ID, k, variant=variant)
        if not sentinel.empty:
            return sentinel
        if variant is not None and not self._supports_variant_column():
            variant = None
        # Same popular→latest→user_id priority as the in-memory path; full top-k fetch.
        variant_clause = f'AND "{VARIANT_COLUMN}" = :variant ' if variant is not None else ""
        pick_sql = text(
            f'SELECT "{USER_COLUMN}", "{SOURCE_COLUMN}" FROM "{self._table}" '
            f'WHERE "{SOURCE_COLUMN}" IN (:popular, :latest) {variant_clause}'
            f"ORDER BY "
            f'CASE "{SOURCE_COLUMN}" '
            f"WHEN :popular THEN 0 WHEN :latest THEN 1 ELSE 99 END, "
            f'"{USER_COLUMN}" ASC '
            f"LIMIT 1"
        )
        params: dict[str, Any] = {"popular": POPULAR_SOURCE, "latest": LATEST_SOURCE}
        if variant is not None:
            params["variant"] = variant
        try:
            picked = pd.read_sql(pick_sql, self._engine, params=params)
        except Exception as exc:
            if variant is not None and self._remember_missing_variant_column(exc):
                return self.get_cold_start_fallback(k)
            return sentinel
        if picked.empty:
            return sentinel
        sample_user = str(picked.iloc[0][USER_COLUMN])
        return self.get_recommendations(sample_user, k, variant=variant)
