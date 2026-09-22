"""SQL recommendation reader."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import pandas as pd
from sqlalchemy import bindparam, create_engine, inspect, text
from sqlalchemy.sql.elements import TextClause

from cicerone.blending import COLD_START_USER_ID, LATEST_SOURCE, POPULAR_SOURCE
from cicerone.io import recommendation_schema as _rec
from cicerone.io.base import BaseRecommendationReader
from cicerone.io.db_errors import db_error_message, is_missing_column_error
from cicerone.io.db_store import (
    DEFAULT_ITEM_SCORES_TABLE,
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
from cicerone.item_scores import normalize_item_scores
from cicerone.serve.metrics import observe_cache_refresh, record_cache_hit, record_cache_miss

logger = logging.getLogger(__name__)

_RN_COLUMN = "_cicerone_rn"


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
        self._item_scores_table = sql_identifier(
            options.get("item_scores_table", DEFAULT_ITEM_SCORES_TABLE),
            option="item_scores_table",
        )
        self._engine = create_engine(require_option(options, "database_url", "db"), pool_pre_ping=True)
        self._variant_supported: bool | None = None
        self._present_variants: tuple[str, ...] | None = None
        self._init_item_filter_state()
        self.refresh()

    def refresh(self) -> None:
        self._variant_supported = None
        self._present_variants = None
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
        try:
            frame = pd.read_sql(text(f'SELECT * FROM "{self._item_scores_table}"'), self._engine)
            scores = normalize_item_scores(frame)
            with self._lock:
                self._set_item_scores(scores)
        except MISSING_TABLE_ERRORS:
            logger.debug(
                "item_scores table %r not present; keeping previous data",
                self._item_scores_table,
            )
        except Exception:
            logger.exception("Failed to refresh item scores; keeping previous data")
        observe_cache_refresh(duration_seconds=time.perf_counter() - started, success=items_ok)

    def _supports_variant_column(self) -> bool | None:
        cached = self._variant_supported
        if cached is not None:
            return cached
        try:
            columns = {col["name"] for col in inspect(self._engine).get_columns(self._table)}
        except Exception:
            logger.exception("Failed to inspect recommendations table %r for variant column", self._table)
            return None
        supported = VARIANT_COLUMN in columns
        self._variant_supported = supported
        return supported

    def _remember_missing_variant_column(self, exc: BaseException) -> bool:
        message = db_error_message(exc)
        if VARIANT_COLUMN not in message:
            return False
        if not is_missing_column_error(exc) and "no such column" not in message:
            return False
        self._variant_supported = False
        return True

    def _assigned_variant(self, variant: str | None) -> str | None:
        if variant is not None and self._supports_variant_column() is False:
            return None
        return variant

    def present_variant_names(self) -> tuple[str, ...] | None:
        cached = self._present_variants
        if cached is not None:
            return cached
        if self._supports_variant_column() is False:
            self._present_variants = ()
            return ()
        try:
            frame = pd.read_sql(
                text(f'SELECT DISTINCT "{VARIANT_COLUMN}" FROM "{self._table}"'),
                self._engine,
            )
        except Exception as exc:
            if self._remember_missing_variant_column(exc):
                self._present_variants = ()
                return ()
            logger.exception("Failed to list recommendation variants for %r", self._table)
            return None
        names = tuple(sorted({str(value) for value in frame.iloc[:, 0] if value and str(value)}))
        self._present_variants = names
        return names

    def get_recommendations(self, user_id: str, k: int, *, variant: str | None = None) -> pd.DataFrame:
        assigned = self._assigned_variant(variant)
        prefer_fallback = assigned is None and self._supports_variant_column() is not False
        if assigned is not None:
            sql = text(
                f'SELECT * FROM "{self._table}" WHERE "{USER_COLUMN}" = :user_id '
                f'AND "{VARIANT_COLUMN}" = :variant '
                f'ORDER BY "{RANK_COLUMN}" ASC LIMIT :k'
            )
            params: dict[str, Any] = {"user_id": user_id, "k": k, "variant": assigned}
        elif prefer_fallback:
            sql = text(
                f'SELECT * FROM "{self._table}" WHERE "{USER_COLUMN}" = :user_id '
                f'ORDER BY CASE WHEN "{VARIANT_COLUMN}" = :fallback THEN 0 ELSE 1 END, '
                f'"{VARIANT_COLUMN}" ASC, "{RANK_COLUMN}" ASC LIMIT :k'
            )
            params = {"user_id": user_id, "k": k, "fallback": _rec.FALLBACK_VARIANT}
        else:
            sql = text(
                f'SELECT * FROM "{self._table}" WHERE "{USER_COLUMN}" = :user_id '
                f'ORDER BY "{RANK_COLUMN}" ASC LIMIT :k'
            )
            params = {"user_id": user_id, "k": k}
        try:
            rows = pd.read_sql(sql, self._engine, params=params)
        except Exception as exc:
            if assigned is not None:
                if not self._remember_missing_variant_column(exc):
                    raise
                return self.get_recommendations(user_id, k)
            if prefer_fallback and self._remember_missing_variant_column(exc):
                return self.get_recommendations(user_id, k)
            if prefer_fallback:
                logger.exception(
                    "Failed to prefer leftover variant for user %r in %r",
                    user_id,
                    self._table,
                )
                rows = pd.read_sql(
                    text(
                        f'SELECT * FROM "{self._table}" WHERE "{USER_COLUMN}" = :user_id '
                        f'ORDER BY "{RANK_COLUMN}" ASC LIMIT :k'
                    ),
                    self._engine,
                    params={"user_id": user_id, "k": k},
                )
            else:
                raise
        if assigned is None:
            rows = _rec.collapse_mixed_variants(rows)
            if not rows.empty:
                rows = rows.head(k).reset_index(drop=True)
        if rows.empty:
            record_cache_miss()
        else:
            record_cache_hit()
        return rows

    def get_recommendations_for_users(
        self, user_ids: Sequence[str], k: int, *, variant: str | None = None
    ) -> dict[str, pd.DataFrame]:
        ids = [str(user_id) for user_id in user_ids]
        if not ids:
            return {}
        assigned = self._assigned_variant(variant)
        prefer_fallback = assigned is None and self._supports_variant_column() is not False
        if assigned is not None:
            sql = _top_k_per_user_sql(
                self._table,
                where=f'"{USER_COLUMN}" IN :user_ids AND "{VARIANT_COLUMN}" = :variant',
                order_by=f'"{RANK_COLUMN}" ASC',
            )
            params: dict[str, Any] = {"user_ids": ids, "k": k, "variant": assigned}
        elif prefer_fallback:
            sql = _top_k_per_user_sql(
                self._table,
                where=f'"{USER_COLUMN}" IN :user_ids',
                order_by=(
                    f'CASE WHEN "{VARIANT_COLUMN}" = :fallback THEN 0 ELSE 1 END, '
                    f'"{VARIANT_COLUMN}" ASC, "{RANK_COLUMN}" ASC'
                ),
            )
            params = {"user_ids": ids, "k": k, "fallback": _rec.FALLBACK_VARIANT}
        else:
            sql = _top_k_per_user_sql(
                self._table,
                where=f'"{USER_COLUMN}" IN :user_ids',
                order_by=f'"{RANK_COLUMN}" ASC',
            )
            params = {"user_ids": ids, "k": k}
        try:
            rows = _drop_window_rank(pd.read_sql(sql, self._engine, params=params))
        except Exception as exc:
            if assigned is not None:
                if not self._remember_missing_variant_column(exc):
                    raise
                return self.get_recommendations_for_users(ids, k)
            if prefer_fallback and self._remember_missing_variant_column(exc):
                return self.get_recommendations_for_users(ids, k)
            if prefer_fallback:
                logger.exception(
                    "Failed to prefer leftover variant for users %r in %r",
                    ids,
                    self._table,
                )
                rows = _drop_window_rank(
                    pd.read_sql(
                        _top_k_per_user_sql(
                            self._table,
                            where=f'"{USER_COLUMN}" IN :user_ids',
                            order_by=f'"{RANK_COLUMN}" ASC',
                        ),
                        self._engine,
                        params={"user_ids": ids, "k": k},
                    )
                )
            else:
                raise
        return _frames_by_user(rows, ids, k, collapse=assigned is None)

    def get_cold_start_fallback(self, k: int, *, variant: str | None = None) -> pd.DataFrame:
        variant = self._assigned_variant(variant)
        sentinel = self.get_recommendations(COLD_START_USER_ID, k, variant=variant)
        if not sentinel.empty:
            return sentinel
        variant = self._assigned_variant(variant)
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


def _top_k_per_user_sql(table: str, *, where: str, order_by: str) -> TextClause:
    return text(
        f"SELECT * FROM ("
        f'SELECT *, ROW_NUMBER() OVER (PARTITION BY "{USER_COLUMN}" ORDER BY {order_by}) '
        f'AS "{_RN_COLUMN}" FROM "{table}" WHERE {where}'
        f') AS ranked WHERE "{_RN_COLUMN}" <= :k'
    ).bindparams(bindparam("user_ids", expanding=True))


def _drop_window_rank(rows: pd.DataFrame) -> pd.DataFrame:
    if _RN_COLUMN in rows.columns:
        return rows.drop(columns=[_RN_COLUMN])
    return rows


def _frames_by_user(
    rows: pd.DataFrame, user_ids: Sequence[str], k: int, *, collapse: bool
) -> dict[str, pd.DataFrame]:
    empty = rows.iloc[0:0]
    if rows.empty:
        for _user_id in user_ids:
            record_cache_miss()
        return {user_id: empty.copy() for user_id in user_ids}
    frame = rows.copy()
    if USER_COLUMN in frame.columns:
        frame[USER_COLUMN] = frame[USER_COLUMN].astype(str)
    grouped = {str(key): group for key, group in frame.groupby(USER_COLUMN, sort=False)}
    out: dict[str, pd.DataFrame] = {}
    for user_id in user_ids:
        part = grouped.get(user_id)
        if part is None:
            record_cache_miss()
            out[user_id] = empty.copy()
            continue
        if collapse:
            part = _rec.collapse_mixed_variants(part)
        part = part.head(k).reset_index(drop=True)
        if part.empty:
            record_cache_miss()
        else:
            record_cache_hit()
        out[user_id] = part
    return out
