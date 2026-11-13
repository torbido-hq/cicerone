"""SQL recommendation reader."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import pandas as pd
from sqlalchemy import bindparam, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.elements import TextClause

from cicerone.blending import COLD_START_USER_ID, LATEST_SOURCE, POPULAR_SOURCE
from cicerone.io import recommendation_schema as _rec
from cicerone.io.base import BaseRecommendationReader
from cicerone.io.db_errors import (
    MISSING_TABLE_ERRORS,
    SQL_READ_ERRORS,
    db_error_message,
    is_missing_column_error,
)
from cicerone.io.db_store import (
    DEFAULT_ITEM_SCORES_TABLE,
    DEFAULT_RECOMMENDATION_ITEMS_TABLE,
    DEFAULT_RECOMMENDATIONS_TABLE,
)
from cicerone.io.engines import engine_for, quote_identifier
from cicerone.io.options import require_option, sql_identifier, table_option
from cicerone.io.recommendation_reader_common import (
    ItemFilterMixin,
    item_snapshot_columns,
    normalize_items_snapshot,
)
from cicerone.io.recommendation_schema import RANK_COLUMN, SOURCE_COLUMN, USER_COLUMN, VARIANT_COLUMN
from cicerone.item_scores import normalize_item_scores
from cicerone.metrics import observe_cache_refresh, record_cache_hit, record_cache_miss

logger = logging.getLogger(__name__)

_RN_COLUMN = "_cicerone_rn"


class DbRecommendationReader(ItemFilterMixin, BaseRecommendationReader):
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._table = table_option(options, "recommendations_table", DEFAULT_RECOMMENDATIONS_TABLE)
        self._items_table = table_option(
            options, "recommendation_items_table", DEFAULT_RECOMMENDATION_ITEMS_TABLE
        )
        self._item_scores_table = table_option(options, "item_scores_table", DEFAULT_ITEM_SCORES_TABLE)
        self._engine = engine_for(require_option(options, "database_url", "db"))
        self._sql = {
            name: quote_identifier(self._engine, name)
            for name in (
                self._table,
                self._items_table,
                self._item_scores_table,
                _RN_COLUMN,
                USER_COLUMN,
                VARIANT_COLUMN,
                RANK_COLUMN,
                SOURCE_COLUMN,
            )
        }
        self._variant_supported: bool | None = None
        self._present_variants: tuple[str, ...] | None = None
        self._init_item_filter_state()
        self.refresh()

    def _items_select(self):
        wanted = item_snapshot_columns(
            category_column=self._category_column,
            availability_filters=self._availability_filters,
        )
        if wanted is None:
            return text(f"SELECT * FROM {self._sql[self._items_table]}")
        try:
            existing = {col["name"] for col in inspect(self._engine).get_columns(self._items_table)}
        except SQLAlchemyError:
            return text(f"SELECT * FROM {self._sql[self._items_table]}")
        chosen = [name for name in wanted if name in existing]
        if not chosen:
            return text(f"SELECT * FROM {self._sql[self._items_table]}")
        quoted = ", ".join(
            quote_identifier(self._engine, sql_identifier(name, option="items snapshot column"))
            for name in chosen
        )
        return text(f"SELECT {quoted} FROM {self._sql[self._items_table]}")

    def _reload_items(self) -> bool:
        try:
            frame = pd.read_sql(self._items_select(), self._engine)
            items = normalize_items_snapshot(
                frame,
                category_column=self._category_column,
                availability_filters=self._availability_filters,
            )
            with self._lock:
                self._items = items
                self._items_version += 1
            return True
        except MISSING_TABLE_ERRORS:
            logger.debug(
                "recommendation items table %r not present; continuing without it",
                self._items_table,
            )
            with self._lock:
                self._items = None
                self._items_version += 1
            return True
        except SQL_READ_ERRORS:
            logger.exception("Failed to refresh recommendation items snapshot; keeping previous data")
            return False

    def refresh(self) -> bool:
        """False when the items snapshot could not be reloaded and the previous one was kept."""
        self._variant_supported = None
        self._present_variants = None
        started = time.perf_counter()
        items_ok = self._reload_items()
        self._refresh_item_scores()
        observe_cache_refresh(duration_seconds=time.perf_counter() - started, success=items_ok)
        return items_ok

    def _refresh_item_scores(self) -> None:
        try:
            frame = pd.read_sql(text(f"SELECT * FROM {self._sql[self._item_scores_table]}"), self._engine)
            scores = normalize_item_scores(frame)
        except MISSING_TABLE_ERRORS:
            logger.debug("item_scores table %r not present; keeping previous data", self._item_scores_table)
            return
        except SQL_READ_ERRORS + (ValueError,):
            logger.exception("Failed to refresh item scores; keeping previous data")
            return
        with self._lock:
            self._set_item_scores(scores)

    def _supports_variant_column(self) -> bool | None:
        cached = self._variant_supported
        if cached is not None:
            return cached
        try:
            columns = {col["name"] for col in inspect(self._engine).get_columns(self._table)}
        except SQLAlchemyError:
            logger.exception("Failed to inspect recommendations table %r for variant column", self._table)
            return None
        supported = VARIANT_COLUMN in columns
        self._variant_supported = supported
        return supported

    def _remember_missing_variant_column(self, exc: BaseException) -> bool:
        message = db_error_message(exc)
        if VARIANT_COLUMN not in message:
            return False
        if not is_missing_column_error(exc):
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
                text(f"SELECT DISTINCT {self._sql[VARIANT_COLUMN]} FROM {self._sql[self._table]}"),
                self._engine,
            )
        except SQL_READ_ERRORS as exc:
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
                f"SELECT * FROM {self._sql[self._table]} WHERE {self._sql[USER_COLUMN]} = :user_id "
                f"AND {self._sql[VARIANT_COLUMN]} = :variant "
                f"ORDER BY {self._sql[RANK_COLUMN]} ASC LIMIT :k"
            )
            params: dict[str, Any] = {"user_id": user_id, "k": k, "variant": assigned}
        elif prefer_fallback:
            sql = text(
                f"SELECT * FROM {self._sql[self._table]} WHERE {self._sql[USER_COLUMN]} = :user_id "
                f"ORDER BY CASE WHEN {self._sql[VARIANT_COLUMN]} = :fallback THEN 0 ELSE 1 END, "
                f"{self._sql[VARIANT_COLUMN]} ASC, {self._sql[RANK_COLUMN]} ASC LIMIT :k"
            )
            params = {"user_id": user_id, "k": k, "fallback": _rec.CONTROL_NAME}
        else:
            sql = text(
                f"SELECT * FROM {self._sql[self._table]} WHERE {self._sql[USER_COLUMN]} = :user_id "
                f"ORDER BY {self._sql[RANK_COLUMN]} ASC LIMIT :k"
            )
            params = {"user_id": user_id, "k": k}
        try:
            rows = pd.read_sql(sql, self._engine, params=params)
        except SQL_READ_ERRORS as exc:
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
                        f"SELECT * FROM {self._sql[self._table]} WHERE {self._sql[USER_COLUMN]} = :user_id "
                        f"ORDER BY {self._sql[RANK_COLUMN]} ASC LIMIT :k"
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
        user_in = f"{self._sql[USER_COLUMN]} IN :user_ids"
        by_rank = f"{self._sql[RANK_COLUMN]} ASC"
        if assigned is not None:
            sql = self._top_k_per_user_sql(
                where=f"{user_in} AND {self._sql[VARIANT_COLUMN]} = :variant", order_by=by_rank
            )
            params: dict[str, Any] = {"user_ids": ids, "k": k, "variant": assigned}
        elif prefer_fallback:
            sql = self._top_k_per_user_sql(
                where=user_in,
                order_by=(
                    f"CASE WHEN {self._sql[VARIANT_COLUMN]} = :fallback THEN 0 ELSE 1 END, "
                    f"{self._sql[VARIANT_COLUMN]} ASC, {by_rank}"
                ),
            )
            params = {"user_ids": ids, "k": k, "fallback": _rec.CONTROL_NAME}
        else:
            sql = self._top_k_per_user_sql(where=user_in, order_by=by_rank)
            params = {"user_ids": ids, "k": k}
        try:
            rows = _drop_window_rank(pd.read_sql(sql, self._engine, params=params))
        except SQL_READ_ERRORS as exc:
            if assigned is not None:
                if not self._remember_missing_variant_column(exc):
                    raise
                return self.get_recommendations_for_users(ids, k)
            if prefer_fallback and self._remember_missing_variant_column(exc):
                return self.get_recommendations_for_users(ids, k)
            if not prefer_fallback:
                raise
            logger.exception("Failed to prefer leftover variant for users %r in %r", ids, self._table)
            rows = _drop_window_rank(
                pd.read_sql(
                    self._top_k_per_user_sql(where=user_in, order_by=by_rank),
                    self._engine,
                    params={"user_ids": ids, "k": k},
                )
            )
        return _frames_by_user(rows, ids, k, collapse=assigned is None)

    def _top_k_per_user_sql(self, *, where: str, order_by: str) -> TextClause:
        rn = self._sql[_RN_COLUMN]
        return text(
            f"SELECT * FROM ("
            f"SELECT *, ROW_NUMBER() OVER (PARTITION BY {self._sql[USER_COLUMN]} ORDER BY {order_by}) "
            f"AS {rn} FROM {self._sql[self._table]} WHERE {where}"
            f") AS ranked WHERE {rn} <= :k "
            f"ORDER BY {self._sql[USER_COLUMN]} ASC, {rn} ASC"
        ).bindparams(bindparam("user_ids", expanding=True))

    def get_cold_start_fallback(self, k: int, *, variant: str | None = None) -> pd.DataFrame:
        variant = self._assigned_variant(variant)
        sentinel = self.get_recommendations(COLD_START_USER_ID, k, variant=variant)
        if not sentinel.empty:
            return sentinel
        variant = self._assigned_variant(variant)
        variant_clause = f"AND {self._sql[VARIANT_COLUMN]} = :variant " if variant is not None else ""
        pick_sql = text(
            f"SELECT {self._sql[USER_COLUMN]}, {self._sql[SOURCE_COLUMN]} FROM {self._sql[self._table]} "
            f"WHERE {self._sql[SOURCE_COLUMN]} IN (:popular, :latest) {variant_clause}"
            f"ORDER BY "
            f"CASE {self._sql[SOURCE_COLUMN]} "
            f"WHEN :popular THEN 0 WHEN :latest THEN 1 ELSE 99 END, "
            f"{self._sql[USER_COLUMN]} ASC "
            f"LIMIT 1"
        )
        params: dict[str, Any] = {"popular": POPULAR_SOURCE, "latest": LATEST_SOURCE}
        if variant is not None:
            params["variant"] = variant
        try:
            picked = pd.read_sql(pick_sql, self._engine, params=params)
        except SQL_READ_ERRORS as exc:
            if variant is not None and self._remember_missing_variant_column(exc):
                return self.get_cold_start_fallback(k)
            return sentinel
        if picked.empty:
            return sentinel
        sample_user = str(picked.iloc[0][USER_COLUMN])
        return self.get_recommendations(sample_user, k, variant=variant)


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
        if RANK_COLUMN in part.columns:
            part = part.sort_values(RANK_COLUMN, kind="mergesort")
        part = part.head(k).reset_index(drop=True)
        if part.empty:
            record_cache_miss()
        else:
            record_cache_hit()
        out[user_id] = part
    return out
