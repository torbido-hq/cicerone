"""Read-only access to precomputed recommendations for serve mode."""

from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from cicerone.blending import COLD_START_USER_ID, LATEST_SOURCE, POPULAR_SOURCE
from cicerone.io.base import BaseRecommendationReader
from cicerone.io.db_store import (
    DEFAULT_RECOMMENDATION_ITEMS_TABLE,
    DEFAULT_RECOMMENDATIONS_TABLE,
)
from cicerone.io.options import build_s3_client, is_s3_not_found, object_key, require_option, sql_identifier

logger = logging.getLogger(__name__)

USER_COLUMN = "user_id"
RANK_COLUMN = "rank"
SOURCE_COLUMN = "source"
ITEM_COLUMN = "item_id"
ITEMS_SNAPSHOT_FILENAME = "items_snapshot.parquet"

# When __cold_start__ is missing: popular/latest only (never warm "blended").
_FALLBACK_SOURCES = frozenset({POPULAR_SOURCE, LATEST_SOURCE})
# Schema-level only (e.g. missing table). OperationalError stays on the generic path.
_MISSING_TABLE_ERRORS = (ProgrammingError,)


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
            with pd.option_context("future.no_silent_downcasting", True):
                filled = out[column].fillna(False)
            out[column] = filled.infer_objects(copy=False).astype(bool)
    return out


def select_cold_start_fallback(
    recommendations: pd.DataFrame,
    k: int,
    *,
    sentinel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pick ``__cold_start__`` rows, else one popular/latest user's top-K.

    ``sentinel`` may be a pre-fetched ``__cold_start__`` slice (e.g. from SQL);
    when omitted, it is derived from ``recommendations``.
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
    sample_user = candidates[USER_COLUMN].astype(str).iloc[0]
    rows = candidates[candidates[USER_COLUMN].astype(str) == sample_user].sort_values(RANK_COLUMN)
    return rows.head(k).reset_index(drop=True)


class DatasetRecommendationReader(BaseRecommendationReader):
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._backend = options.get("storage_backend", "local")
        self._cache = pd.DataFrame(columns=[USER_COLUMN, RANK_COLUMN, SOURCE_COLUMN])
        self._items: pd.DataFrame | None = None
        self._items_version = 0
        self._category_column: str | None = None
        self._availability_filters: list[str] = []
        self.refresh()

    def configure_item_filters(
        self,
        *,
        category_column: str | None = None,
        availability_filters: Sequence[str] = (),
    ) -> None:
        self._category_column = category_column
        self._availability_filters = list(availability_filters)
        self._items = normalize_items_snapshot(
            self._items,
            category_column=self._category_column,
            availability_filters=self._availability_filters,
        )
        self._items_version += 1

    def _read_recommendations(self) -> pd.DataFrame:
        if self._backend == "local":
            path = Path(require_option(self._options, "path", "local")) / "recommendations.parquet"
            logger.info("Loading recommendations from %s", path)
            return pd.read_parquet(path)

        bucket = require_option(self._options, "bucket", "s3")
        key = object_key(self._options, "recommendations.parquet")
        logger.info("Loading recommendations from s3://%s/%s", bucket, key)
        client = build_s3_client(self._options)
        obj = client.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))

    def _read_items_snapshot(self) -> pd.DataFrame | None:
        try:
            if self._backend == "local":
                path = Path(require_option(self._options, "path", "local")) / ITEMS_SNAPSHOT_FILENAME
                if not path.exists():
                    return None
                logger.info("Loading items snapshot from %s", path)
                return pd.read_parquet(path)

            bucket = require_option(self._options, "bucket", "s3")
            key = object_key(self._options, ITEMS_SNAPSHOT_FILENAME)
            logger.info("Loading items snapshot from s3://%s/%s", bucket, key)
            client = build_s3_client(self._options)
            obj = client.get_object(Bucket=bucket, Key=key)
            return pd.read_parquet(io.BytesIO(obj["Body"].read()))
        except FileNotFoundError:
            return None
        except Exception as exc:
            if is_s3_not_found(exc):
                return None
            logger.exception("Failed to load items snapshot; continuing without item filters")
            return None

    def refresh(self) -> None:
        try:
            self._cache = self._read_recommendations()
        except Exception:
            logger.exception("Failed to refresh recommendations cache; keeping previous data")
        try:
            self._items = normalize_items_snapshot(
                self._read_items_snapshot(),
                category_column=self._category_column,
                availability_filters=self._availability_filters,
            )
            self._items_version += 1
        except Exception:
            logger.exception("Failed to refresh items snapshot; keeping previous data")

    def items_version(self) -> int:
        return self._items_version

    def get_items(self) -> pd.DataFrame | None:
        return self._items

    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame:
        rows = self._cache[self._cache[USER_COLUMN].astype(str) == str(user_id)].sort_values(RANK_COLUMN)
        return rows.head(k).reset_index(drop=True)

    def get_cold_start_fallback(self, k: int) -> pd.DataFrame:
        return select_cold_start_fallback(self._cache, k)


class DbRecommendationReader(BaseRecommendationReader):
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
        self._items: pd.DataFrame | None = None
        self._items_version = 0
        self._category_column: str | None = None
        self._availability_filters: list[str] = []
        self.refresh()

    def configure_item_filters(
        self,
        *,
        category_column: str | None = None,
        availability_filters: Sequence[str] = (),
    ) -> None:
        self._category_column = category_column
        self._availability_filters = list(availability_filters)
        self._items = normalize_items_snapshot(
            self._items,
            category_column=self._category_column,
            availability_filters=self._availability_filters,
        )
        self._items_version += 1

    def refresh(self) -> None:
        try:
            frame = pd.read_sql(text(f'SELECT * FROM "{self._items_table}"'), self._engine)
            self._items = normalize_items_snapshot(
                frame,
                category_column=self._category_column,
                availability_filters=self._availability_filters,
            )
            self._items_version += 1
        except _MISSING_TABLE_ERRORS:
            logger.debug(
                "recommendation items table %r not present; continuing without it",
                self._items_table,
            )
            self._items = None
            self._items_version += 1
        except Exception:
            # Keep the previous snapshot on transient failures (matches dataset reader).
            logger.exception("Failed to refresh recommendation items snapshot; keeping previous data")

    def items_version(self) -> int:
        return self._items_version

    def get_items(self) -> pd.DataFrame | None:
        return self._items

    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame:
        sql = text(
            f'SELECT * FROM "{self._table}" WHERE "{USER_COLUMN}" = :user_id '
            f'ORDER BY "{RANK_COLUMN}" ASC LIMIT :k'
        )
        return pd.read_sql(sql, self._engine, params={"user_id": user_id, "k": k})

    def get_cold_start_fallback(self, k: int) -> pd.DataFrame:
        sentinel = self.get_recommendations(COLD_START_USER_ID, k)
        if not sentinel.empty:
            return sentinel
        sql = text(
            f'SELECT * FROM "{self._table}" WHERE "{SOURCE_COLUMN}" IN '
            f"(:popular, :latest) "
            f'ORDER BY "{USER_COLUMN}" ASC, "{RANK_COLUMN}" ASC LIMIT :k'
        )
        sample = pd.read_sql(
            sql,
            self._engine,
            params={
                "popular": POPULAR_SOURCE,
                "latest": LATEST_SOURCE,
                "k": max(k * 20, k),
            },
        )
        return select_cold_start_fallback(sample, k, sentinel=sentinel)
