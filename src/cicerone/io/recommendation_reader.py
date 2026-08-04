"""Read-only access to precomputed recommendations for serve mode."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from cicerone.blending import COLD_START_USER_ID, LATEST_SOURCE, POPULAR_SOURCE
from cicerone.io.db_store import (
    DEFAULT_RECOMMENDATION_ITEMS_TABLE,
    DEFAULT_RECOMMENDATIONS_TABLE,
)
from cicerone.io.options import build_s3_client, is_s3_not_found, object_key, require_option, sql_identifier

logger = logging.getLogger(__name__)

USER_COLUMN = "user_id"
RANK_COLUMN = "rank"
SOURCE_COLUMN = "source"
ITEMS_SNAPSHOT_FILENAME = "items_snapshot.parquet"

_FALLBACK_SOURCES = frozenset({POPULAR_SOURCE, LATEST_SOURCE, "blended"})


class DatasetRecommendationReader:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._backend = options.get("storage_backend", "local")
        self._cache = pd.DataFrame(columns=[USER_COLUMN, RANK_COLUMN, SOURCE_COLUMN])
        self._items: pd.DataFrame | None = None
        self.refresh()

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
            self._items = self._read_items_snapshot()
        except Exception:
            logger.exception("Failed to refresh items snapshot; keeping previous data")

    def get_items(self) -> pd.DataFrame | None:
        return self._items

    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame:
        rows = self._cache[self._cache[USER_COLUMN].astype(str) == str(user_id)].sort_values(RANK_COLUMN)
        return rows.head(k).reset_index(drop=True)

    def get_cold_start_fallback(self, k: int) -> pd.DataFrame:
        """Reuse a precomputed popular/latest list when ``user_id`` is unknown."""
        sentinel = self.get_recommendations(COLD_START_USER_ID, k)
        if not sentinel.empty:
            return sentinel
        if self._cache.empty or SOURCE_COLUMN not in self._cache.columns:
            return sentinel
        candidates = self._cache[self._cache[SOURCE_COLUMN].isin(_FALLBACK_SOURCES)]
        if candidates.empty:
            return sentinel
        sample_user = candidates[USER_COLUMN].astype(str).iloc[0]
        rows = candidates[candidates[USER_COLUMN].astype(str) == sample_user].sort_values(RANK_COLUMN)
        return rows.head(k).reset_index(drop=True)


class DbRecommendationReader:
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
        self.refresh()

    def refresh(self) -> None:
        try:
            from sqlalchemy import inspect

            if not inspect(self._engine).has_table(self._items_table):
                self._items = None
                return
            self._items = pd.read_sql(text(f'SELECT * FROM "{self._items_table}"'), self._engine)
        except Exception:
            logger.exception("Failed to refresh recommendation items snapshot; continuing without it")
            self._items = None

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
            f"(:popular, :latest, :blended) "
            f'ORDER BY "{USER_COLUMN}" ASC, "{RANK_COLUMN}" ASC LIMIT :k'
        )
        # Pull a small window then keep one user's rows.
        sample = pd.read_sql(
            sql,
            self._engine,
            params={
                "popular": POPULAR_SOURCE,
                "latest": LATEST_SOURCE,
                "blended": "blended",
                "k": max(k * 20, k),
            },
        )
        if sample.empty:
            return sample
        sample_user = sample[USER_COLUMN].astype(str).iloc[0]
        rows = sample[sample[USER_COLUMN].astype(str) == sample_user].sort_values(RANK_COLUMN)
        return rows.head(k).reset_index(drop=True)
