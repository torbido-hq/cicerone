"""Read-only access to precomputed recommendations, for the serve mode
(cicerone.serve). Deliberately independent of cicerone.model: serving reads
whatever the batch job already wrote to the output store, it never loads or
runs a model, so a serve-only deployment doesn't need lightfm/rectools/
implicit installed.

Mirrors the two output backends in cicerone.io ("dataset" and "db"):

  DatasetRecommendationReader - the whole recommendations.parquet file (S3 or
    local) is cached in memory and filtered per request; call refresh() to
    reload it (cicerone.serve does this on a background timer).
  DbRecommendationReader - queries the recommendations table directly on
    every call (already indexed by user_id at the database level), so
    refresh() is a no-op.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from cicerone.io.options import build_s3_client, require_option

logger = logging.getLogger(__name__)

USER_COLUMN = "user_id"
RANK_COLUMN = "rank"


class DatasetRecommendationReader:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._backend = options.get("storage_backend", "local")
        self._cache = pd.DataFrame(columns=[USER_COLUMN, RANK_COLUMN])
        self.refresh()

    def _read(self) -> pd.DataFrame:
        if self._backend == "local":
            path = Path(require_option(self._options, "path", "local")) / "recommendations.parquet"
            logger.info("Loading recommendations from %s", path)
            return pd.read_parquet(path)

        bucket = require_option(self._options, "bucket", "s3")
        prefix = str(self._options.get("prefix", "")).strip("/")
        key = f"{prefix}/recommendations.parquet" if prefix else "recommendations.parquet"
        logger.info("Loading recommendations from s3://%s/%s", bucket, key)
        client = build_s3_client(self._options)
        obj = client.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))

    def refresh(self) -> None:
        try:
            self._cache = self._read()
        except Exception:
            logger.exception("Failed to refresh recommendations cache; keeping previous data")

    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame:
        rows = self._cache[self._cache[USER_COLUMN] == user_id].sort_values(RANK_COLUMN)
        return rows.head(k).reset_index(drop=True)


class DbRecommendationReader:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._table = options.get("recommendations_table", "recommendations")
        self._engine = create_engine(require_option(options, "database_url", "db"), pool_pre_ping=True)

    def refresh(self) -> None:
        pass

    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame:
        sql = text(
            f'SELECT * FROM "{self._table}" WHERE "{USER_COLUMN}" = :user_id '
            f'ORDER BY "{RANK_COLUMN}" ASC LIMIT :k'
        )
        return pd.read_sql(sql, self._engine, params={"user_id": user_id, "k": k})
