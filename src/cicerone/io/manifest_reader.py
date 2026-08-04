"""Read-only access to job run manifests for the dashboard.

DatasetManifestReader — latest manifest.json only.
DbManifestReader — history from the manifest table.

NOTE: upgrading an existing db output may need ALTER TABLE for new
manifest columns (status/error/…); pandas to_sql(append) will not add them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, inspect, select

from cicerone.io.db_store import DEFAULT_MANIFEST_TABLE
from cicerone.io.options import build_s3_client, is_s3_not_found, object_key, require_option, sql_identifier

logger = logging.getLogger(__name__)


class DatasetManifestReader:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._backend = options.get("storage_backend", "local")

    def _read(self) -> dict[str, Any] | None:
        if self._backend == "local":
            path = Path(require_option(self._options, "path", "local")) / "manifest.json"
            if not path.exists():
                return None
            return json.loads(path.read_text())

        bucket = require_option(self._options, "bucket", "s3")
        key = object_key(self._options, "manifest.json")
        client = build_s3_client(self._options)
        try:
            obj = client.get_object(Bucket=bucket, Key=key)
        except Exception as exc:
            if is_s3_not_found(exc):
                return None
            raise
        return json.loads(obj["Body"].read())

    def read_latest(self) -> dict[str, Any] | None:
        return self._read()

    def read_recent(self, limit: int) -> list[dict[str, Any]]:
        del limit  # dataset backend only ever has the latest run
        latest = self._read()
        return [latest] if latest is not None else []


class DbManifestReader:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._table = sql_identifier(
            options.get("manifest_table", DEFAULT_MANIFEST_TABLE),
            option="manifest_table",
        )
        self._engine = create_engine(require_option(options, "database_url", "db"), pool_pre_ping=True)

    def read_latest(self) -> dict[str, Any] | None:
        rows = self.read_recent(1)
        return rows[0] if rows else None

    def read_recent(self, limit: int) -> list[dict[str, Any]]:
        if not inspect(self._engine).has_table(self._table):
            return []
        table = Table(self._table, MetaData(), autoload_with=self._engine)
        stmt = select(table).order_by(table.c.generated_at.desc()).limit(limit)
        df = pd.read_sql(stmt, self._engine)
        df = df.astype(object).where(df.notna(), None)
        return df.to_dict(orient="records")
