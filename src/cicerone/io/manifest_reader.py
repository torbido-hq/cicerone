"""Read-only access to job run manifests (status/error/counts per run), for
the dashboard (cicerone.dashboard). Mirrors the two backends in
recommendation_reader.py -- reads back whatever job.run() already wrote via
OutputSink.write_manifest(), never recomputes anything:

  DatasetManifestReader - reads the single manifest.json the dataset output
    sink overwrites on every run. Only ever the latest run.
  DbManifestReader - queries the manifest table directly, so history/trends
    are only available for this backend.

NOTE on upgrading an existing "db" output deployment: job.py always writes
"status"/"error" (and every other manifest key) on every run, including
failures. pandas' to_sql(..., if_exists="append") does not add missing
columns to an already-existing table, so the first write against an older
manifest table will fail with an "unknown column" error until you add the
new columns yourself, e.g.:
  ALTER TABLE recommendation_runs ADD COLUMN status TEXT;
  ALTER TABLE recommendation_runs ADD COLUMN error TEXT;
(or drop/recreate the table -- it's just a run log, not the recommendations
themselves).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from botocore.exceptions import ClientError
from sqlalchemy import MetaData, Table, create_engine, inspect, select

from cicerone.io.db_store import DEFAULT_MANIFEST_TABLE, _sql_identifier
from cicerone.io.options import build_s3_client, require_option

logger = logging.getLogger(__name__)

# S3 codes meaning "the manifest doesn't exist yet"; anything else is a real
# backend/configuration problem and should propagate.
_S3_NOT_FOUND_CODES = {"NoSuchKey", "404"}


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
        prefix = str(self._options.get("prefix", "")).strip("/")
        key = f"{prefix}/manifest.json" if prefix else "manifest.json"
        client = build_s3_client(self._options)
        try:
            obj = client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code in _S3_NOT_FOUND_CODES:
                return None
            raise
        return json.loads(obj["Body"].read())

    def read_latest(self) -> dict[str, Any] | None:
        return self._read()

    def read_recent(self, limit: int) -> list[dict[str, Any]]:
        latest = self._read()
        return [latest] if latest is not None else []


class DbManifestReader:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._table = _sql_identifier(
            options.get("manifest_table", DEFAULT_MANIFEST_TABLE),
            option="manifest_table",
        )
        self._engine = create_engine(require_option(options, "database_url", "db"), pool_pre_ping=True)

    def read_latest(self) -> dict[str, Any] | None:
        rows = self.read_recent(1)
        return rows[0] if rows else None

    def read_recent(self, limit: int) -> list[dict[str, Any]]:
        # No manifest table yet means no runs recorded, not an error.
        if not inspect(self._engine).has_table(self._table):
            return []
        # Reflect via SQLAlchemy Core rather than interpolating `self._table`
        # into raw SQL, so Core quotes/escapes the identifier itself.
        table = Table(self._table, MetaData(), autoload_with=self._engine)
        stmt = select(table).order_by(table.c.generated_at.desc()).limit(limit)
        df = pd.read_sql(stmt, self._engine)
        # NaN (truthy in Python) for columns a pre-upgrade row never had
        # must become None, not NaN.
        df = df.astype(object).where(df.notna(), None)
        return df.to_dict(orient="records")
