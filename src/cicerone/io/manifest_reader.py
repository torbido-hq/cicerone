"""Read-only access to job run manifests (status/error/counts per run), for
the dashboard (cicerone.dashboard). Mirrors the two backends in
recommendation_reader.py -- reads back whatever job.run() already wrote via
OutputSink.write_manifest(), never recomputes anything:

  DatasetManifestReader - reads the single manifest.json the dataset output
    sink overwrites on every run (S3 or local, see io/dataset_store.py).
    Only ever the latest run -- there is no history for this backend.
  DbManifestReader - queries the manifest table (appended to on every run
    by io/db_store.py's write_manifest()) directly, so real history/trends
    are only available for this backend.

NOTE on upgrading an existing "db" output deployment: job.py now always
writes "status"/"error" (and every other manifest key) on every run,
including failures. pandas' to_sql(..., if_exists="append") does not add
missing columns to an already-existing table, so the first write against an
older manifest table (created before this change) will fail with an
"unknown column" error until you add the new columns yourself, e.g.:
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
from sqlalchemy import create_engine, text

from cicerone.io.options import build_s3_client, require_option

logger = logging.getLogger(__name__)


class DatasetManifestReader:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._backend = options.get("storage_backend", "local")

    def _read(self) -> dict[str, Any] | None:
        try:
            if self._backend == "local":
                path = Path(require_option(self._options, "path", "local")) / "manifest.json"
                if not path.exists():
                    return None
                return json.loads(path.read_text())

            bucket = require_option(self._options, "bucket", "s3")
            prefix = str(self._options.get("prefix", "")).strip("/")
            key = f"{prefix}/manifest.json" if prefix else "manifest.json"
            client = build_s3_client(self._options)
            obj = client.get_object(Bucket=bucket, Key=key)
            return json.loads(obj["Body"].read())
        except Exception:
            logger.exception("Failed to read manifest")
            return None

    def read_latest(self) -> dict[str, Any] | None:
        return self._read()

    def read_recent(self, limit: int) -> list[dict[str, Any]]:
        latest = self._read()
        return [latest] if latest is not None else []


class DbManifestReader:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._table = options.get("manifest_table", "recommendation_runs")
        self._engine = create_engine(require_option(options, "database_url", "db"), pool_pre_ping=True)

    def read_latest(self) -> dict[str, Any] | None:
        rows = self.read_recent(1)
        return rows[0] if rows else None

    def read_recent(self, limit: int) -> list[dict[str, Any]]:
        sql = text(f'SELECT * FROM "{self._table}" ORDER BY "generated_at" DESC LIMIT :limit')
        try:
            df = pd.read_sql(sql, self._engine, params={"limit": limit})
        except Exception:
            logger.exception("Failed to read manifest history from database table %r", self._table)
            return []
        # NaN/NaT for columns a pre-upgrade row never had (see module
        # docstring) must become None, not NaN -- NaN is truthy in Python,
        # so an old run without an "error" column would otherwise render as
        # if it had one.
        df = df.astype(object).where(df.notna(), None)
        return df.to_dict(orient="records")
