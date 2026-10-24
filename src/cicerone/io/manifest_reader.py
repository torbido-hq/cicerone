"""Read-only access to job run manifests for the dashboard.

DatasetManifestReader — latest manifest.json only.
DbManifestReader — history from the manifest table.

NOTE: upgrading an existing db output may need ALTER TABLE for new
manifest columns (status/error/…); pandas to_sql(append) will not add them.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from cicerone.io.blob import read_storage_bytes
from cicerone.io.filenames import MANIFEST_FILENAME
from cicerone.io.options import require_option, table_option, validate_storage_options

logger = logging.getLogger(__name__)


class DatasetManifestReader:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        validate_storage_options(options)

    def _read(self) -> dict[str, Any] | None:
        payload = read_storage_bytes(self._options, MANIFEST_FILENAME)
        return None if payload is None else json.loads(payload)

    def read_latest(self) -> dict[str, Any] | None:
        return self._read()

    def read_recent(self, limit: int) -> list[dict[str, Any]]:
        del limit  # dataset backend only ever has the latest run
        latest = self._read()
        return [latest] if latest is not None else []


class DbManifestReader:
    def __init__(self, options: dict[str, Any]):
        from cicerone.io.db_store import DEFAULT_MANIFEST_TABLE
        from cicerone.io.engines import engine_for

        self._options = options
        self._table = table_option(options, "manifest_table", DEFAULT_MANIFEST_TABLE)
        self._engine = engine_for(require_option(options, "database_url", "db"))

    def read_latest(self) -> dict[str, Any] | None:
        rows = self.read_recent(1)
        return rows[0] if rows else None

    def read_recent(self, limit: int) -> list[dict[str, Any]]:
        from sqlalchemy import MetaData, Table, inspect, select

        if not inspect(self._engine).has_table(self._table):
            return []
        table = Table(self._table, MetaData(), autoload_with=self._engine)
        stmt = select(table).order_by(table.c.generated_at.desc()).limit(limit)
        # Driver rows, not read_sql: pandas turns an integer column with a NULL into float64 (5 -> 5.0).
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [{key: None if _is_nan(value) else value for key, value in row.items()} for row in rows]


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)
