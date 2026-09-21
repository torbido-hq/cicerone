"""Database backend for TrackStore."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from typing import Any

import pandas as pd
from sqlalchemy import Engine, bindparam, create_engine, text

from cicerone.io.db_errors import is_missing_table_error
from cicerone.io.options import require_option, sql_identifier
from cicerone.track.store_common import (
    DEFAULT_EVAL_TABLE,
    DEFAULT_HISTORY_TABLE,
    DEFAULT_TRACK_TABLE,
    HISTORY_COLUMNS,
    _history_sql_filter,
    _jsonish,
    _track_row_sql_filter,
)

logger = logging.getLogger(__name__)


class TrackDbBackend:
    _engine: Engine | None
    _engine_lock: threading.Lock
    _options: dict[str, Any]

    def _ensure_fence(self) -> None:
        return None

    def _db_engine(self) -> Engine:
        engine = self._engine
        if engine is not None:
            return engine
        with self._engine_lock:
            if self._engine is None:
                self._engine = create_engine(
                    require_option(self._options, "database_url", "db"), pool_pre_ping=True
                )
            return self._engine

    def _ensure_track_table(self, conn: Any, table: str) -> None:
        conn.execute(
            text(
                f'CREATE TABLE IF NOT EXISTS "{table}" ('
                "event_id TEXT PRIMARY KEY, "
                "kind TEXT NOT NULL, "
                "user_id TEXT NOT NULL, "
                "item_id TEXT NOT NULL, "
                "rank INTEGER, "
                "occurred_at TEXT NOT NULL, "
                "variant TEXT, "
                "experiment_id TEXT, "
                "generated_at TEXT"
                ")"
            )
        )

    def _append_rows_db(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        table = sql_identifier(
            self._options.get("track_table", DEFAULT_TRACK_TABLE),
            option="track_table",
        )
        engine = self._db_engine()
        insert_sql = text(
            f'INSERT INTO "{table}" (event_id, kind, user_id, item_id, rank, occurred_at, '
            "variant, experiment_id, generated_at) "
            "VALUES (:event_id, :kind, :user_id, :item_id, :rank, :occurred_at, "
            ":variant, :experiment_id, :generated_at) "
            "ON CONFLICT (event_id) DO NOTHING"
        )
        params: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            event_id = str(row.get("event_id") or "")
            if event_id and event_id in seen:
                continue
            if event_id:
                seen.add(event_id)
            params.append(
                {
                    "event_id": event_id,
                    "kind": str(row.get("kind") or ""),
                    "user_id": str(row.get("user_id") or ""),
                    "item_id": str(row.get("item_id") or ""),
                    "rank": row.get("rank"),
                    "occurred_at": str(row.get("occurred_at") or ""),
                    "variant": row.get("variant"),
                    "experiment_id": row.get("experiment_id"),
                    "generated_at": row.get("generated_at"),
                }
            )
        with engine.begin() as conn:
            self._ensure_track_table(conn, table)
            existing = _existing_event_ids(conn, table, [row["event_id"] for row in params])
            fresh = [row for row in params if not row["event_id"] or row["event_id"] not in existing]
            if not fresh:
                return []
            self._ensure_fence()
            conn.execute(insert_sql, fresh)
            self._ensure_fence()
            return fresh

    def _read_rows_db(
        self,
        *,
        kind: str | None,
        experiment_id: str | None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        table = sql_identifier(
            self._options.get("track_table", DEFAULT_TRACK_TABLE),
            option="track_table",
        )
        engine = self._db_engine()
        clause, params = _track_row_sql_filter(kind=kind, experiment_id=experiment_id, since=since)
        try:
            frame = pd.read_sql(text(f'SELECT * FROM "{table}"{clause}'), engine, params=params)
        except Exception as exc:
            if is_missing_table_error(exc):
                return []
            logger.exception("Failed to read track table %r", table)
            raise
        if frame.empty:
            return []
        records = frame.to_dict(orient="records")
        return [{str(key): _jsonish(value) for key, value in row.items()} for row in records]

    def _write_eval_db(self, payload: dict[str, Any]) -> None:
        table = sql_identifier(
            self._options.get("eval_table", DEFAULT_EVAL_TABLE),
            option="eval_table",
        )
        engine = self._db_engine()
        encoded = json.dumps(payload)
        with engine.begin() as conn:
            conn.execute(
                text(
                    f'CREATE TABLE IF NOT EXISTS "{table}" (payload TEXT NOT NULL, written_at TEXT NOT NULL)'
                )
            )
            conn.execute(text(f'DELETE FROM "{table}"'))
            self._ensure_fence()
            conn.execute(
                text(f'INSERT INTO "{table}" (payload, written_at) VALUES (:payload, :written_at)'),
                {"payload": encoded, "written_at": pd.Timestamp.now(tz="UTC").isoformat()},
            )
            self._ensure_fence()

    def _read_eval_db(self) -> dict[str, Any] | None:
        table = sql_identifier(
            self._options.get("eval_table", DEFAULT_EVAL_TABLE),
            option="eval_table",
        )
        engine = self._db_engine()
        try:
            frame = pd.read_sql(text(f'SELECT payload FROM "{table}" LIMIT 1'), engine)
        except Exception as exc:
            if is_missing_table_error(exc):
                return None
            logger.exception("Failed to read eval table %r", table)
            raise
        if frame.empty:
            return None
        raw = frame.iloc[0]["payload"]
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _append_history_db(self, frame: pd.DataFrame) -> None:
        table = sql_identifier(
            self._options.get("history_table", DEFAULT_HISTORY_TABLE),
            option="history_table",
        )
        with self._db_engine().begin() as conn:
            self._ensure_fence()
            frame.to_sql(table, conn, if_exists="append", index=False)
            self._ensure_fence()

    def _read_history_db(
        self,
        *,
        generated_ats: set[str] | None,
    ) -> pd.DataFrame:
        table = sql_identifier(
            self._options.get("history_table", DEFAULT_HISTORY_TABLE),
            option="history_table",
        )
        engine = self._db_engine()
        clause, params = _history_sql_filter(generated_ats=generated_ats)
        try:
            if params:
                stmt = text(f'SELECT * FROM "{table}"{clause}')
                if generated_ats:
                    stmt = stmt.bindparams(bindparam("generated_ats", expanding=True))
                return pd.read_sql(stmt, engine, params=params)
            return pd.read_sql(text(f'SELECT * FROM "{table}"'), engine)
        except Exception as exc:
            if is_missing_table_error(exc):
                return pd.DataFrame(columns=list(HISTORY_COLUMNS))
            logger.exception("Failed to read history table %r", table)
            raise


def _existing_event_ids(conn: Any, table: str, event_ids: Sequence[str]) -> set[str]:
    ids = [event_id for event_id in event_ids if event_id]
    if not ids:
        return set()
    stmt = text(f'SELECT event_id FROM "{table}" WHERE event_id IN :ids').bindparams(
        bindparam("ids", expanding=True)
    )
    return {str(row[0]) for row in conn.execute(stmt, {"ids": ids}) if row[0] is not None}
