"""Database backend for TrackStore."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
from sqlalchemy import Engine, bindparam, text

from cicerone.io.engines import LazyEngine, create_table_sql, insert_sql, quote_identifier
from cicerone.io.options import require_option, table_option
from cicerone.io.sql_rows import append_frame_fenced, frame_records, read_sql_or_missing, replace_rows_fenced
from cicerone.locks import WriteFence
from cicerone.track.store_common import (
    DEFAULT_EVAL_TABLE,
    DEFAULT_HISTORY_TABLE,
    DEFAULT_TRACK_TABLE,
    empty_history_frame,
    filter_history,
    history_sql_filter,
    track_row_sql_filter,
    unique_matching_rows,
)

_TRACK_TABLE = (
    ("event_id", "TEXT PRIMARY KEY"),
    ("kind", "TEXT NOT NULL"),
    ("user_id", "TEXT NOT NULL"),
    ("item_id", "TEXT NOT NULL"),
    ("rank", "INTEGER"),
    ("occurred_at", "TEXT NOT NULL"),
    ("variant", "TEXT"),
    ("experiment_id", "TEXT"),
    ("generated_at", "TEXT"),
)
_EVAL_TABLE = (("payload", "TEXT NOT NULL"), ("written_at", "TEXT NOT NULL"))


class TrackDbBackend:
    """Track rows, eval report and history in ``[output]`` tables."""

    def __init__(self, options: dict[str, Any], fence: WriteFence):
        self._options = options
        self._fence = fence
        self._engine = LazyEngine(lambda: require_option(options, "database_url", "db"))
        self._append_lock = threading.Lock()

    def append_accepted(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with self._append_lock, self._fence.lease():
            return self._append_rows_db(rows)

    def read_rows(
        self, *, kind: str | None, experiment_id: str | None, since: str | None
    ) -> list[dict[str, Any]]:
        rows = self._read_rows_db(kind=kind, experiment_id=experiment_id, since=since)
        return unique_matching_rows(rows, kind=kind, experiment_id=experiment_id, since=since)

    def eval_writer(self, payload: dict[str, Any]) -> Callable[[], None]:
        encoded = json.dumps(payload)
        return lambda: self._write_eval_db(encoded)

    def read_eval(self) -> dict[str, Any] | None:
        return self._read_eval_db()

    def history_writer(self, frame: pd.DataFrame, generated_at: str) -> Callable[[], None]:
        del generated_at
        return lambda: self._append_history_db(frame)

    def read_history(self, *, generated_ats: set[str] | None, since: str | None) -> pd.DataFrame:
        frame = self._read_history_db(generated_ats=generated_ats)
        return filter_history(frame, generated_ats=generated_ats, since=since)

    def engine(self) -> Engine:
        return self._engine.get()

    def _ensure_track_table(self, conn: Any, table: str) -> None:
        conn.execute(text(create_table_sql(conn, table, _TRACK_TABLE)))

    def _append_rows_db(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        table = table_option(self._options, "track_table", DEFAULT_TRACK_TABLE)
        engine = self.engine()
        row_sql = text(
            insert_sql(engine, table, [name for name, _kind in _TRACK_TABLE])
            + f" ON CONFLICT ({quote_identifier(engine, 'event_id')}) DO NOTHING"
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
            self._fence.ensure()
            conn.execute(row_sql, fresh)
            self._fence.ensure()
            return fresh

    def _read_rows_db(
        self,
        *,
        kind: str | None,
        experiment_id: str | None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        table = table_option(self._options, "track_table", DEFAULT_TRACK_TABLE)
        engine = self.engine()
        quoted = quote_identifier(engine, table)
        clause, params = track_row_sql_filter(kind=kind, experiment_id=experiment_id, since=since)
        frame = read_sql_or_missing(engine, text(f"SELECT * FROM {quoted}{clause}"), table, params)
        return frame_records(frame)

    def _write_eval_db(self, encoded: str) -> None:
        table = table_option(self._options, "eval_table", DEFAULT_EVAL_TABLE)
        engine = self.engine()
        with engine.begin() as conn:
            conn.execute(text(create_table_sql(conn, table, _EVAL_TABLE)))
            written_at = pd.Timestamp.now(tz="UTC").isoformat()
            replace_rows_fenced(conn, table, {"payload": encoded, "written_at": written_at}, self._fence)

    def _read_eval_db(self) -> dict[str, Any] | None:
        table = table_option(self._options, "eval_table", DEFAULT_EVAL_TABLE)
        engine = self.engine()
        quoted = quote_identifier(engine, table)
        frame = read_sql_or_missing(
            engine, text(f"SELECT {quote_identifier(engine, 'payload')} FROM {quoted} LIMIT 1"), table
        )
        if frame is None or frame.empty:
            return None
        raw = frame.iloc[0]["payload"]
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _append_history_db(self, frame: pd.DataFrame) -> None:
        table = table_option(self._options, "history_table", DEFAULT_HISTORY_TABLE)
        append_frame_fenced(self.engine(), table, frame, self._fence)

    def _read_history_db(
        self,
        *,
        generated_ats: set[str] | None,
    ) -> pd.DataFrame:
        table = table_option(self._options, "history_table", DEFAULT_HISTORY_TABLE)
        engine = self.engine()
        quoted = quote_identifier(engine, table)
        clause, params = history_sql_filter(generated_ats=generated_ats)
        stmt = text(f"SELECT * FROM {quoted}{clause}")
        if generated_ats:
            stmt = stmt.bindparams(bindparam("generated_ats", expanding=True))
        frame = read_sql_or_missing(engine, stmt, table, params)
        return empty_history_frame() if frame is None else frame


def _existing_event_ids(conn: Any, table: str, event_ids: Sequence[str]) -> set[str]:
    ids = [event_id for event_id in event_ids if event_id]
    if not ids:
        return set()
    event_id, quoted_table = quote_identifier(conn, "event_id"), quote_identifier(conn, table)
    stmt = text(f"SELECT {event_id} FROM {quoted_table} WHERE {event_id} IN :ids").bindparams(
        bindparam("ids", expanding=True)
    )
    return {str(row[0]) for row in conn.execute(stmt, {"ids": ids}) if row[0] is not None}
