"""Database backend for TrackStore."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any

import pandas as pd
from sqlalchemy import Column, Engine, Insert, Integer, MetaData, String, Table, Text, bindparam, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.schema import CreateTable

from cicerone.io.engines import LazyEngine, create_table_sql, is_mysql_family, quote_identifier
from cicerone.io.options import require_option, table_option
from cicerone.io.sql_rows import append_frame_fenced, frame_records, read_sql_or_missing, replace_rows_fenced
from cicerone.locks import WriteFence
from cicerone.track.normalize import TrackNormalizeError
from cicerone.track.store_common import (
    DEFAULT_EVAL_TABLE,
    DEFAULT_HISTORY_TABLE,
    DEFAULT_TRACK_TABLE,
    empty_history_frame,
    filter_history,
    history_sql_filter,
    track_row_sql_filter,
    unique_event_rows,
    unique_matching_rows,
)

_MYSQL_EVENT_ID_CHARS = 255
# MySQL can't index plain TEXT; a binary collation keeps ids that differ only by case or accent distinct.
_EVENT_ID_TYPE = Text().with_variant(
    String(_MYSQL_EVENT_ID_CHARS, collation="utf8mb4_bin"), "mysql", "mariadb"
)
_TEXT_DEFAULTED = frozenset({"event_id", "kind", "user_id", "item_id", "occurred_at"})
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

    @cached_property
    def _track_table(self) -> Table:
        return Table(
            table_option(self._options, "track_table", DEFAULT_TRACK_TABLE),
            MetaData(),
            Column("event_id", _EVENT_ID_TYPE, primary_key=True),
            Column("kind", Text, nullable=False),
            Column("user_id", Text, nullable=False),
            Column("item_id", Text, nullable=False),
            Column("rank", Integer),
            Column("occurred_at", Text, nullable=False),
            Column("variant", Text),
            Column("experiment_id", Text),
            Column("generated_at", Text),
            mysql_charset="utf8mb4",
        )

    def _ensure_track_table(self, conn: Any) -> None:
        conn.execute(CreateTable(self._track_table, if_not_exists=True))

    def _append_rows_db(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        table = self._track_table
        engine = self.engine()
        names = [column.name for column in table.c]
        params: list[dict[str, Any]] = [
            {name: str(row.get(name) or "") if name in _TEXT_DEFAULTED else row.get(name) for name in names}
            for row in unique_event_rows(rows)
        ]
        if is_mysql_family(engine) and any(len(row["event_id"]) > _MYSQL_EVENT_ID_CHARS for row in params):
            raise TrackNormalizeError(f"event_id is longer than {_MYSQL_EVENT_ID_CHARS} characters")
        with engine.begin() as conn:
            self._ensure_track_table(conn)
            existing = _existing_event_ids(conn, table.name, [row["event_id"] for row in params])
            fresh = [row for row in params if not row["event_id"] or row["event_id"] not in existing]
            if not fresh:
                return []
            self._fence.ensure()
            conn.execute(_insert_ignoring_duplicates(engine, table), fresh)
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


def _insert_ignoring_duplicates(engine: Engine, table: Table) -> Insert:
    # Not INSERT IGNORE: that also turns too-long values and NOT NULL violations into warnings.
    if is_mysql_family(engine):
        stmt = mysql_insert(table)
        return stmt.on_duplicate_key_update(event_id=stmt.inserted.event_id)
    dialect_insert = postgresql_insert if engine.dialect.name == "postgresql" else sqlite_insert
    return dialect_insert(table).on_conflict_do_nothing(index_elements=["event_id"])


def _existing_event_ids(conn: Any, table: str, event_ids: Sequence[str]) -> set[str]:
    ids = [event_id for event_id in event_ids if event_id]
    if not ids:
        return set()
    event_id, quoted_table = quote_identifier(conn, "event_id"), quote_identifier(conn, table)
    stmt = text(f"SELECT {event_id} FROM {quoted_table} WHERE {event_id} IN :ids").bindparams(
        bindparam("ids", expanding=True)
    )
    return {str(row[0]) for row in conn.execute(stmt, {"ids": ids}) if row[0] is not None}
