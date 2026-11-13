"""Database backend for TrackStore."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any

import pandas as pd
from sqlalchemy import BigInteger, Column, Engine, Insert, Integer, Table, Text, bindparam, select, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.schema import CreateTable

from cicerone.config.constants import MYSQL_KEY_CHARS, ConfigError
from cicerone.io.db_errors import MISSING_TABLE_ERRORS, is_missing_column_error, is_missing_table_error
from cicerone.io.engines import (
    JSON_PAYLOAD_TYPE,
    TEXT_KEY_TYPE,
    LazyEngine,
    database_backend_name,
    is_mysql_family,
    quote_identifier,
    store_table,
)
from cicerone.io.options import require_option, sql_identifier, table_option
from cicerone.io.recommendation_schema import RANK_COLUMN, USER_COLUMN
from cicerone.io.sql_rows import append_frame_fenced, frame_records, read_sql_or_missing, replace_rows_fenced
from cicerone.locks import WriteFence
from cicerone.track.normalize import TrackNormalizeError
from cicerone.track.store_common import (
    DEFAULT_EVAL_TABLE,
    DEFAULT_HISTORY_TABLE,
    DEFAULT_TRACK_TABLE,
    HISTORY_COLUMNS,
    empty_history_frame,
    filter_history,
    history_sql_filter,
    track_row_sql_filter,
    unique_event_rows,
    unique_matching_rows,
)

logger = logging.getLogger(__name__)

_MYSQL_TEXT_BYTES = 65535


class TrackDbBackend:
    """Track rows, eval report and history in ``[output]`` tables."""

    def __init__(self, options: dict[str, Any], fence: WriteFence):
        self._options = options
        self._fence = fence
        self._engine = LazyEngine(options)
        self._append_lock = threading.Lock()
        self._mysql_table_ready = False

    def append_accepted(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with self._append_lock, self._fence.lease():
            return self._append_rows_db(rows)

    def read_rows(
        self, *, kind: str | None, experiment_id: str | None, since: str | None
    ) -> list[dict[str, Any]]:
        rows = self._read_rows_db(kind=kind, experiment_id=experiment_id, since=since)
        return unique_matching_rows(rows, kind=kind, experiment_id=experiment_id, since=since)

    def eval_writer(self, payload: dict[str, Any]) -> Callable[[], None]:
        if self.engine().dialect.name == "mariadb":
            # Only here: a job always writes eval before history, so this warns once for both.
            logger.warning("Track eval and recommendation history aren't supported on MariaDB; skipping them")
            return lambda: None
        encoded = json.dumps(payload)
        return lambda: self._write_eval_db(encoded)

    def read_eval(self) -> dict[str, Any] | None:
        return self._read_eval_db()

    def history_writer(self, frame: pd.DataFrame, generated_at: str) -> Callable[[], None]:
        del generated_at
        if self.engine().dialect.name == "mariadb":
            return lambda: None
        # pandas binds an all-NA object column as text, which Postgres won't assign to BIGINT.
        numeric = frame.assign(**{RANK_COLUMN: pd.to_numeric(frame[RANK_COLUMN], errors="coerce")})
        return lambda: self._append_history_db(numeric)

    def read_history(self, *, generated_ats: set[str] | None, since: str | None) -> pd.DataFrame:
        frame = self._read_history_db(generated_ats=generated_ats)
        return filter_history(frame, generated_ats=generated_ats, since=since)

    def engine(self) -> Engine:
        return self._engine.get()

    def delete_user_rows(self, user_id: str, *, conn: Any) -> int:
        return delete_rows_for_user(conn, self._track_table.name, user_id, fence=self._fence.ensure)

    def delete_history_for_user(self, user_id: str, *, conn: Any) -> int:
        return delete_rows_for_user(conn, self._history_table.name, user_id, fence=self._fence.ensure)

    @cached_property
    def _track_table(self) -> Table:
        return store_table(
            table_option(self._options, "track_table", DEFAULT_TRACK_TABLE),
            Column("event_id", TEXT_KEY_TYPE, primary_key=True),
            Column("kind", Text, nullable=False),
            Column("user_id", Text, nullable=False),
            Column("item_id", Text, nullable=False),
            Column("rank", Integer),
            Column("occurred_at", Text, nullable=False),
            Column("variant", Text),
            Column("experiment_id", Text),
            Column("generated_at", Text),
        )

    @cached_property
    def _eval_table(self) -> Table:
        return store_table(
            table_option(self._options, "eval_table", DEFAULT_EVAL_TABLE),
            Column("payload", JSON_PAYLOAD_TYPE, nullable=False),
            Column("written_at", Text, nullable=False),
        )

    @cached_property
    def _history_table(self) -> Table:
        return store_table(
            table_option(self._options, "history_table", DEFAULT_HISTORY_TABLE),
            *(Column(name, BigInteger if name == RANK_COLUMN else Text) for name in HISTORY_COLUMNS),
        )

    def _ensure_track_table(self, conn: Any) -> None:
        if self._mysql_table_ready:
            return
        conn.execute(CreateTable(self._track_table, if_not_exists=True))
        # DDL commits implicitly on MySQL, so don't repeat it on every append.
        self._mysql_table_ready = is_mysql_family(conn)

    def _append_rows_db(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        table = self._track_table
        engine = self.engine()
        params: list[dict[str, Any]] = [
            {
                column.name: row.get(column.name) if column.nullable else str(row.get(column.name) or "")
                for column in table.c
            }
            for row in rows
        ]
        insert_rows = _insert_ignoring_duplicates(engine, table)
        on_mysql = is_mysql_family(engine)
        if on_mysql:
            _reject_values_mysql_cannot_store(params)
        with engine.begin() as conn:
            self._ensure_track_table(conn)
            existing = _existing_event_ids(conn, table, [row["event_id"] for row in params])
            fresh = unique_event_rows(params, existing)
            if not fresh:
                return []
            self._fence.ensure()
            # Sorted so overlapping batches from other replicas lock in one order on MySQL.
            conn.execute(insert_rows, sorted(fresh, key=lambda row: row["event_id"]) if on_mysql else fresh)
            self._fence.ensure()
            return fresh

    def _read_rows_db(
        self,
        *,
        kind: str | None,
        experiment_id: str | None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        table = self._track_table.name
        engine = self.engine()
        quoted = quote_identifier(engine, table)
        clause, params = track_row_sql_filter(kind=kind, experiment_id=experiment_id, since=since)
        frame = read_sql_or_missing(engine, text(f"SELECT * FROM {quoted}{clause}"), table, params)
        return frame_records(frame)

    def _write_eval_db(self, encoded: str) -> None:
        table = self._eval_table
        with self.engine().begin() as conn:
            self._fence.ensure()
            conn.execute(CreateTable(table, if_not_exists=True))
            written_at = pd.Timestamp.now(tz="UTC").isoformat()
            replace_rows_fenced(conn, table.name, {"payload": encoded, "written_at": written_at}, self._fence)

    def _read_eval_db(self) -> dict[str, Any] | None:
        table = self._eval_table.name
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
        append_frame_fenced(self.engine(), self._history_table, frame, self._fence)

    def _read_history_db(
        self,
        *,
        generated_ats: set[str] | None,
    ) -> pd.DataFrame:
        table = self._history_table.name
        engine = self.engine()
        quoted = quote_identifier(engine, table)
        clause, params = history_sql_filter(generated_ats=generated_ats)
        stmt = text(f"SELECT * FROM {quoted}{clause}")
        if generated_ats:
            stmt = stmt.bindparams(bindparam("generated_ats", expanding=True))
        frame = read_sql_or_missing(engine, stmt, table, params)
        return empty_history_frame() if frame is None else frame


def _on_duplicate_key_insert(table: Table) -> Insert:
    # Not INSERT IGNORE: that also turns too-long values and NOT NULL violations into warnings.
    stmt = mysql_insert(table)
    return stmt.on_duplicate_key_update(event_id=stmt.inserted.event_id)


_INSERTS_IGNORING_DUPLICATES: dict[str, Callable[[Table], Insert]] = {
    "mysql": _on_duplicate_key_insert,
    "postgresql": lambda table: postgresql_insert(table).on_conflict_do_nothing(index_elements=["event_id"]),
    "sqlite": lambda table: sqlite_insert(table).on_conflict_do_nothing(index_elements=["event_id"]),
}


def _require_track_dialect(name: str) -> None:
    if name not in _INSERTS_IGNORING_DUPLICATES:
        raise ConfigError(f"The track table supports {', '.join(_INSERTS_IGNORING_DUPLICATES)}, not {name}")


def require_track_ingest_database(options: dict[str, Any]) -> None:
    _require_track_dialect(database_backend_name(require_option(options, "database_url", "db")))


def _insert_ignoring_duplicates(engine: Engine, table: Table) -> Insert:
    _require_track_dialect(engine.dialect.name)
    return _INSERTS_IGNORING_DUPLICATES[engine.dialect.name](table)


def _existing_event_ids(conn: Any, table: Table, event_ids: Sequence[str]) -> set[str]:
    ids = [event_id for event_id in event_ids if event_id]
    if not ids:
        return set()
    return set(conn.scalars(select(table.c.event_id).where(table.c.event_id.in_(ids))))


def _reject_values_mysql_cannot_store(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if len(row["event_id"]) > MYSQL_KEY_CHARS:
            raise TrackNormalizeError(f"event_id is longer than {MYSQL_KEY_CHARS} characters")
        for name, value in row.items():
            try:
                if isinstance(value, str) and len(value.encode()) > _MYSQL_TEXT_BYTES:
                    raise TrackNormalizeError(f"{name} is longer than {_MYSQL_TEXT_BYTES} bytes")
            except UnicodeEncodeError as exc:
                raise TrackNormalizeError(f"{name} is not valid UTF-8 text") from exc


def delete_rows_for_user(
    conn: Any,
    table: str,
    user_id: str,
    *,
    fence: Callable[[], None],
    user_column: str = USER_COLUMN,
) -> int:
    """Delete one user's rows on an open transaction. A missing table counts as zero.

    Stored ids are trimmed and compared as text, so padded or integer ids still match.
    """
    table = sql_identifier(table, option="table")
    user_column = sql_identifier(user_column, option="user_column")
    # The column stays unquoted: SQLite treats an unknown quoted name as a string.
    match = f"TRIM(CAST({user_column} AS VARCHAR)) = :user_id"
    delete_sql = text(f"DELETE FROM {quote_identifier(conn, table)} WHERE {match}")
    try:
        with conn.begin_nested():
            fence()
            removed = max(int(conn.execute(delete_sql, {"user_id": user_id}).rowcount or 0), 0)
            fence()
            return removed
    except MISSING_TABLE_ERRORS as exc:
        if is_missing_column_error(exc):
            raise ValueError(f"{table} is missing {user_column}") from exc
        if is_missing_table_error(exc):
            return 0
        raise
