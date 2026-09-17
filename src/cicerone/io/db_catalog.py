"""SQL catalog writes against the input events/users/items tables."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

import pandas as pd
from sqlalchemy import bindparam, inspect, text

from cicerone.io.catalog import (
    EVENT_ID_COLUMN,
    dedupe_event_rows,
    event_pairs,
    item_row_or_none,
    jsonable_row,
    normalize_event_row,
    require_id,
    user_row_or_none,
)
from cicerone.io.db_store import (
    DEFAULT_EVENTS_TABLE,
    DEFAULT_ITEMS_TABLE,
    DEFAULT_USERS_TABLE,
    MISSING_TABLE_ERRORS,
    create_db_engine,
)
from cicerone.io.options import require_option, sql_identifier
from cicerone.io.recommendation_schema import ITEM_COLUMN, USER_COLUMN
from cicerone.io.user_lookup import OCCURRED_AT_COLUMN, newest_events
from cicerone.values import is_missing

logger = logging.getLogger(__name__)
_SQL_HISTORY_OVERFETCH = 8


class DatabaseCatalogStore:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._engine = create_db_engine(require_option(options, "database_url", "db"), options=options)
        self._users = sql_identifier(options.get("users_table", DEFAULT_USERS_TABLE), option="users_table")
        self._items = sql_identifier(options.get("items_table", DEFAULT_ITEMS_TABLE), option="items_table")
        self._events = sql_identifier(
            options.get("events_table", DEFAULT_EVENTS_TABLE), option="events_table"
        )
        self._write_lock = threading.Lock()

    def _table_exists(self, table: str) -> bool:
        return inspect(self._engine).has_table(table)

    def _columns(self, conn, table: str) -> list[str] | None:
        inspector = inspect(conn)
        if not inspector.has_table(table):
            return None
        return [column["name"] for column in inspector.get_columns(table)]

    def _read_id(self, table: str, key: str, value: str) -> pd.DataFrame:
        if not self._table_exists(table):
            return pd.DataFrame()
        sql = text(f'SELECT * FROM "{table}" WHERE "{key}" = :value')
        try:
            return pd.read_sql(sql, self._engine, params={"value": value})
        except MISSING_TABLE_ERRORS:
            return pd.DataFrame()

    def _delete_id(self, conn, table: str, key: str, value: str) -> int:
        savepoint = conn.begin_nested()
        try:
            result = conn.execute(text(f'DELETE FROM "{table}" WHERE "{key}" = :value'), {"value": value})
            savepoint.commit()
        except MISSING_TABLE_ERRORS:
            savepoint.rollback()
            return 0
        return int(result.rowcount or 0)

    @staticmethod
    def _sql_value(value: Any) -> Any:
        if is_missing(value):
            return None
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        if hasattr(value, "isoformat") and not isinstance(value, str):
            return value.isoformat()
        return value

    @staticmethod
    def _decode_sql_value(value: Any) -> Any:
        if not isinstance(value, str) or not value or value[0] not in "{[":
            return value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return parsed if isinstance(parsed, (dict, list)) else value

    def _decode_row(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {key: self._decode_sql_value(value) for key, value in row.items()}

    def _sql_ready(self, row: dict[str, Any]) -> dict[str, Any]:
        return {key: self._sql_value(value) for key, value in row.items()}

    def _sql_records(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for row in frame.to_dict(orient="records"):
            records.append({key: self._sql_value(value) for key, value in row.items()})
        return records

    def _align_frame(self, conn, table: str, frame: pd.DataFrame) -> pd.DataFrame:
        columns = self._columns(conn, table)
        if columns is None:
            return frame
        keep = [name for name in frame.columns if name in columns]
        return frame.loc[:, keep]

    def _ensure_table(self, conn, table: str, frame: pd.DataFrame) -> None:
        if inspect(conn).has_table(table):
            return
        savepoint = conn.begin_nested()
        try:
            frame.head(0).to_sql(table, conn, if_exists="fail", index=False)
            savepoint.commit()
        except Exception:
            savepoint.rollback()
            if not inspect(conn).has_table(table):
                raise

    def _ensure_column(self, conn, table: str, column: str) -> None:
        columns = self._columns(conn, table)
        if columns is None or column in columns:
            return
        savepoint = conn.begin_nested()
        try:
            conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{column}" TEXT'))
            savepoint.commit()
        except Exception:
            savepoint.rollback()
            columns = self._columns(conn, table) or []
            if column not in columns:
                raise

    def _ensure_unique(self, conn, table: str, key: str) -> bool:
        index = f"catalog_{table}_{key}_uidx"
        savepoint = conn.begin_nested()
        try:
            conn.execute(text(f'CREATE UNIQUE INDEX IF NOT EXISTS "{index}" ON "{table}" ("{key}")'))
            savepoint.commit()
        except Exception:
            savepoint.rollback()
            logger.exception("Failed to ensure unique index on %s.%s", table, key)
            return False
        return True

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        for name in frame.columns:
            sql_identifier(str(name), option="column")

    def _events_by_ids(self, conn, event_ids: list[str]) -> list[dict[str, Any]]:
        if not event_ids or not inspect(conn).has_table(self._events):
            return []
        columns = self._columns(conn, self._events) or []
        if EVENT_ID_COLUMN not in columns:
            return []
        sql = text(f'SELECT * FROM "{self._events}" WHERE "{EVENT_ID_COLUMN}" IN :ids').bindparams(
            bindparam("ids", expanding=True)
        )
        frame = pd.read_sql(sql, conn, params={"ids": event_ids})
        if frame.empty:
            return []
        return [self._decode_row(jsonable_row(row)) or {} for row in frame.to_dict(orient="records")]

    def _pairs_present(self, conn, pairs: list[tuple[str, str]]) -> set[tuple[str, str]]:
        if not pairs or not inspect(conn).has_table(self._events):
            return set()
        columns = self._columns(conn, self._events) or []
        if USER_COLUMN not in columns or ITEM_COLUMN not in columns:
            return set()
        wanted = set(pairs)
        users = list({user_id for user_id, _ in wanted})
        sql = text(
            f'SELECT "{USER_COLUMN}", "{ITEM_COLUMN}" FROM "{self._events}" WHERE "{USER_COLUMN}" IN :users'
        ).bindparams(bindparam("users", expanding=True))
        frame = pd.read_sql(sql, conn, params={"users": users})
        if frame.empty:
            return set()
        found: set[tuple[str, str]] = set()
        for row in frame.to_dict(orient="records"):
            pair = (str(row[USER_COLUMN]), str(row[ITEM_COLUMN]))
            if pair in wanted:
                found.add(pair)
        return found

    def _upsert_frame(self, conn, table: str, key: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self._validate_columns(frame)
        self._ensure_table(conn, table, frame)
        for name in frame.columns:
            self._ensure_column(conn, table, str(name))
        aligned = self._align_frame(conn, table, frame)
        if aligned.empty:
            return
        if key not in aligned.columns or not self._ensure_unique(conn, table, key):
            if key in aligned.columns:
                for value in aligned[key].astype(str).tolist():
                    self._delete_id(conn, table, key, value)
            aligned.to_sql(table, conn, if_exists="append", index=False)
            return
        columns = list(aligned.columns)
        cols = ", ".join(f'"{sql_identifier(name, option="column")}"' for name in columns)
        placeholders = ", ".join(f":{name}" for name in columns)
        updates = ", ".join(
            f'"{sql_identifier(name, option="column")}" = EXCLUDED."{sql_identifier(name, option="column")}"'
            for name in columns
            if name != key
        )
        sql = f'INSERT INTO "{table}" ({cols}) VALUES ({placeholders})'
        sql += (
            f' ON CONFLICT ("{key}") DO UPDATE SET {updates}'
            if updates
            else f' ON CONFLICT ("{key}") DO NOTHING'
        )
        conn.execute(text(sql), self._sql_records(aligned))

    def upsert_user(self, row: dict[str, Any]) -> None:
        user_id = require_id(row, USER_COLUMN)
        payload = self._sql_ready(row)
        payload[USER_COLUMN] = user_id
        frame = pd.DataFrame([payload])
        with self._write_lock, self._engine.begin() as conn:
            self._upsert_frame(conn, self._users, USER_COLUMN, frame)

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        return self._decode_row(user_row_or_none(self._read_id(self._users, USER_COLUMN, user_id), user_id))

    def delete_user(self, user_id: str) -> int:
        with self._write_lock, self._engine.begin() as conn:
            users = self._delete_id(conn, self._users, USER_COLUMN, user_id)
            events = self._delete_id(conn, self._events, USER_COLUMN, user_id)
        return users + events

    def upsert_item(self, row: dict[str, Any]) -> None:
        item_id = require_id(row, ITEM_COLUMN)
        payload = self._sql_ready(row)
        payload[ITEM_COLUMN] = item_id
        frame = pd.DataFrame([payload])
        with self._write_lock, self._engine.begin() as conn:
            self._upsert_frame(conn, self._items, ITEM_COLUMN, frame)

    def get_item(self, item_id: str) -> dict[str, Any] | None:
        return self._decode_row(item_row_or_none(self._read_id(self._items, ITEM_COLUMN, item_id), item_id))

    def delete_item(self, item_id: str) -> int:
        with self._write_lock, self._engine.begin() as conn:
            items = self._delete_id(conn, self._items, ITEM_COLUMN, item_id)
            events = self._delete_id(conn, self._events, ITEM_COLUMN, item_id)
        return items + events

    def upsert_events(self, rows: list[dict[str, Any]]) -> int:
        accepted, _, _ = self.replace_events(rows)
        return accepted

    def replace_events(
        self, rows: list[dict[str, Any]]
    ) -> tuple[int, list[tuple[str, str]], list[tuple[str, str]]]:
        if not rows:
            return 0, [], []
        incoming = dedupe_event_rows([normalize_event_row(row) for row in rows])
        frame = pd.DataFrame(incoming)
        ids = [str(row[EVENT_ID_COLUMN]) for row in incoming]
        with self._write_lock, self._engine.begin() as conn:
            previous = self._events_by_ids(conn, ids)
            self._upsert_frame(conn, self._events, EVENT_ID_COLUMN, frame)
            old_pairs = event_pairs(previous)
            remaining = self._pairs_present(conn, old_pairs)
            discard = [pair for pair in old_pairs if pair not in remaining]
            return int(len(incoming)), discard, event_pairs(incoming)

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        if not event_id or not self._table_exists(self._events):
            return None
        try:
            columns = [column["name"] for column in inspect(self._engine).get_columns(self._events)]
        except MISSING_TABLE_ERRORS:
            return None
        if EVENT_ID_COLUMN not in columns:
            return None
        frame = self._read_id(self._events, EVENT_ID_COLUMN, event_id)
        if frame.empty:
            return None
        return self._decode_row(jsonable_row(frame.iloc[0].to_dict()))

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
        if not self._table_exists(self._events):
            return pd.DataFrame()
        sql_limit = max(int(limit) * _SQL_HISTORY_OVERFETCH, int(limit))
        params = {"user_id": user_id, "limit": sql_limit}
        ordered = (
            f'SELECT * FROM "{self._events}" WHERE "{USER_COLUMN}" = :user_id '
            f'ORDER BY "{OCCURRED_AT_COLUMN}" DESC LIMIT :limit'
        )
        fallback = f'SELECT * FROM "{self._events}" WHERE "{USER_COLUMN}" = :user_id LIMIT :limit'
        try:
            frame = pd.read_sql(text(ordered), self._engine, params=params)
        except MISSING_TABLE_ERRORS:
            try:
                frame = pd.read_sql(text(fallback), self._engine, params=params)
            except MISSING_TABLE_ERRORS:
                return pd.DataFrame()
        return newest_events(frame, limit)

    def delete_events_for_user(self, user_id: str, *, item_id: str | None = None) -> int:
        with self._write_lock, self._engine.begin() as conn:
            savepoint = conn.begin_nested()
            try:
                if item_id is None:
                    result = conn.execute(
                        text(f'DELETE FROM "{self._events}" WHERE "{USER_COLUMN}" = :user_id'),
                        {"user_id": user_id},
                    )
                else:
                    result = conn.execute(
                        text(
                            f'DELETE FROM "{self._events}" WHERE "{USER_COLUMN}" = :user_id '
                            f'AND "{ITEM_COLUMN}" = :item_id'
                        ),
                        {"user_id": user_id, "item_id": item_id},
                    )
                savepoint.commit()
            except MISSING_TABLE_ERRORS:
                savepoint.rollback()
                return 0
            return int(result.rowcount or 0)
