"""SQL catalog writes against the input events/users/items tables."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

import pandas as pd
from sqlalchemy import inspect, text

from cicerone.io.catalog import (
    EVENT_ID_COLUMN,
    dedupe_event_rows,
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

logger = logging.getLogger(__name__)


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
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        if hasattr(value, "isoformat") and not isinstance(value, str):
            return value.isoformat()
        return value

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

    def _upsert_frame(self, conn, table: str, key: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self._ensure_table(conn, table, frame)
        if table == self._events and key == EVENT_ID_COLUMN:
            self._ensure_column(conn, table, EVENT_ID_COLUMN)
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
        cols = ", ".join(f'"{name}"' for name in columns)
        placeholders = ", ".join(f":{name}" for name in columns)
        updates = ", ".join(f'"{name}" = EXCLUDED."{name}"' for name in columns if name != key)
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
        return user_row_or_none(self._read_id(self._users, USER_COLUMN, user_id), user_id)

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
        return item_row_or_none(self._read_id(self._items, ITEM_COLUMN, item_id), item_id)

    def delete_item(self, item_id: str) -> int:
        with self._write_lock, self._engine.begin() as conn:
            return self._delete_id(conn, self._items, ITEM_COLUMN, item_id)

    def upsert_events(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        incoming = dedupe_event_rows([normalize_event_row(row) for row in rows])
        frame = pd.DataFrame(incoming)
        with self._write_lock, self._engine.begin() as conn:
            self._upsert_frame(conn, self._events, EVENT_ID_COLUMN, frame)
        return int(len(incoming))

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
        return jsonable_row(frame.iloc[0].to_dict())

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
        if not self._table_exists(self._events):
            return pd.DataFrame()
        sql = text(
            f'SELECT * FROM "{self._events}" WHERE "{USER_COLUMN}" = :user_id '
            f'ORDER BY "{OCCURRED_AT_COLUMN}" DESC LIMIT :limit'
        )
        try:
            frame = pd.read_sql(sql, self._engine, params={"user_id": user_id, "limit": int(limit)})
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
