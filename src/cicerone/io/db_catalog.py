"""SQL catalog writes against the input events/users/items tables."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sqlalchemy import bindparam, create_engine, inspect, text

from cicerone.io.catalog import (
    EVENT_ID_COLUMN,
    item_row_or_none,
    normalize_event_row,
    require_id,
    user_row_or_none,
)
from cicerone.io.db_store import (
    DEFAULT_EVENTS_TABLE,
    DEFAULT_ITEMS_TABLE,
    DEFAULT_USERS_TABLE,
    MISSING_TABLE_ERRORS,
)
from cicerone.io.options import require_option, sql_identifier
from cicerone.io.recommendation_schema import ITEM_COLUMN, USER_COLUMN
from cicerone.io.user_lookup import OCCURRED_AT_COLUMN, newest_events

logger = logging.getLogger(__name__)


class DatabaseCatalogStore:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._engine = create_engine(require_option(options, "database_url", "db"), pool_pre_ping=True)
        self._users = sql_identifier(options.get("users_table", DEFAULT_USERS_TABLE), option="users_table")
        self._items = sql_identifier(options.get("items_table", DEFAULT_ITEMS_TABLE), option="items_table")
        self._events = sql_identifier(
            options.get("events_table", DEFAULT_EVENTS_TABLE), option="events_table"
        )

    def _table_exists(self, table: str) -> bool:
        return inspect(self._engine).has_table(table)

    def _read_id(self, table: str, key: str, value: str) -> pd.DataFrame:
        if not self._table_exists(table):
            return pd.DataFrame()
        sql = text(f'SELECT * FROM "{table}" WHERE "{key}" = :value')
        try:
            return pd.read_sql(sql, self._engine, params={"value": value})
        except MISSING_TABLE_ERRORS:
            return pd.DataFrame()

    def _delete_id(self, conn, table: str, key: str, value: str) -> int:
        try:
            result = conn.execute(text(f'DELETE FROM "{table}" WHERE "{key}" = :value'), {"value": value})
        except MISSING_TABLE_ERRORS:
            return 0
        return int(result.rowcount or 0)

    def upsert_user(self, row: dict[str, Any]) -> None:
        user_id = require_id(row, USER_COLUMN)
        payload = dict(row)
        payload[USER_COLUMN] = user_id
        frame = pd.DataFrame([payload])
        with self._engine.begin() as conn:
            self._delete_id(conn, self._users, USER_COLUMN, user_id)
            frame.to_sql(self._users, conn, if_exists="append", index=False)

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        return user_row_or_none(self._read_id(self._users, USER_COLUMN, user_id), user_id)

    def delete_user(self, user_id: str) -> int:
        with self._engine.begin() as conn:
            users = self._delete_id(conn, self._users, USER_COLUMN, user_id)
            events = 0
            try:
                result = conn.execute(
                    text(f'DELETE FROM "{self._events}" WHERE "{USER_COLUMN}" = :user_id'),
                    {"user_id": user_id},
                )
                events = int(result.rowcount or 0)
            except MISSING_TABLE_ERRORS:
                events = 0
        return users + events

    def upsert_item(self, row: dict[str, Any]) -> None:
        item_id = require_id(row, ITEM_COLUMN)
        payload = dict(row)
        payload[ITEM_COLUMN] = item_id
        frame = pd.DataFrame([payload])
        with self._engine.begin() as conn:
            self._delete_id(conn, self._items, ITEM_COLUMN, item_id)
            frame.to_sql(self._items, conn, if_exists="append", index=False)

    def get_item(self, item_id: str) -> dict[str, Any] | None:
        return item_row_or_none(self._read_id(self._items, ITEM_COLUMN, item_id), item_id)

    def delete_item(self, item_id: str) -> int:
        with self._engine.begin() as conn:
            return self._delete_id(conn, self._items, ITEM_COLUMN, item_id)

    def upsert_events(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        incoming = [normalize_event_row(row) for row in rows]
        frame = pd.DataFrame(incoming)
        with self._engine.begin() as conn:
            if EVENT_ID_COLUMN in frame.columns and inspect(self._engine).has_table(self._events):
                ids = [str(value) for value in frame[EVENT_ID_COLUMN].tolist() if value not in (None, "")]
                if ids:
                    conn.execute(
                        text(f'DELETE FROM "{self._events}" WHERE "{EVENT_ID_COLUMN}" IN :ids').bindparams(
                            bindparam("ids", expanding=True)
                        ),
                        {"ids": ids},
                    )
            frame.to_sql(self._events, conn, if_exists="append", index=False)
        return int(len(frame))

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
        with self._engine.begin() as conn:
            if not inspect(conn).has_table(self._events):
                return 0
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
            return int(result.rowcount or 0)
