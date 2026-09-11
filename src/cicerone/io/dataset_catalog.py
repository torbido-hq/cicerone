"""Read-modify-write catalog on a dataset (parquet) input store."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from cicerone.io.catalog import (
    EVENT_ID_COLUMN,
    item_row_or_none,
    normalize_event_row,
    require_id,
    user_row_or_none,
)
from cicerone.io.options import (
    build_s3_client,
    is_s3_not_found,
    object_key,
    read_parquet,
    require_option,
    validate_storage_options,
)
from cicerone.io.recommendation_schema import ITEM_COLUMN, USER_COLUMN
from cicerone.io.user_lookup import filter_rows_for_user, newest_events

logger = logging.getLogger(__name__)

_USERS = "users.parquet"
_ITEMS = "items.parquet"
_EVENTS = "events.parquet"


class DatasetCatalogStore:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._backend = validate_storage_options(options)

    def _read(self, filename: str) -> pd.DataFrame:
        try:
            return read_parquet(self._options, filename)
        except FileNotFoundError:
            return pd.DataFrame()
        except Exception as exc:
            if is_s3_not_found(exc):
                return pd.DataFrame()
            raise

    def _write(self, filename: str, frame: pd.DataFrame) -> None:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        payload = buffer.getvalue()
        if self._backend == "local":
            path = Path(require_option(self._options, "path", "local")) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.write_bytes(payload)
            tmp.replace(path)
            return
        bucket = require_option(self._options, "bucket", "s3")
        key = object_key(self._options, filename)
        client = build_s3_client(self._options)
        client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="application/octet-stream")

    def _replace_row(self, filename: str, key: str, value: str, row: dict[str, Any]) -> None:
        frame = self._read(filename)
        if not frame.empty and key in frame.columns:
            frame = frame.loc[frame[key].astype(str) != str(value)]
        incoming = pd.DataFrame([row])
        merged = pd.concat([frame, incoming], ignore_index=True) if not frame.empty else incoming
        self._write(filename, merged)

    def upsert_user(self, row: dict[str, Any]) -> None:
        user_id = require_id(row, USER_COLUMN)
        payload = dict(row)
        payload[USER_COLUMN] = user_id
        self._replace_row(_USERS, USER_COLUMN, user_id, payload)

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        return user_row_or_none(self._read(_USERS), user_id)

    def delete_user(self, user_id: str) -> int:
        users = self._read(_USERS)
        before = 0 if users.empty else int((users[USER_COLUMN].astype(str) == str(user_id)).sum())
        if not users.empty and USER_COLUMN in users.columns:
            remaining = users.loc[users[USER_COLUMN].astype(str) != str(user_id)]
            self._write(_USERS, remaining.reset_index(drop=True))
        events_deleted = self.delete_events_for_user(user_id)
        return before + events_deleted

    def upsert_item(self, row: dict[str, Any]) -> None:
        item_id = require_id(row, ITEM_COLUMN)
        payload = dict(row)
        payload[ITEM_COLUMN] = item_id
        self._replace_row(_ITEMS, ITEM_COLUMN, item_id, payload)

    def get_item(self, item_id: str) -> dict[str, Any] | None:
        return item_row_or_none(self._read(_ITEMS), item_id)

    def delete_item(self, item_id: str) -> int:
        items = self._read(_ITEMS)
        if items.empty or ITEM_COLUMN not in items.columns:
            return 0
        before = int((items[ITEM_COLUMN].astype(str) == str(item_id)).sum())
        remaining = items.loc[items[ITEM_COLUMN].astype(str) != str(item_id)]
        self._write(_ITEMS, remaining.reset_index(drop=True))
        return before

    def upsert_events(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        incoming = pd.DataFrame([normalize_event_row(row) for row in rows])
        existing = self._read(_EVENTS)
        if existing.empty:
            self._write(_EVENTS, incoming)
            return int(len(incoming))
        if EVENT_ID_COLUMN in incoming.columns and EVENT_ID_COLUMN in existing.columns:
            ids = set(incoming[EVENT_ID_COLUMN].astype(str))
            existing = existing.loc[~existing[EVENT_ID_COLUMN].astype(str).isin(ids)]
        merged = pd.concat([existing, incoming], ignore_index=True)
        self._write(_EVENTS, merged)
        return int(len(incoming))

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
        return newest_events(filter_rows_for_user(self._read(_EVENTS), user_id), limit)

    def delete_events_for_user(self, user_id: str, *, item_id: str | None = None) -> int:
        events = self._read(_EVENTS)
        if events.empty or USER_COLUMN not in events.columns:
            return 0
        mask = events[USER_COLUMN].astype(str) == str(user_id)
        if item_id is not None and ITEM_COLUMN in events.columns:
            mask = mask & (events[ITEM_COLUMN].astype(str) == str(item_id))
        deleted = int(mask.sum())
        self._write(_EVENTS, events.loc[~mask].reset_index(drop=True))
        return deleted
