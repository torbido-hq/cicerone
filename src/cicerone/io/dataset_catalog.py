"""Read-modify-write catalog on a dataset (parquet) input store."""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path
from typing import Any

import pandas as pd

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
        self._locks = {
            _USERS: threading.Lock(),
            _ITEMS: threading.Lock(),
            _EVENTS: threading.Lock(),
        }

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
        with self._locks[filename]:
            frame = self._read(filename)
            existing: dict[str, Any] = {}
            if not frame.empty and key in frame.columns:
                matched = frame.loc[frame[key].astype(str) == str(value)]
                if not matched.empty:
                    existing = matched.iloc[0].to_dict()
                frame = frame.loc[frame[key].astype(str) != str(value)]
            incoming = pd.DataFrame([{**existing, **row}])
            merged = incoming if len(frame.columns) == 0 else pd.concat([frame, incoming], ignore_index=True)
            self._write(filename, merged)

    def upsert_user(self, row: dict[str, Any]) -> None:
        user_id = require_id(row, USER_COLUMN)
        payload = dict(row)
        payload[USER_COLUMN] = user_id
        self._replace_row(_USERS, USER_COLUMN, user_id, payload)

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        return user_row_or_none(self._read(_USERS), user_id)

    def delete_user(self, user_id: str) -> int:
        with self._locks[_USERS], self._locks[_EVENTS]:
            users = self._read(_USERS)
            before = 0
            if not users.empty and USER_COLUMN in users.columns:
                before = int((users[USER_COLUMN].astype(str) == str(user_id)).sum())
                remaining = users.loc[users[USER_COLUMN].astype(str) != str(user_id)]
                self._write(_USERS, remaining.reset_index(drop=True))
            return before + self._delete_events_locked(user_id)

    def upsert_item(self, row: dict[str, Any]) -> None:
        item_id = require_id(row, ITEM_COLUMN)
        payload = dict(row)
        payload[ITEM_COLUMN] = item_id
        self._replace_row(_ITEMS, ITEM_COLUMN, item_id, payload)

    def get_item(self, item_id: str) -> dict[str, Any] | None:
        return item_row_or_none(self._read(_ITEMS), item_id)

    def delete_item(self, item_id: str) -> int:
        with self._locks[_ITEMS]:
            items = self._read(_ITEMS)
            if items.empty or ITEM_COLUMN not in items.columns:
                return 0
            before = int((items[ITEM_COLUMN].astype(str) == str(item_id)).sum())
            remaining = items.loc[items[ITEM_COLUMN].astype(str) != str(item_id)]
            self._write(_ITEMS, remaining.reset_index(drop=True))
            return before

    def upsert_events(self, rows: list[dict[str, Any]]) -> int:
        accepted, _, _ = self.replace_events(rows)
        return accepted

    def replace_events(
        self, rows: list[dict[str, Any]]
    ) -> tuple[int, list[tuple[str, str]], list[tuple[str, str]]]:
        if not rows:
            return 0, [], []
        incoming_rows = dedupe_event_rows([normalize_event_row(row) for row in rows])
        incoming = pd.DataFrame(incoming_rows)
        with self._locks[_EVENTS]:
            existing = self._read(_EVENTS)
            previous: list[dict[str, Any]] = []
            if (
                not existing.empty
                and EVENT_ID_COLUMN in incoming.columns
                and EVENT_ID_COLUMN in existing.columns
            ):
                ids = set(incoming[EVENT_ID_COLUMN].astype(str))
                matched = existing.loc[existing[EVENT_ID_COLUMN].astype(str).isin(ids)]
                previous = [jsonable_row(row) for row in matched.to_dict(orient="records")]
                existing = existing.loc[~existing[EVENT_ID_COLUMN].astype(str).isin(ids)]
            merged = incoming if existing.empty else pd.concat([existing, incoming], ignore_index=True)
            self._write(_EVENTS, merged)
            remaining = set(event_pairs([jsonable_row(row) for row in merged.to_dict(orient="records")]))
            discard = [pair for pair in event_pairs(previous) if pair not in remaining]
            return int(len(incoming_rows)), discard, event_pairs(incoming_rows)

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        if not event_id:
            return None
        with self._locks[_EVENTS]:
            events = self._read(_EVENTS)
            if events.empty or EVENT_ID_COLUMN not in events.columns:
                return None
            matched = events.loc[events[EVENT_ID_COLUMN].astype(str) == str(event_id)]
            if matched.empty:
                return None
            return jsonable_row(matched.iloc[0].to_dict())

    def _read_for_user(self, filename: str, user_id: str) -> pd.DataFrame:
        try:
            frame = read_parquet(self._options, filename, filters=[(USER_COLUMN, "==", user_id)])
        except FileNotFoundError:
            return pd.DataFrame()
        except Exception as exc:
            if is_s3_not_found(exc):
                return pd.DataFrame()
            message = str(exc).lower()
            if "user_id" in message or "fieldref" in message or "filter" in message:
                logger.warning("Filtered %s read failed; falling back to full-file load: %s", filename, exc)
                frame = self._read(filename)
            else:
                raise
        if USER_COLUMN not in frame.columns:
            frame = self._read(filename)
        return filter_rows_for_user(frame, user_id)

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
        return newest_events(self._read_for_user(_EVENTS, user_id), limit)

    def delete_events_for_user(self, user_id: str, *, item_id: str | None = None) -> int:
        with self._locks[_EVENTS]:
            return self._delete_events_locked(user_id, item_id=item_id)

    def _delete_events_locked(self, user_id: str, *, item_id: str | None = None) -> int:
        events = self._read(_EVENTS)
        if events.empty or USER_COLUMN not in events.columns:
            return 0
        mask = events[USER_COLUMN].astype(str) == str(user_id)
        if item_id is not None:
            if ITEM_COLUMN not in events.columns:
                return 0
            mask = mask & (events[ITEM_COLUMN].astype(str) == str(item_id))
        deleted = int(mask.sum())
        self._write(_EVENTS, events.loc[~mask].reset_index(drop=True))
        return deleted
