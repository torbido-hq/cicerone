"""Live catalog writes (users / items / events) against the input store."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Protocol

import pandas as pd

from cicerone.events.normalize import normalize_event
from cicerone.io.recommendation_schema import ITEM_COLUMN, USER_COLUMN
from cicerone.io.user_lookup import OCCURRED_AT_COLUMN, filter_rows_for_user
from cicerone.values import is_missing

EVENT_TYPE_COLUMN = "event_type"
EVENT_ID_COLUMN = "event_id"
QUANTITY_COLUMN = "quantity"
EVENT_COLUMNS: tuple[str, ...] = (
    USER_COLUMN,
    ITEM_COLUMN,
    EVENT_TYPE_COLUMN,
    QUANTITY_COLUMN,
    OCCURRED_AT_COLUMN,
    EVENT_ID_COLUMN,
)


class CatalogStore(Protocol):
    def upsert_user(self, row: dict[str, Any]) -> None: ...

    def get_user(self, user_id: str) -> dict[str, Any] | None: ...

    def delete_user(self, user_id: str) -> int: ...

    def upsert_item(self, row: dict[str, Any]) -> None: ...

    def get_item(self, item_id: str) -> dict[str, Any] | None: ...

    def delete_item(self, item_id: str) -> int: ...

    def upsert_events(self, rows: list[dict[str, Any]]) -> int: ...

    def replace_events(
        self, rows: list[dict[str, Any]]
    ) -> tuple[int, list[tuple[str, str]], list[tuple[str, str]]]: ...

    def get_event(self, event_id: str) -> dict[str, Any] | None: ...

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame: ...

    def delete_events_for_user(self, user_id: str, *, item_id: str | None = None) -> int: ...


def require_id(row: dict[str, Any], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def normalize_event_row(row: dict[str, Any]) -> dict[str, Any]:
    event = normalize_event(row)
    return {
        USER_COLUMN: event.user_id,
        ITEM_COLUMN: event.item_id,
        EVENT_TYPE_COLUMN: event.event_type,
        QUANTITY_COLUMN: event.quantity,
        OCCURRED_AT_COLUMN: event.occurred_at,
        EVENT_ID_COLUMN: event.event_id,
    }


def dedupe_event_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last: dict[str, dict[str, Any]] = {}
    for row in rows:
        last[str(row[EVENT_ID_COLUMN])] = row
    return list(last.values())


def event_pairs(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for row in rows:
        user_id = str(row.get(USER_COLUMN, "")).strip()
        item_id = str(row.get(ITEM_COLUMN, "")).strip()
        if user_id and item_id:
            pairs.append((user_id, item_id))
    return pairs


def filter_item_row(frame: pd.DataFrame, item_id: str) -> pd.DataFrame:
    if ITEM_COLUMN not in frame.columns:
        return frame.iloc[0:0].copy()
    matched = frame.loc[frame[ITEM_COLUMN].astype(str) == str(item_id)]
    return matched.reset_index(drop=True)


def user_row_or_none(frame: pd.DataFrame, user_id: str) -> dict[str, Any] | None:
    matched = filter_rows_for_user(frame, user_id)
    if matched.empty:
        return None
    return jsonable_row(matched.iloc[0].to_dict())


def item_row_or_none(frame: pd.DataFrame, item_id: str) -> dict[str, Any] | None:
    matched = filter_item_row(frame, item_id)
    if matched.empty:
        return None
    return jsonable_row(matched.iloc[0].to_dict())


def jsonable_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if hasattr(value, "item") and not isinstance(value, (bytes, bytearray, str)):
            with suppress(ValueError, AttributeError):
                value = value.item()
        if is_missing(value):
            value = None
        elif hasattr(value, "isoformat") and not isinstance(value, str):
            value = value.isoformat()
        out[str(key)] = value
    return out
