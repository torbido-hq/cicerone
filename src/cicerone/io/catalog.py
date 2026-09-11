"""Live catalog writes (users / items / events) for the Gorse-style serve loop."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Protocol

import pandas as pd

from cicerone.io.recommendation_schema import ITEM_COLUMN, USER_COLUMN
from cicerone.io.user_lookup import OCCURRED_AT_COLUMN, filter_rows_for_user

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

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame: ...

    def delete_events_for_user(self, user_id: str, *, item_id: str | None = None) -> int: ...


def require_id(row: dict[str, Any], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def normalize_event_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload[USER_COLUMN] = require_id(payload, USER_COLUMN)
    payload[ITEM_COLUMN] = require_id(payload, ITEM_COLUMN)
    payload[EVENT_TYPE_COLUMN] = require_id(payload, EVENT_TYPE_COLUMN)
    if QUANTITY_COLUMN not in payload or payload[QUANTITY_COLUMN] in (None, ""):
        payload[QUANTITY_COLUMN] = 1
    return payload


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
        if hasattr(value, "isoformat") and not isinstance(value, str):
            value = value.isoformat()
        if isinstance(value, float) and value != value:  # NaN
            value = None
        out[str(key)] = value
    return out
