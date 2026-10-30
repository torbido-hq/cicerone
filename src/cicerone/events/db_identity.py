"""Identity and cursor helpers for DB watermark EventSource."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from cicerone.events.base import NormalizedEvent
from cicerone.events.normalize import normalize_event, parse_occurred_at, supplied_event_id
from cicerone.option_parse import NUMBER_PARSE_ERRORS

ROW_IDENTITY_KEYS = ("id", "rowid", "ctid")
_CTID_TUPLE = re.compile(r"\((\d+),(\d+)\)")
_IDENTITY_SORT_SEP = "\x1f"
# SQLite/Postgres expression matching `_identity_bind_sort_key`; `event_id` stands for the quoted column.
_SQLITE_CTID_INNER = "replace(replace(replace(event_id, ' ', ''), 'ctid:(', ''), ')', '')"
_SQLITE_IDENTITY_SORT = (
    "CASE"
    " WHEN event_id GLOB 'id:[0-9]*' AND substr(event_id, 4) NOT GLOB '*[^0-9]*'"
    " THEN 'id:' || char(31) || printf('%020d', CAST(substr(event_id, 4) AS INTEGER))"
    " WHEN event_id LIKE 'id:%'"
    " THEN 'id:' || char(31) || substr(event_id, 4)"
    " WHEN event_id GLOB 'rowid:[0-9]*' AND substr(event_id, 7) NOT GLOB '*[^0-9]*'"
    " THEN 'rowid:' || char(31) || printf('%020d', CAST(substr(event_id, 7) AS INTEGER))"
    " WHEN event_id LIKE 'rowid:%'"
    " THEN 'rowid:' || char(31) || substr(event_id, 7)"
    f" WHEN replace(event_id, ' ', '') LIKE 'ctid:(%,%)'"
    f" THEN 'ctid:' || char(31)"
    f" || printf('%020d', CAST(substr({_SQLITE_CTID_INNER}, 1, instr({_SQLITE_CTID_INNER}, ',') - 1)"
    f" AS INTEGER))"
    f" || char(31)"
    f" || printf('%020d', CAST(substr({_SQLITE_CTID_INNER}, instr({_SQLITE_CTID_INNER}, ',') + 1)"
    f" AS INTEGER))"
    " WHEN event_id LIKE 'ctid:%'"
    " THEN 'ctid:' || char(31) || char(31) || replace(substr(event_id, 6), ' ', '')"
    " ELSE char(31) || event_id END"
)
_POSTGRES_IDENTITY_SORT = (
    "CASE"
    " WHEN event_id ~ '^id:[0-9]+$'"
    " THEN 'id:' || chr(31) || lpad(substring(event_id from 4), 20, '0')"
    " WHEN event_id LIKE 'id:%'"
    " THEN 'id:' || chr(31) || substring(event_id from 4)"
    " WHEN event_id ~ '^rowid:[0-9]+$'"
    " THEN 'rowid:' || chr(31) || lpad(substring(event_id from 7), 20, '0')"
    " WHEN event_id LIKE 'rowid:%'"
    " THEN 'rowid:' || chr(31) || substring(event_id from 7)"
    " WHEN replace(event_id, ' ', '') ~ '^ctid:\\([0-9]+,[0-9]+\\)$'"
    " THEN 'ctid:' || chr(31)"
    " || lpad((regexp_match(replace(event_id, ' ', ''),"
    " '^ctid:\\(([0-9]+),([0-9]+)\\)$'))[1], 20, '0')"
    " || chr(31)"
    " || lpad((regexp_match(replace(event_id, ' ', ''),"
    " '^ctid:\\(([0-9]+),([0-9]+)\\)$'))[2], 20, '0')"
    " WHEN event_id LIKE 'ctid:%'"
    " THEN 'ctid:' || chr(31) || chr(31) || replace(substring(event_id from 6), ' ', '')"
    " ELSE chr(31) || event_id END"
)


def db_occurred_at(value: Any) -> datetime:
    # SQLAlchemy/SQLite often yield naive datetimes that are UTC in practice.
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return parse_occurred_at(value)


def _row_identity(payload: dict[str, Any]) -> str | None:
    for key in ROW_IDENTITY_KEYS:
        value = payload.get(key)
        if value in (None, ""):
            continue
        if key == "rowid":
            try:
                return f"{key}:{int(value):020d}"
            except NUMBER_PARSE_ERRORS:
                return f"{key}:{value}"
        return f"{key}:{value}"
    return None


def _identity_sort_key(event_id: str) -> tuple[str, tuple[str, ...]]:
    """Numeric order for synthetic identities; suffixes stay strings."""
    for prefix in ("id:", "rowid:"):
        if event_id.startswith(prefix):
            rest = event_id[len(prefix) :]
            try:
                return (prefix, (f"{int(rest):020d}",))
            except ValueError:
                return (prefix, (rest,))
    if event_id.startswith("ctid:"):
        rest = event_id[5:].replace(" ", "")
        match = _CTID_TUPLE.fullmatch(rest)
        if match:
            return ("ctid:", (f"{int(match.group(1)):020d}", f"{int(match.group(2)):020d}"))
        return ("ctid:", ("", rest))
    return ("", (event_id,))


def _is_numeric_identity(event_id: str) -> bool:
    """True when SQL text order would disagree with `_identity_sort_key`."""
    prefix, _parts = _identity_sort_key(event_id)
    if prefix in ("id:", "rowid:"):
        try:
            int(event_id[len(prefix) :])
        except ValueError:
            return False
        return True
    if prefix == "ctid:":
        return _CTID_TUPLE.fullmatch(event_id[5:].replace(" ", "")) is not None
    return False


def identity_bind_sort_key(event_id: str) -> str:
    prefix, parts = _identity_sort_key(event_id)
    return prefix + _IDENTITY_SORT_SEP + _IDENTITY_SORT_SEP.join(parts)


def identity_sql_sort_expr(dialect: str, event_id: str) -> str | None:
    if dialect == "sqlite":
        return _SQLITE_IDENTITY_SORT.replace("event_id", event_id)
    if dialect == "postgresql":
        return _POSTGRES_IDENTITY_SORT.replace("event_id", event_id)
    return None


def occurred_at_predicates(dialect: str, occurred_at: str) -> tuple[str, str, str]:
    """SQL fragments: later-than watermark, same watermark, order-by ``occurred_at`` (a quoted column)."""
    if dialect == "sqlite":
        occurred_at = f"strftime('%Y-%m-%d %H:%M:%f', {occurred_at})"
        watermark_at = "strftime('%Y-%m-%d %H:%M:%f', :watermark_at)"
    else:
        watermark_at = ":watermark_at"
    return (f"{occurred_at} > {watermark_at}", f"{occurred_at} = {watermark_at}", f"{occurred_at} ASC")


def cursor_tuple(occurred_at: datetime, event_id: str) -> tuple[datetime, tuple[str, tuple[str, ...]]]:
    return (occurred_at, _identity_sort_key(event_id))


def _stable_event_id(payload: dict[str, Any], occurred_at: datetime) -> str:
    existing = supplied_event_id(payload)
    if existing is not None:
        return existing
    identity = _row_identity(payload)
    if identity is not None:
        return identity
    quantity = payload.get("quantity", 1)
    return "|".join(
        (
            str(payload.get("user_id", "")),
            str(payload.get("item_id", "")),
            str(payload.get("event_type", "")),
            str(quantity),
            occurred_at.isoformat(),
        )
    )


def cursor_key(event: NormalizedEvent) -> tuple[datetime, tuple[str, tuple[str, ...]]]:
    return cursor_tuple(event.occurred_at, event.event_id)


def payload_cursor(payload: dict[str, Any]) -> tuple[datetime, str]:
    """``(occurred_at, event_id)`` of a raw row without full normalize_event."""
    occurred_at = db_occurred_at(payload.get("occurred_at"))
    return occurred_at, _stable_event_id(payload, occurred_at)


def cursor_key_from_payload(payload: dict[str, Any]) -> tuple[datetime, tuple[str, tuple[str, ...]]]:
    return cursor_tuple(*payload_cursor(payload))


def row_to_event(payload: dict[str, Any]) -> NormalizedEvent:
    occurred_at = db_occurred_at(payload.get("occurred_at"))
    return normalize_event(
        {
            **payload,
            "occurred_at": occurred_at.isoformat(),
            "event_id": _stable_event_id(payload, occurred_at),
        }
    )
