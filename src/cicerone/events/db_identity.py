"""Identity and cursor helpers for DB watermark EventSource."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from cicerone.events.base import NormalizedEvent
from cicerone.events.normalize import normalize_event, parse_occurred_at

_ROW_IDENTITY_KEYS = ("id", "rowid", "ctid")
_CTID_TUPLE = re.compile(r"\((\d+),(\d+)\)")
_IDENTITY_SORT_SEP = "\x1f"
_SQLITE_OCCURRED_AT = "strftime('%Y-%m-%d %H:%M:%f', occurred_at)"
_SQLITE_WATERMARK_AT = "strftime('%Y-%m-%d %H:%M:%f', :watermark_at)"
# SQLite/Postgres expression matching `_identity_bind_sort_key`.
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


def _db_occurred_at(value: Any) -> datetime:
    # SQLAlchemy/SQLite often yield naive datetimes that are UTC in practice.
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return parse_occurred_at(value)


def _row_identity(payload: dict[str, Any]) -> str | None:
    for key in _ROW_IDENTITY_KEYS:
        value = payload.get(key)
        if value in (None, ""):
            continue
        if key == "rowid":
            try:
                return f"{key}:{int(value):020d}"
            except (TypeError, ValueError):
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


def _identity_bind_sort_key(event_id: str) -> str:
    prefix, parts = _identity_sort_key(event_id)
    return prefix + _IDENTITY_SORT_SEP + _IDENTITY_SORT_SEP.join(parts)


def _identity_sql_sort_expr(dialect: str) -> str | None:
    if dialect == "sqlite":
        return _SQLITE_IDENTITY_SORT
    if dialect == "postgresql":
        return _POSTGRES_IDENTITY_SORT
    return None


def _occurred_at_predicates(dialect: str) -> tuple[str, str, str]:
    """SQL fragments: later-than watermark, same watermark, order-by occurred_at."""
    if dialect == "sqlite":
        return (
            f"{_SQLITE_OCCURRED_AT} > {_SQLITE_WATERMARK_AT}",
            f"{_SQLITE_OCCURRED_AT} = {_SQLITE_WATERMARK_AT}",
            f"{_SQLITE_OCCURRED_AT} ASC",
        )
    return ("occurred_at > :watermark_at", "occurred_at = :watermark_at", "occurred_at ASC")


def _cursor_tuple(occurred_at: datetime, event_id: str) -> tuple[datetime, tuple[str, tuple[str, ...]]]:
    return (occurred_at, _identity_sort_key(event_id))


def _stable_event_id(payload: dict[str, Any], occurred_at: datetime) -> str:
    existing = payload.get("event_id") or payload.get("idempotency_key")
    if existing not in (None, ""):
        return str(existing)
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


def _cursor_key(event: NormalizedEvent) -> tuple[datetime, tuple[str, tuple[str, ...]]]:
    return _cursor_tuple(event.occurred_at, event.event_id)


def _cursor_key_from_payload(payload: dict[str, Any]) -> tuple[datetime, tuple[str, tuple[str, ...]]]:
    """Lag path: parse occurred_at + stable id without full normalize_event."""
    occurred_at = _db_occurred_at(payload.get("occurred_at"))
    return _cursor_tuple(occurred_at, _stable_event_id(payload, occurred_at))


def _row_to_event(payload: dict[str, Any]) -> NormalizedEvent:
    occurred_at = _db_occurred_at(payload.get("occurred_at"))
    return normalize_event(
        {
            **payload,
            "occurred_at": occurred_at.isoformat(),
            "event_id": _stable_event_id(payload, occurred_at),
        }
    )
