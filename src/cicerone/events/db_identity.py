"""Identity and cursor helpers for DB watermark EventSource."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from cicerone.events.base import NormalizedEvent
from cicerone.events.normalize import (
    EventNormalizeError,
    normalize_event,
    parse_occurred_at,
    supplied_event_id,
)
from cicerone.option_parse import NUMBER_PARSE_ERRORS

ROW_IDENTITY_KEYS = ("id", "rowid", "ctid")
_CTID_TUPLE = re.compile(r"\((\d+),(\d+)\)")
_IDENTITY_SORT_SEP = "\x1f"
_EVENT_KEY = "cicerone_event_key"
_SORT_KEY = "cicerone_sort_key"
_SQL_COLLATE = {"sqlite": "BINARY", "postgresql": '"C"'}
# SQLite/Postgres expression matching `_identity_bind_sort_key`; `event_id` stands for the key expression.
# Only `id:` / `rowid:` suffixes of 1-20 ASCII digits are zero-padded; other ids sort as plain text.
_SQLITE_CTID_INNER = "replace(replace(replace(event_id, ' ', ''), 'ctid:(', ''), ')', '')"
_SQLITE_IDENTITY_SORT = (
    "CASE"
    " WHEN substr(event_id, 1, 3) = 'id:' AND substr(event_id, 4) GLOB '[0-9]*'"
    " AND substr(event_id, 4) NOT GLOB '*[^0-9]*' AND length(substr(event_id, 4)) <= 20"
    " THEN 'id:' || char(31) || substr('00000000000000000000' || substr(event_id, 4), -20)"
    " WHEN substr(event_id, 1, 3) = 'id:'"
    " THEN 'id:' || char(31) || substr(event_id, 4)"
    " WHEN substr(event_id, 1, 6) = 'rowid:' AND substr(event_id, 7) GLOB '[0-9]*'"
    " AND substr(event_id, 7) NOT GLOB '*[^0-9]*' AND length(substr(event_id, 7)) <= 20"
    " THEN 'rowid:' || char(31) || substr('00000000000000000000' || substr(event_id, 7), -20)"
    " WHEN substr(event_id, 1, 6) = 'rowid:'"
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
    " WHEN event_id ~ '^id:[0-9]{1,20}$'"
    " THEN 'id:' || chr(31) || lpad(substring(event_id from 4), 20, '0')"
    " WHEN event_id LIKE 'id:%'"
    " THEN 'id:' || chr(31) || substring(event_id from 4)"
    " WHEN event_id ~ '^rowid:[0-9]{1,20}$'"
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
_IDENTITY_SORT_SQL = {"sqlite": _SQLITE_IDENTITY_SORT, "postgresql": _POSTGRES_IDENTITY_SORT}
# Characters str.strip() removes (U+3000 is the highest whitespace code point), bound as :id_whitespace.
ID_WHITESPACE = "".join(char for char in map(chr, range(0x3001)) if char.isspace())
# `supplied_event_id` for one column: numeric 0 is missing, text is stripped, empty is missing.
_SUPPLIED_ID_SQL = {
    "sqlite": (
        "CASE WHEN typeof({col}) IN ('integer', 'real') AND {col} = 0 THEN NULL"
        " ELSE NULLIF(trim(CAST({col} AS TEXT), :id_whitespace), '') END"
    ),
    "postgresql": (
        "CASE WHEN pg_typeof({col}) IN ('smallint'::regtype, 'integer'::regtype, 'bigint'::regtype,"
        " 'numeric'::regtype, 'real'::regtype, 'double precision'::regtype)"
        " AND CAST({col} AS TEXT) ~ '^-?0+(\\.0+)?$' THEN NULL"
        " ELSE NULLIF(btrim(CAST({col} AS TEXT), :id_whitespace), '') END"
    ),
}


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


def _padded_digits(rest: str) -> str | None:
    """``rest`` zero-padded when it is 1-20 ASCII digits (as the SQL sort pads it), else None."""
    if rest.isascii() and rest.isdigit() and len(rest) <= 20:
        return rest.zfill(20)
    return None


def _identity_sort_key(event_id: str) -> tuple[str, tuple[str, ...]]:
    """Numeric order for synthetic identities; suffixes stay strings."""
    for prefix in ("id:", "rowid:"):
        if event_id.startswith(prefix):
            rest = event_id[len(prefix) :]
            return (prefix, (_padded_digits(rest) or rest,))
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
        return _padded_digits(event_id[len(prefix) :]) is not None
    if prefix == "ctid:":
        return _CTID_TUPLE.fullmatch(event_id[5:].replace(" ", "")) is not None
    return False


def identity_bind_sort_key(event_id: str) -> str:
    prefix, parts = _identity_sort_key(event_id)
    return prefix + _IDENTITY_SORT_SEP + _IDENTITY_SORT_SEP.join(parts)


def cursor_source_sql(dialect: str, source: str, columns: Mapping[str, str]) -> tuple[str, str] | None:
    """``source`` plus each row's cursor sort key, and the collated key to filter and order by.

    ``columns`` maps the selected lowercase names (with ``rowid`` / ``ctid`` when selected) to quoted
    source columns. None when SQL can't rebuild `_stable_event_id` (other dialects, payload hashes).
    """
    if dialect not in _IDENTITY_SORT_SQL:
        return None
    terms = [
        _SUPPLIED_ID_SQL[dialect].format(col=columns[name])
        for name in ("event_id", "idempotency_key")
        if name in columns
    ]
    if "id" in columns:
        terms.append(f"'id:' || NULLIF(CAST({columns['id']} AS TEXT), '')")
    if "rowid" in columns:
        terms.append(f"'rowid:' || printf('%020d', {columns['rowid']})")
    if "ctid" in columns:
        terms.append(f"'ctid:' || CAST({columns['ctid']} AS TEXT)")
    if "event_id" in columns:
        terms.append("''")
    elif not {"rowid", "ctid"} & columns.keys():
        return None
    exposed = "".join(f", {columns[name]}" for name in ("rowid", "ctid") if name in columns)
    key = terms[0] if len(terms) == 1 else f"COALESCE({', '.join(terms)})"
    keyed = f"(SELECT *{exposed}, {key} AS {_EVENT_KEY} FROM {source}) AS cicerone_keyed"
    sort = _IDENTITY_SORT_SQL[dialect].replace("event_id", _EVENT_KEY)
    return (
        f"(SELECT *, {sort} AS {_SORT_KEY} FROM {keyed}) AS cicerone_sorted",
        f"{_SORT_KEY} COLLATE {_SQL_COLLATE[dialect]}",
    )


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
    if "event_id" in payload:
        # Keyless row of an event_id-column source (rejected): '' is where the SQL cursor places it.
        return ""
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
    if "event_id" in payload and supplied_event_id(payload) is None:
        raise EventNormalizeError("event_id or idempotency_key is required when the source has event_id")
    occurred_at = db_occurred_at(payload.get("occurred_at"))
    return normalize_event(
        {
            **payload,
            "occurred_at": occurred_at.isoformat(),
            "event_id": _stable_event_id(payload, occurred_at),
        }
    )
