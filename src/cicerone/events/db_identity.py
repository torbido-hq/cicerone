"""Identity and cursor helpers for DB watermark EventSource."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, NamedTuple

from cicerone.events.base import NormalizedEvent
from cicerone.events.normalize import (
    EventNormalizeError,
    normalize_event,
    parse_occurred_at,
    supplied_event_id,
)
from cicerone.option_parse import NUMBER_PARSE_ERRORS

ROW_IDENTITY_KEYS = ("id", "rowid", "ctid")
# Per-row cursor key computed in SQL; Python stores it verbatim and never builds one.
CURSOR_COLUMN = "cicerone_cursor"
_IDENTITY = "cicerone_identity"
_PAYLOAD_KEYS = ("user_id", "item_id", "event_type", "quantity")


class _CursorSql(NamedTuple):
    row_identity: str
    row_identity_key: str
    supplied_id: str
    timestamp: str
    identity: str
    sep: str
    collate: str
    prefilter: str


# Characters str.strip() removes (U+3000 is the highest whitespace code point), bound as :id_whitespace.
ID_WHITESPACE = "".join(char for char in map(chr, range(0x3001)) if char.isspace())
# Key = UTC occurred_at at full precision, a separator, then the row identity; `id:<1-20 digits>` sorts
# numerically, followed by the raw identity so the key stays distinct. SQLite date functions keep only
# milliseconds, so its fraction is read from the stored text (up to 9 digits).
_CURSOR_SQL = {
    "sqlite": _CursorSql(
        row_identity="rowid",
        row_identity_key="'rowid:' || printf('%020d', {col})",
        supplied_id=(
            "CASE WHEN typeof({col}) IN ('integer', 'real') AND {col} = 0 THEN NULL"
            " ELSE NULLIF(trim(CAST({col} AS TEXT), :id_whitespace), '') END"
        ),
        timestamp=(
            "strftime('%Y-%m-%d %H:%M:%S', {at}) || '.' || substr(CASE WHEN substr({at}, 20, 1) = '.'"
            " THEN replace(substr({at}, 21), ltrim(substr({at}, 21), '0123456789'), '') ELSE '' END"
            " || '000000000', 1, 9)"
        ),
        identity=(
            f"CASE WHEN {_IDENTITY} GLOB 'id:[0-9]*' AND substr({_IDENTITY}, 4) NOT GLOB '*[^0-9]*'"
            f" AND length({_IDENTITY}) <= 23 THEN 'id:' || substr('00000000000000000000'"
            f" || substr({_IDENTITY}, 4), -20) || char(31) || {_IDENTITY} ELSE {_IDENTITY} END"
        ),
        sep="char(31)",
        collate="BINARY",
        prefilter="",
    ),
    "postgresql": _CursorSql(
        row_identity="ctid",
        row_identity_key="'ctid:' || CAST({col} AS TEXT)",
        supplied_id=(
            "CASE WHEN pg_typeof({col}) IN ('smallint'::regtype, 'integer'::regtype, 'bigint'::regtype,"
            " 'numeric'::regtype, 'real'::regtype, 'double precision'::regtype)"
            " AND CAST({col} AS TEXT) ~ '^-?0+(\\.0+)?$' THEN NULL"
            " ELSE NULLIF(btrim(CAST({col} AS TEXT), :id_whitespace), '') END"
        ),
        timestamp="to_char(CAST({at} AS timestamptz) AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.US')",
        identity=(
            f"CASE WHEN {_IDENTITY} ~ '^id:[0-9]{{1,20}}$'"
            f" THEN 'id:' || lpad(substr({_IDENTITY}, 4), 20, '0') || chr(31) || {_IDENTITY}"
            f" ELSE {_IDENTITY} END"
        ),
        sep="chr(31)",
        collate='"C"',
        # Implied by the key predicate; lets an occurred_at index serve polls and lag counts.
        prefilter=" WHERE {at} >= CAST(:watermark_at AS timestamptz)",
    ),
}
CURSOR_DIALECTS = frozenset(_CURSOR_SQL)


def row_identity_column(dialect: str) -> str:
    return _CURSOR_SQL[dialect].row_identity


def keyed_source_sql(dialect: str, source: str, columns: Mapping[str, str]) -> str:
    """``source`` projected to ``columns`` (lowercase name -> quoted source column) plus `CURSOR_COLUMN`.

    The identity is the trimmed supplied id, else ``id``, else rowid/ctid, else the payload.
    """
    sql = _CURSOR_SQL[dialect]
    terms = [
        sql.supplied_id.format(col=columns[name])
        for name in ("event_id", "idempotency_key")
        if name in columns
    ]
    if "id" in columns:
        terms.append(f"'id:' || NULLIF(CAST({columns['id']} AS TEXT), '')")
    if sql.row_identity in columns:
        terms.append(sql.row_identity_key.format(col=columns[sql.row_identity]))
    payload = [f"COALESCE(CAST({columns[name]} AS TEXT), '')" for name in _PAYLOAD_KEYS if name in columns]
    terms.append(" || ".join([sql.sep, *(f"{part} || {sql.sep}" for part in payload)]))
    projected = ", ".join(f"{column} AS {name}" for name, column in columns.items())
    identity = terms[0] if len(terms) == 1 else f"COALESCE({', '.join(terms)})"
    prefilter = sql.prefilter.format(at=columns["occurred_at"])
    identified = (
        f"(SELECT {projected}, {identity} AS {_IDENTITY} FROM {source}{prefilter}) AS cicerone_identified"
    )
    key = f"{sql.timestamp.format(at='occurred_at')} || {sql.sep} || {sql.identity}"
    return f"(SELECT {', '.join(columns)}, {key} AS {CURSOR_COLUMN} FROM {identified}) AS cicerone_keyed"


def after_watermark_sql(dialect: str) -> tuple[str, str]:
    """WHERE and ORDER BY for keys after ``:watermark_cursor`` (or, when NULL, ``:watermark_at``)."""
    sql = _CURSOR_SQL[dialect]
    key = f"{CURSOR_COLUMN} COLLATE {sql.collate}"
    start = sql.timestamp.format(at=":watermark_at")
    return f"{key} > COALESCE(:watermark_cursor, {start})", f"{key} ASC"


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


def _stable_event_id(payload: dict[str, Any], occurred_at: datetime) -> str:
    existing = supplied_event_id(payload)
    if existing is not None:
        return existing
    if "event_id" in payload:
        # Keyless row of an event_id-column source (rejected).
        return ""
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


def payload_identity(payload: dict[str, Any]) -> tuple[datetime, str]:
    """``(occurred_at, event_id)`` of a raw row without full normalize_event."""
    occurred_at = db_occurred_at(payload.get("occurred_at"))
    return occurred_at, _stable_event_id(payload, occurred_at)


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
