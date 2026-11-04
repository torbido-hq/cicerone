"""DB watermark EventSource (poll new interaction rows after a cursor)."""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, Literal

from sqlalchemy import Connection, Engine, make_url, text

from cicerone.config import ConfigError
from cicerone.events.base import EventSource, EventSourceError, EventSourceHealth, NormalizedEvent
from cicerone.events.db_identity import (
    CURSOR_COLUMN,
    CURSOR_DIALECTS,
    ID_WHITESPACE,
    ROW_IDENTITY_KEYS,
    after_watermark_sql,
    db_occurred_at,
    keyed_source_sql,
    payload_identity,
    row_identity_column,
    row_to_event,
)
from cicerone.events.normalize import EventNormalizeError
from cicerone.events.state_file import read_json_object_file, write_json_object_file
from cicerone.io.db_errors import MISSING_TABLE_ERRORS, SQL_READ_ERRORS, is_missing_column_error
from cicerone.io.db_store import DEFAULT_EVENTS_TABLE
from cicerone.io.engines import engine_for, quote_identifier, release_engine, sql_source
from cicerone.io.options import readonly_select, require_option, table_option

logger = logging.getLogger(__name__)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_EVENTS_QUERY_ALIAS = "cicerone_events_src"
# Columns an invalid row is reported by; user and item ids stay out of the log.
_ROW_LOG_KEYS = ("event_id", "idempotency_key", *ROW_IDENTITY_KEYS, "occurred_at")
# Columns consumed by _row_to_event / normalize_event (optional ones omitted if absent).
_FETCH_COLUMNS = (
    "user_id",
    "item_id",
    "event_type",
    "quantity",
    "occurred_at",
    "event_id",
    "idempotency_key",
    "id",
)


def _validate_events_query(query: str) -> str:
    return readonly_select(query, option="events.options.events_query")


def _rejection(payload: dict[str, Any], exc: EventNormalizeError) -> str:
    identity = {name: payload[name] for name in _ROW_LOG_KEYS if payload.get(name) is not None}
    return f"({exc}): {json.dumps(identity, default=str):.200}"


def _row_event(payload: dict[str, Any]) -> NormalizedEvent:
    try:
        return row_to_event(payload)
    except EventNormalizeError as exc:
        reason = _rejection(payload, exc)
    # Placed on the cursor so the worker acks it in order; raises when occurred_at is unusable.
    occurred_at, event_id = payload_identity(payload)
    return NormalizedEvent("", "", "", 0, occurred_at, event_id, rejected=reason)


@contextmanager
def _utc_connection(engine: Engine) -> Iterator[Connection]:
    with engine.connect() as conn:
        if engine.dialect.name == "postgresql":
            # Naive occurred_at values are UTC (db_occurred_at), so compare them in UTC, not the session zone.
            conn.execute(text("SET LOCAL TIME ZONE 'UTC'"))
        elif engine.dialect.name in ("mysql", "mariadb"):
            # Session-wide on MySQL (no SET LOCAL): TIMESTAMP columns read in UTC, DATETIME as stored.
            conn.execute(text("SET time_zone = '+00:00'"))
        yield conn


@dataclass(slots=True)
class _Delivery:
    event: NormalizedEvent
    state: Literal["pending", "acked", "nacked"]
    polled_at: float


class DbEventSource(EventSource):
    """Poll ``events`` (or trusted ``events_query``) after a watermark; advance on ``ack``.

    ``events_query`` is deploy-time config interpolated into SQL (read-only SELECT
    validated at construction); do not pass untrusted client input.
    """

    def __init__(self, options: dict[str, Any] | None = None):
        options = dict(options or {})
        self._database_url = require_option(options, "database_url", "db")
        dialect = make_url(self._database_url).get_backend_name()
        if dialect not in CURSOR_DIALECTS:
            raise ConfigError(
                f"events db source supports sqlite, postgresql and mysql/mariadb, not {dialect!r}"
            )
        self._events_table = table_option(options, "events_table", DEFAULT_EVENTS_TABLE)
        self._events_query = options.get("events_query")
        if self._events_query is not None:
            if not isinstance(self._events_query, str):
                raise ValueError("events.options.events_query must be a string")
            self._events_query = _validate_events_query(self._events_query)
        self._watermark_path = Path(options["watermark_path"]) if options.get("watermark_path") else None
        initial = options.get("initial_watermark")
        self._watermark_at = db_occurred_at(initial) if initial not in (None, "") else _EPOCH
        self._watermark_event_id = ""
        # SQL cursor key of the last acked row; None starts from the beginning of _watermark_at.
        self._watermark_cursor: str | None = None
        self._engine: Engine | None = None
        self._lock = threading.Lock()
        # Polled events by SQL cursor key, in key order; the watermark advances over the leading acked run.
        self._in_flight: OrderedDict[str, _Delivery] = OrderedDict()
        self._connected = False
        self._last_event_at: datetime | None = None
        self._keyed_source: str | None = None

    def connect(self) -> None:
        created = False
        connected = False
        try:
            with self._lock:
                if self._engine is None:
                    self._engine = engine_for(self._database_url)
                    created = True
                self._load_watermark_unlocked()
                self._connected = True
                connected = True
        finally:
            if created and not connected:
                with self._lock:
                    self._engine = None
                    self._connected = False
                release_engine(self._database_url)

    def close(self) -> None:
        with self._lock:
            held = self._engine is not None
            self._engine = None
            self._connected = False
            self._keyed_source = None
        if held:
            release_engine(self._database_url)

    # Polls never overlap: the worker is the only caller, from tick() under _tick_guard.
    def poll(self, max_events: int = 100) -> Sequence[NormalizedEvent]:
        if max_events < 1:
            return []
        with self._lock:
            if self._engine is None:
                raise EventSourceError("DbEventSource.connect() required before poll")
            watermark = (self._watermark_cursor, self._watermark_at)
            limit = max_events + len(self._in_flight) + 8
            engine = self._engine
        rows = self._fetch_rows(engine, *watermark, limit=limit)
        candidates: list[tuple[str, NormalizedEvent]] = []
        for payload in rows:
            cursor = payload.pop(CURSOR_COLUMN)
            try:
                candidates.append((cursor, _row_event(payload)))
            except EventNormalizeError as exc:
                logger.warning("Skipping invalid event row %s", _rejection(payload, exc))

        out: list[NormalizedEvent] = []
        with self._lock:
            self._drop_vanished_heads_unlocked(candidates, complete=len(rows) < limit)
            for cursor, event in candidates:
                if len(out) >= max_events:
                    break
                if cursor <= (self._watermark_cursor or ""):
                    continue
                delivery = self._in_flight.get(cursor)
                if delivery is None:
                    self._in_flight[cursor] = _Delivery(event, "pending", monotonic())
                elif delivery.state == "nacked":
                    delivery.event, delivery.state = event, "pending"
                else:
                    continue
                self._last_event_at = event.occurred_at
                out.append(event)
            if out:
                # Keys compare as code points, the order of SQL's binary collation.
                self._in_flight = OrderedDict(sorted(self._in_flight.items()))
        return out

    def nack(self, events: Sequence[NormalizedEvent]) -> Sequence[NormalizedEvent]:
        with self._lock:
            self._mark_pending_unlocked((event.event_id for event in events), "nacked")
        return ()

    def ack(self, event_ids: Sequence[str]) -> Sequence[str]:
        with self._lock:
            confirmed = self._mark_pending_unlocked(map(str, event_ids), "acked")
            self._advance_unlocked()
        return tuple(confirmed)

    def _mark_pending_unlocked(
        self, event_ids: Iterable[str], state: Literal["acked", "nacked"]
    ) -> list[str]:
        """Move every pending delivery of ``event_ids`` to ``state``; the ids that had one, in order."""
        wanted = dict.fromkeys(event_ids)
        marked: set[str] = set()
        for delivery in self._in_flight.values():
            if delivery.state == "pending" and delivery.event.event_id in wanted:
                delivery.state = state
                marked.add(delivery.event.event_id)
        return [event_id for event_id in wanted if event_id in marked]

    def _advance_unlocked(self) -> None:
        advanced = False
        while self._in_flight and next(iter(self._in_flight.values())).state == "acked":
            cursor, delivery = self._in_flight.popitem(last=False)
            self._watermark_at = delivery.event.occurred_at
            self._watermark_event_id = delivery.event.event_id
            self._watermark_cursor = cursor
            advanced = True
        if advanced:
            self._persist_watermark_unlocked()

    def _drop_vanished_heads_unlocked(
        self, candidates: Sequence[tuple[str, NormalizedEvent]], *, complete: bool
    ) -> None:
        # A page cut short before the head's key says nothing about the head's row.
        returned = {cursor for cursor, _event in candidates}
        reached = candidates[-1][0] if candidates else None
        while self._in_flight:
            cursor, delivery = next(iter(self._in_flight.items()))
            if cursor in returned:
                return
            if not complete and (reached is None or reached <= cursor):
                return
            logger.warning(
                "Dropping in-flight event %s at %s: its row is no longer returned after the watermark",
                delivery.event.event_id,
                delivery.event.occurred_at.isoformat(),
            )
            del self._in_flight[cursor]
            self._advance_unlocked()

    def health(self) -> EventSourceHealth:
        with self._lock:
            detail = f"db watermark={self._watermark_at.isoformat()}"
            if self._in_flight:
                head = next(iter(self._in_flight.values()))
                acked = sum(1 for delivery in self._in_flight.values() if delivery.state == "acked")
                detail += f" acked_behind_head={acked} head_wait={monotonic() - head.polled_at:.0f}s"
            connected = self._connected
            last_event_at = self._last_event_at
            engine = self._engine
            watermark = (self._watermark_cursor, self._watermark_at)
            in_flight = len(self._in_flight)
        # COUNT already includes unacked rows past the watermark — do not add in_flight.
        lag: int | None = in_flight
        if engine is not None:
            try:
                lag = self._count_after(engine, *watermark)
            except SQL_READ_ERRORS:
                logger.exception("Failed to estimate DB event lag")
        return EventSourceHealth(
            connected=connected,
            lag=lag,
            last_event_at=last_event_at,
            detail=detail,
        )

    def _from_clause(self, engine: Engine) -> str:
        return sql_source(engine, self._events_query, self._events_table, _EVENTS_QUERY_ALIAS)

    def _source_sql(self, engine: Engine) -> str:
        """The source projected to the columns events use, plus each row's `CURSOR_COLUMN` key."""
        with self._lock:
            if self._keyed_source is not None:
                return self._keyed_source
        source = self._from_clause(engine)
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {source} LIMIT 0"))
            by_lower = {str(name).lower(): str(name) for name in result.keys()}  # noqa: SIM118 — Result.keys()
        if "occurred_at" not in by_lower:
            raise ConfigError(
                f"events DB source must expose an occurred_at column for watermark polling (from {source})"
            )
        if CURSOR_COLUMN in by_lower:
            raise ConfigError(
                f"events DB source must not expose a column named {CURSOR_COLUMN} (from {source})"
            )
        # Alias mixed-case source columns to the lowercase names events are normalised from.
        columns = {
            name: quote_identifier(engine, by_lower[name]) for name in _FETCH_COLUMNS if name in by_lower
        }
        identity = None if self._events_query else _table_row_identity(engine, source)
        if identity is not None:
            columns[identity] = identity
        keyed = keyed_source_sql(engine.dialect.name, source, columns)
        with self._lock:
            if self._keyed_source is None:
                self._keyed_source = keyed
            return self._keyed_source

    def _after_watermark(
        self, engine: Engine, cursor: str | None, watermark_at: datetime
    ) -> tuple[str, str, str, dict[str, Any]]:
        """FROM, WHERE and ORDER BY for rows whose SQL key is after the watermark, and their params."""
        where, order = after_watermark_sql(engine.dialect.name)
        params = {
            "watermark_cursor": cursor,
            # Naive UTC text: MariaDB and older MySQL reject offsets in datetime literals.
            "watermark_at": watermark_at.astimezone(UTC).replace(tzinfo=None).isoformat(" "),
            "id_whitespace": ID_WHITESPACE,
        }
        return self._source_sql(engine), where, order, params

    def _fetch_rows(
        self, engine: Engine, cursor: str | None, watermark_at: datetime, *, limit: int
    ) -> list[dict[str, Any]]:
        source, where, order, params = self._after_watermark(engine, cursor, watermark_at)
        sql = text(f"SELECT * FROM {source} WHERE {where} ORDER BY {order} LIMIT :limit")
        with _utc_connection(engine) as conn:
            result = conn.execute(sql, {**params, "limit": max(limit, 1)})
            return [dict(row) for row in result.mappings()]

    def _count_after(self, engine: Engine, cursor: str | None, watermark_at: datetime) -> int:
        source, where, _order, params = self._after_watermark(engine, cursor, watermark_at)
        sql = text(f"SELECT COUNT(*) AS n FROM {source} WHERE {where}")
        with _utc_connection(engine) as conn:
            return int(conn.execute(sql, params).scalar() or 0)

    def _load_watermark_unlocked(self) -> None:
        if self._watermark_path is None:
            return
        try:
            raw = read_json_object_file(self._watermark_path)
            if raw is None:
                return
            watermark_at = self._watermark_at
            occurred_at = raw.get("occurred_at")
            if occurred_at:
                watermark_at = db_occurred_at(occurred_at)
            # A file without a cursor (written before it existed) resumes at the start of occurred_at.
            cursor = raw.get("cursor")
        except (OSError, ValueError):
            logger.exception(
                "Ignoring corrupt watermark file %s; keeping watermark %s",
                self._watermark_path,
                self._watermark_at.isoformat(),
            )
            return
        self._watermark_at = watermark_at
        self._watermark_cursor = str(cursor) if cursor else None

    def _persist_watermark_unlocked(self) -> None:
        if self._watermark_path is None:
            return
        write_json_object_file(
            self._watermark_path,
            {
                "occurred_at": self._watermark_at.isoformat(),
                "event_id": self._watermark_event_id,
                "cursor": self._watermark_cursor,
            },
        )


def _table_row_identity(engine: Engine, source: str) -> str | None:
    """SQLite ``rowid`` / Postgres ``ctid`` when the table has one (not WITHOUT ROWID tables or views)."""
    name = row_identity_column(engine.dialect.name)
    if name is None:
        return None
    try:
        with engine.connect() as conn:
            conn.execute(text(f"SELECT {name} FROM {source} LIMIT 0"))
    except MISSING_TABLE_ERRORS as exc:
        if not is_missing_column_error(exc):
            raise
        return None
    return name
