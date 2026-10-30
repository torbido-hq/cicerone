"""DB watermark EventSource (poll new interaction rows after a cursor)."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from sqlalchemy import Connection, Engine, text

from cicerone.config import ConfigError
from cicerone.events.base import EventSource, EventSourceError, EventSourceHealth, NormalizedEvent
from cicerone.events.db_identity import (
    ROW_IDENTITY_KEYS,
    cursor_key,
    cursor_key_from_payload,
    cursor_tuple,
    db_occurred_at,
    identity_bind_sort_key,
    identity_sql_sort_expr,
    occurred_at_predicates,
    payload_cursor,
    row_to_event,
)
from cicerone.events.normalize import EventNormalizeError
from cicerone.events.state_file import read_json_object_file
from cicerone.io.db_errors import SQL_READ_ERRORS
from cicerone.io.db_store import DEFAULT_EVENTS_TABLE
from cicerone.io.engines import engine_for, quote_identifier, release_engine, sql_source
from cicerone.io.options import readonly_select, require_option, table_option

logger = logging.getLogger(__name__)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_EVENTS_QUERY_ALIAS = "cicerone_events_src"
# Bound for health lag when SQL COUNT is unavailable; None means unknown/too large.
_LAG_SCAN_LIMIT = 10_000
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


_T = TypeVar("_T")


def _validate_events_query(query: str) -> str:
    return readonly_select(query, option="events.options.events_query")


def _parsed_rows(
    rows: Sequence[dict[str, Any]],
    parse: Callable[[dict[str, Any]], _T],
    on_invalid: Callable[[dict[str, Any], EventNormalizeError], None],
) -> Iterator[_T]:
    for payload in rows:
        try:
            yield parse(payload)
        except EventNormalizeError as exc:
            on_invalid(payload, exc)


def _rejection(payload: dict[str, Any], exc: EventNormalizeError) -> str:
    identity = {name: payload[name] for name in _ROW_LOG_KEYS if payload.get(name) is not None}
    return f"({exc}): {json.dumps(identity, default=str):.200}"


def _row_event(payload: dict[str, Any]) -> NormalizedEvent:
    try:
        return row_to_event(payload)
    except EventNormalizeError as exc:
        reason = _rejection(payload, exc)
    # Placed on the cursor so the worker acks it in order; raises when occurred_at is unusable.
    occurred_at, event_id = payload_cursor(payload)
    return NormalizedEvent("", "", "", 0, occurred_at, event_id, rejected=reason)


@contextmanager
def _utc_connection(engine: Engine) -> Iterator[Connection]:
    with engine.connect() as conn:
        if engine.dialect.name == "postgresql":
            # Naive occurred_at values are UTC (db_occurred_at), so compare them in UTC, not the session zone.
            conn.execute(text("SET LOCAL TIME ZONE 'UTC'"))
        yield conn


def _log_unplaceable_row(payload: dict[str, Any], exc: EventNormalizeError) -> None:
    logger.warning("Skipping invalid event row %s", _rejection(payload, exc))


class DbEventSource(EventSource):
    """Poll ``events`` (or trusted ``events_query``) after a watermark; advance on ``ack``.

    ``events_query`` is deploy-time config interpolated into SQL (read-only SELECT
    validated at construction); do not pass untrusted client input.
    """

    def __init__(self, options: dict[str, Any] | None = None):
        options = dict(options or {})
        self._database_url = require_option(options, "database_url", "db")
        self._events_table = table_option(options, "events_table", DEFAULT_EVENTS_TABLE)
        self._events_query = options.get("events_query")
        if self._events_query is not None:
            if not isinstance(self._events_query, str):
                raise ValueError("events.options.events_query must be a string")
            self._events_query = _validate_events_query(self._events_query)
        self._watermark_path = Path(options["watermark_path"]) if options.get("watermark_path") else None
        initial = options.get("initial_watermark")
        self._watermark_at = db_occurred_at(initial) if initial not in (None, "") else _EPOCH
        self._watermark_event_id = str(options.get("initial_watermark_event_id") or "")
        self._engine: Engine | None = None
        self._lock = threading.Lock()
        self._in_flight: OrderedDict[str, NormalizedEvent] = OrderedDict()
        self._connected = False
        self._last_event_at: datetime | None = None
        self._source_columns: frozenset[str] | None = None
        self._select_clause: str | None = None
        self._has_event_id_column: bool | None = None

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
            self._source_columns = None
            self._select_clause = None
            self._has_event_id_column = None
        if held:
            release_engine(self._database_url)

    def poll(self, max_events: int = 100) -> Sequence[NormalizedEvent]:
        if max_events < 1:
            return []
        with self._lock:
            if self._engine is None:
                raise EventSourceError("DbEventSource.connect() required before poll")
            watermark_at = self._watermark_at
            watermark_event_id = self._watermark_event_id
            in_flight_count = len(self._in_flight)
            engine = self._engine
        rows = self._fetch_rows(
            engine,
            watermark_at,
            watermark_event_id,
            limit=max_events + in_flight_count + 8,
        )
        candidates = sorted(_parsed_rows(rows, _row_event, _log_unplaceable_row), key=cursor_key)

        out: list[NormalizedEvent] = []
        with self._lock:
            cursor = cursor_tuple(self._watermark_at, self._watermark_event_id)
            for event in candidates:
                if len(out) >= max_events:
                    break
                if cursor_key(event) <= cursor:
                    continue
                if event.event_id in self._in_flight:
                    continue
                self._in_flight[event.event_id] = event
                self._last_event_at = event.occurred_at
                out.append(event)
        return out

    def nack(self, events: Sequence[NormalizedEvent]) -> Sequence[NormalizedEvent]:
        with self._lock:
            for event in events:
                self._in_flight.pop(event.event_id, None)
        return ()

    def ack(self, event_ids: Sequence[str]) -> Sequence[str]:
        confirmed: list[str] = []
        with self._lock:
            advanced = False
            for event_id in event_ids:
                eid = str(event_id)
                event = self._in_flight.pop(eid, None)
                if event is None:
                    continue
                confirmed.append(eid)
                key = cursor_key(event)
                if key > cursor_tuple(self._watermark_at, self._watermark_event_id):
                    self._watermark_at = event.occurred_at
                    self._watermark_event_id = event.event_id
                    advanced = True
            if advanced:
                self._persist_watermark_unlocked()
        return tuple(confirmed)

    def health(self) -> EventSourceHealth:
        with self._lock:
            detail = f"db watermark={self._watermark_at.isoformat()}"
            connected = self._connected
            last_event_at = self._last_event_at
            engine = self._engine
            watermark_at = self._watermark_at
            watermark_event_id = self._watermark_event_id
            in_flight = len(self._in_flight)
        # COUNT already includes unacked rows past the watermark — do not add in_flight.
        lag: int | None = in_flight
        if engine is not None:
            try:
                lag = self._count_after(engine, watermark_at, watermark_event_id)
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

    def _ensure_source_schema(self, engine: Engine) -> tuple[str, bool]:
        with self._lock:
            if self._select_clause is not None and self._has_event_id_column is not None:
                return self._select_clause, self._has_event_id_column
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {self._from_clause(engine)} LIMIT 0"))
            columns = frozenset(map(str, result.keys()))  # noqa: SIM118 — Result.keys(), not dict
        by_lower = {name.lower(): name for name in columns}
        if "occurred_at" not in by_lower:
            raise ConfigError(
                "events DB source must expose an occurred_at column for watermark polling "
                f"(from {self._from_clause(engine)})"
            )
        selected = [name for name in _FETCH_COLUMNS if name in by_lower]
        has_event_id = "event_id" in by_lower
        if not has_event_id and not self._events_query:
            dialect = engine.dialect.name
            extra = None
            if dialect == "sqlite" and "rowid" not in by_lower:
                extra = "rowid"
            elif dialect == "postgresql" and "ctid" not in by_lower:
                extra = "ctid"
            if extra is not None:
                selected.append(extra)
        # Alias mixed-case source columns to the lowercase names events are normalised from.
        select_clause = ", ".join(
            quote_identifier(engine, name)
            if by_lower.get(name, name) == name
            else f"{quote_identifier(engine, by_lower[name])} AS {quote_identifier(engine, name)}"
            for name in selected
        )
        with self._lock:
            if self._select_clause is None or self._has_event_id_column is None:
                self._source_columns = columns
                self._select_clause = select_clause
                self._has_event_id_column = has_event_id
            cached_select = self._select_clause
            cached_has_event_id = self._has_event_id_column
        return cached_select, cached_has_event_id

    def _source_column(self, engine: Engine, name: str) -> str:
        """``name`` as the source spells it, quoted (Postgres keeps mixed case only when quoted)."""
        by_lower = {column.lower(): column for column in self._source_columns or ()}
        return quote_identifier(engine, by_lower.get(name, name))

    def _after_watermark(
        self, engine: Engine, watermark_at: datetime, watermark_event_id: str
    ) -> tuple[str, str, dict[str, Any]]:
        """WHERE and ORDER BY for rows after the (occurred_at, event_id) cursor, and their params."""
        occurred_at = self._source_column(engine, "occurred_at")
        event_id = self._source_column(engine, "event_id")
        identity_sort = identity_sql_sort_expr(engine.dialect.name, event_id)
        if identity_sort is not None:
            later, same, occurred_order = occurred_at_predicates(engine.dialect.name, occurred_at)
            return (
                f"{later} OR ({same} AND ({identity_sort}) > :watermark_sort)",
                f"{occurred_order}, ({identity_sort}) ASC",
                {"watermark_at": watermark_at, "watermark_sort": identity_bind_sort_key(watermark_event_id)},
            )
        # Match ack cursor (occurred_at, event_id) so same-timestamp pages cannot skip rows.
        return (
            f"{occurred_at} > :watermark_at OR "
            f"({occurred_at} = :watermark_at AND {event_id} > :watermark_event_id)",
            f"{occurred_at} ASC, {event_id} ASC",
            {"watermark_at": watermark_at, "watermark_event_id": watermark_event_id},
        )

    def _fetch_rows(
        self,
        engine: Engine,
        watermark_at: datetime,
        watermark_event_id: str,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        select_clause, has_event_id = self._ensure_source_schema(engine)
        if has_event_id:
            where, order, params = self._after_watermark(engine, watermark_at, watermark_event_id)
        else:
            occurred_at = self._source_column(engine, "occurred_at")
            where, order = f"{occurred_at} >= :watermark", f"{occurred_at} ASC"
            params = {"watermark": watermark_at}
        sql = text(
            f"SELECT {select_clause} FROM {self._from_clause(engine)} "
            f"WHERE {where} ORDER BY {order} LIMIT :limit"
        )
        with _utc_connection(engine) as conn:
            result = conn.execute(sql, {**params, "limit": max(limit, 1)})
            return [dict(row) for row in result.mappings()]

    def _count_after(self, engine: Engine, watermark_at: datetime, watermark_event_id: str) -> int | None:
        _select_clause, has_event_id = self._ensure_source_schema(engine)
        # Prefer SQL COUNT when event_id exists. SQLite text/datetime binding makes
        # equality unreliable, so scan rows there (still on the narrow projection).
        if has_event_id and engine.dialect.name != "sqlite":
            try:
                return self._count_after_sql(engine, watermark_at, watermark_event_id)
            except SQL_READ_ERRORS:
                logger.debug("SQL lag COUNT unavailable; scanning event rows", exc_info=True)
        return self._count_after_rows(engine, watermark_at, watermark_event_id)

    def _count_after_sql(self, engine: Engine, watermark_at: datetime, watermark_event_id: str) -> int:
        # Match poll cursor (occurred_at, event_id), including empty event_id watermark.
        where, _order, params = self._after_watermark(engine, watermark_at, watermark_event_id)
        sql = text(f"SELECT COUNT(*) AS n FROM {self._from_clause(engine)} WHERE {where}")
        with _utc_connection(engine) as conn:
            return int(conn.execute(sql, params).scalar() or 0)

    def _count_after_rows(
        self, engine: Engine, watermark_at: datetime, watermark_event_id: str
    ) -> int | None:
        # Same synthetic-id path as poll. Cap the scan; None = unknown / too large.
        cursor = cursor_tuple(watermark_at, watermark_event_id)
        rows = self._fetch_rows(engine, watermark_at, watermark_event_id, limit=_LAG_SCAN_LIMIT)
        keys = _parsed_rows(rows, cursor_key_from_payload, lambda _payload, _exc: None)
        count = sum(1 for key in keys if key > cursor)
        if len(rows) >= _LAG_SCAN_LIMIT:
            return None
        return count

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
            watermark_event_id = str(raw.get("event_id") or "")
        except (OSError, ValueError):
            logger.exception(
                "Ignoring corrupt watermark file %s; keeping watermark %s",
                self._watermark_path,
                self._watermark_at.isoformat(),
            )
            return
        self._watermark_at = watermark_at
        self._watermark_event_id = watermark_event_id

    def _persist_watermark_unlocked(self) -> None:
        if self._watermark_path is None:
            return
        self._watermark_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "occurred_at": self._watermark_at.isoformat(),
            "event_id": self._watermark_event_id,
        }
        tmp = self._watermark_path.with_name(f".{self._watermark_path.name}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            dir_fd = os.open(str(tmp.parent), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        tmp.replace(self._watermark_path)
