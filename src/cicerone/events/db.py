"""DB watermark EventSource (poll new interaction rows after a cursor)."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, text

from cicerone.config import ConfigError
from cicerone.events.base import EventSource, EventSourceHealth, NormalizedEvent
from cicerone.events.db_identity import (
    _SQLITE_IDENTITY_SORT,  # noqa: F401
    _cursor_key,
    _cursor_key_from_payload,
    _cursor_tuple,
    _db_occurred_at,
    _identity_bind_sort_key,
    _identity_sort_key,  # noqa: F401
    _identity_sql_sort_expr,
    _is_numeric_identity,  # noqa: F401
    _occurred_at_predicates,
    _row_identity,  # noqa: F401
    _row_to_event,
    _stable_event_id,  # noqa: F401
)
from cicerone.io.db_store import DEFAULT_EVENTS_TABLE
from cicerone.io.options import readonly_select, require_option, sql_identifier

logger = logging.getLogger(__name__)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_EVENTS_QUERY_ALIAS = "cicerone_events_src"
# Bound for health lag when SQL COUNT is unavailable; None means unknown/too large.
_LAG_SCAN_LIMIT = 10_000
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


class DbEventSource(EventSource):
    """Poll ``events`` (or trusted ``events_query``) after a watermark; advance on ``ack``.

    ``events_query`` is deploy-time config interpolated into SQL (read-only SELECT
    validated at construction); do not pass untrusted client input.
    """

    def __init__(self, options: dict[str, Any] | None = None):
        options = dict(options or {})
        self._database_url = require_option(options, "database_url", "db")
        self._events_table = sql_identifier(
            options.get("events_table", DEFAULT_EVENTS_TABLE),
            option="events_table",
        )
        self._events_query = options.get("events_query")
        if self._events_query is not None:
            if not isinstance(self._events_query, str):
                raise ValueError("events.options.events_query must be a string")
            self._events_query = _validate_events_query(self._events_query)
        self._watermark_path = Path(options["watermark_path"]) if options.get("watermark_path") else None
        initial = options.get("initial_watermark")
        self._watermark_at = _db_occurred_at(initial) if initial not in (None, "") else _EPOCH
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
        with self._lock:
            if self._engine is None:
                self._engine = create_engine(self._database_url, pool_pre_ping=True)
            self._load_watermark_unlocked()
            self._connected = True

    def close(self) -> None:
        with self._lock:
            engine = self._engine
            self._engine = None
            self._connected = False
            self._source_columns = None
            self._select_clause = None
            self._has_event_id_column = None
        if engine is not None:
            engine.dispose()

    def poll(self, max_events: int = 100) -> Sequence[NormalizedEvent]:
        if max_events < 1:
            return []
        with self._lock:
            if self._engine is None:
                raise RuntimeError("DbEventSource.connect() required before poll")
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
        candidates = sorted((_row_to_event(payload) for payload in rows), key=_cursor_key)

        out: list[NormalizedEvent] = []
        with self._lock:
            cursor = _cursor_tuple(self._watermark_at, self._watermark_event_id)
            for event in candidates:
                if len(out) >= max_events:
                    break
                if _cursor_key(event) <= cursor:
                    continue
                if event.event_id in self._in_flight:
                    continue
                self._in_flight[event.event_id] = event
                self._last_event_at = event.occurred_at
                out.append(event)
        return out

    def nack(self, events: Sequence[NormalizedEvent]) -> None:
        with self._lock:
            for event in events:
                self._in_flight.pop(event.event_id, None)

    def ack(self, event_ids: Sequence[str]) -> None:
        with self._lock:
            advanced = False
            for event_id in event_ids:
                event = self._in_flight.pop(str(event_id), None)
                if event is None:
                    continue
                key = _cursor_key(event)
                if key > _cursor_tuple(self._watermark_at, self._watermark_event_id):
                    self._watermark_at = event.occurred_at
                    self._watermark_event_id = event.event_id
                    advanced = True
            if advanced:
                self._persist_watermark_unlocked()

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
            except Exception:
                logger.exception("Failed to estimate DB event lag")
        return EventSourceHealth(
            connected=connected,
            lag=lag,
            last_event_at=last_event_at,
            detail=detail,
        )

    def _from_clause(self) -> str:
        if self._events_query:
            return f"({self._events_query}) AS {_EVENTS_QUERY_ALIAS}"
        return self._events_table

    def _ensure_source_schema(self, engine: Engine) -> tuple[str, bool]:
        with self._lock:
            if self._select_clause is not None and self._has_event_id_column is not None:
                return self._select_clause, self._has_event_id_column
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {self._from_clause()} LIMIT 0"))
            columns = frozenset(map(str, result.keys()))  # noqa: SIM118 — Result.keys(), not dict
        by_lower = {name.lower(): name for name in columns}
        if "occurred_at" not in by_lower:
            raise ConfigError(
                "events DB source must expose an occurred_at column for watermark polling "
                f"(from {self._from_clause()})"
            )
        selected = [by_lower[name] for name in _FETCH_COLUMNS if name in by_lower]
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
        select_clause = ", ".join(f'"{name}"' for name in selected)
        with self._lock:
            if self._select_clause is None or self._has_event_id_column is None:
                self._source_columns = columns
                self._select_clause = select_clause
                self._has_event_id_column = has_event_id
            cached_select = self._select_clause
            cached_has_event_id = self._has_event_id_column
        return cached_select, cached_has_event_id

    def _fetch_rows(
        self,
        engine: Engine,
        watermark_at: datetime,
        watermark_event_id: str,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        select_clause, has_event_id = self._ensure_source_schema(engine)
        dialect = engine.dialect.name
        identity_sort = _identity_sql_sort_expr(dialect) if has_event_id else None
        if identity_sort is not None:
            later, same, occurred_order = _occurred_at_predicates(dialect)
            sql = text(
                f"SELECT {select_clause} FROM {self._from_clause()} WHERE "
                f"{later} OR "
                f"({same} AND ({identity_sort}) > :watermark_sort) "
                f"ORDER BY {occurred_order}, ({identity_sort}) ASC "
                "LIMIT :limit"
            )
            params: dict[str, Any] = {
                "watermark_at": watermark_at,
                "watermark_sort": _identity_bind_sort_key(watermark_event_id),
                "limit": max(limit, 1),
            }
        elif has_event_id:
            # Match ack cursor (occurred_at, event_id) so same-timestamp pages cannot skip rows.
            sql = text(
                f"SELECT {select_clause} FROM {self._from_clause()} WHERE "
                "occurred_at > :watermark_at OR "
                "(occurred_at = :watermark_at AND event_id > :watermark_event_id) "
                "ORDER BY occurred_at ASC, event_id ASC "
                "LIMIT :limit"
            )
            params = {
                "watermark_at": watermark_at,
                "watermark_event_id": watermark_event_id,
                "limit": max(limit, 1),
            }
        else:
            sql = text(
                f"SELECT {select_clause} FROM {self._from_clause()} "
                "WHERE occurred_at >= :watermark "
                "ORDER BY occurred_at ASC "
                "LIMIT :limit"
            )
            params = {"watermark": watermark_at, "limit": max(limit, 1)}
        with engine.connect() as conn:
            result = conn.execute(sql, params)
            return [dict(row) for row in result.mappings()]

    def _count_after(self, engine: Engine, watermark_at: datetime, watermark_event_id: str) -> int | None:
        _select_clause, has_event_id = self._ensure_source_schema(engine)
        # Prefer SQL COUNT when event_id exists. SQLite text/datetime binding makes
        # equality unreliable, so scan rows there (still on the narrow projection).
        if has_event_id and engine.dialect.name != "sqlite":
            try:
                return self._count_after_sql(engine, watermark_at, watermark_event_id)
            except Exception:
                logger.debug("SQL lag COUNT unavailable; scanning event rows", exc_info=True)
        return self._count_after_rows(engine, watermark_at, watermark_event_id)

    def _count_after_sql(self, engine: Engine, watermark_at: datetime, watermark_event_id: str) -> int:
        # Match poll cursor (occurred_at, event_id), including empty event_id watermark.
        identity_sort = _identity_sql_sort_expr(engine.dialect.name)
        if identity_sort is not None:
            later, same, _occurred_order = _occurred_at_predicates(engine.dialect.name)
            sql = text(
                f"SELECT COUNT(*) AS n FROM {self._from_clause()} WHERE "
                f"{later} OR "
                f"({same} AND ({identity_sort}) > :watermark_sort)"
            )
            params: dict[str, Any] = {
                "watermark_at": watermark_at,
                "watermark_sort": _identity_bind_sort_key(watermark_event_id),
            }
        else:
            sql = text(
                f"SELECT COUNT(*) AS n FROM {self._from_clause()} WHERE "
                "occurred_at > :watermark_at OR "
                "(occurred_at = :watermark_at AND event_id > :watermark_event_id)"
            )
            params = {"watermark_at": watermark_at, "watermark_event_id": watermark_event_id}
        with engine.connect() as conn:
            return int(conn.execute(sql, params).scalar() or 0)

    def _count_after_rows(
        self, engine: Engine, watermark_at: datetime, watermark_event_id: str
    ) -> int | None:
        # Same synthetic-id path as poll. Cap the scan; None = unknown / too large.
        cursor = _cursor_tuple(watermark_at, watermark_event_id)
        rows = self._fetch_rows(engine, watermark_at, watermark_event_id, limit=_LAG_SCAN_LIMIT)
        count = sum(1 for payload in rows if _cursor_key_from_payload(payload) > cursor)
        if len(rows) >= _LAG_SCAN_LIMIT:
            return None
        return count

    def _load_watermark_unlocked(self) -> None:
        if self._watermark_path is None or not self._watermark_path.is_file():
            return
        try:
            raw = json.loads(self._watermark_path.read_text())
            if not isinstance(raw, dict):
                raise ValueError("watermark root must be an object")
            watermark_at = self._watermark_at
            occurred_at = raw.get("occurred_at")
            if occurred_at:
                watermark_at = _db_occurred_at(occurred_at)
            watermark_event_id = str(raw.get("event_id") or "")
        except Exception:
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
