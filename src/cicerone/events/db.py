"""DB watermark EventSource (poll new interaction rows after a cursor)."""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from cicerone.events.base import EventSourceHealth, NormalizedEvent
from cicerone.events.normalize import EventNormalizeError, normalize_event, parse_occurred_at
from cicerone.io.db_store import DEFAULT_EVENTS_TABLE
from cicerone.io.options import require_option, sql_identifier

logger = logging.getLogger(__name__)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _db_occurred_at(value: Any) -> datetime:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise EventNormalizeError("occurred_at is invalid")
        value = value.to_pydatetime()
    # SQLAlchemy/SQLite often yield naive datetimes that are UTC in practice.
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return parse_occurred_at(value)


def _stable_event_id(payload: dict[str, Any], occurred_at: datetime) -> str:
    existing = payload.get("event_id") or payload.get("idempotency_key")
    if existing not in (None, ""):
        return str(existing)
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


def _cursor_key(event: NormalizedEvent) -> tuple[datetime, str]:
    return (event.occurred_at, event.event_id)


def _row_to_event(payload: dict[str, Any]) -> NormalizedEvent:
    occurred_at = _db_occurred_at(payload.get("occurred_at"))
    return normalize_event(
        {
            **payload,
            "occurred_at": occurred_at.isoformat(),
            "event_id": _stable_event_id(payload, occurred_at),
        }
    )


class DbEventSource:
    """Poll ``events`` (or ``events_query``) after a watermark; advance on ``ack``."""

    def __init__(self, options: dict[str, Any] | None = None):
        options = dict(options or {})
        self._database_url = require_option(options, "database_url", "db")
        self._events_table = sql_identifier(
            options.get("events_table", DEFAULT_EVENTS_TABLE),
            option="events_table",
        )
        self._events_query = options.get("events_query")
        if self._events_query is not None and not isinstance(self._events_query, str):
            raise ValueError("events.options.events_query must be a string")
        self._watermark_path = Path(options["watermark_path"]) if options.get("watermark_path") else None
        initial = options.get("initial_watermark")
        self._watermark_at = _db_occurred_at(initial) if initial not in (None, "") else _EPOCH
        self._watermark_event_id = str(options.get("initial_watermark_event_id") or "")
        self._engine: Engine | None = None
        self._lock = threading.Lock()
        self._in_flight: OrderedDict[str, NormalizedEvent] = OrderedDict()
        self._connected = False
        self._last_event_at: datetime | None = None

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
        if engine is not None:
            engine.dispose()

    def poll(self, max_events: int = 100) -> Sequence[NormalizedEvent]:
        if max_events < 1:
            return []
        with self._lock:
            if self._engine is None:
                raise RuntimeError("DbEventSource.connect() required before poll")
            watermark_at = self._watermark_at
            in_flight_count = len(self._in_flight)
            engine = self._engine
        rows = self._fetch_rows(engine, watermark_at, limit=max_events + in_flight_count + 8)
        candidates = sorted((_row_to_event(payload) for payload in rows), key=_cursor_key)

        out: list[NormalizedEvent] = []
        with self._lock:
            cursor = (self._watermark_at, self._watermark_event_id)
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
                if key > (self._watermark_at, self._watermark_event_id):
                    self._watermark_at, self._watermark_event_id = key
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
            return f"({self._events_query}) AS cicerone_events_src"
        return f'"{self._events_table}"'

    def _fetch_rows(self, engine: Engine, watermark_at: datetime, *, limit: int) -> list[dict[str, Any]]:
        sql = text(
            f"SELECT * FROM {self._from_clause()} "
            "WHERE occurred_at >= :watermark "
            "ORDER BY occurred_at ASC "
            "LIMIT :limit"
        )
        frame = pd.read_sql_query(sql, engine, params={"watermark": watermark_at, "limit": max(limit, 1)})
        if frame.empty:
            return []
        return frame.to_dict(orient="records")

    def _count_after(self, engine: Engine, watermark_at: datetime, watermark_event_id: str) -> int:
        # Match poll's (occurred_at, event_id) cursor; SQL timestamp equality is backend-fragile.
        rows = self._fetch_rows(engine, watermark_at, limit=10_000)
        cursor = (watermark_at, watermark_event_id)
        return sum(1 for payload in rows if _cursor_key(_row_to_event(payload)) > cursor)

    def _load_watermark_unlocked(self) -> None:
        if self._watermark_path is None or not self._watermark_path.is_file():
            return
        raw = json.loads(self._watermark_path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"invalid watermark file {self._watermark_path}")
        occurred_at = raw.get("occurred_at")
        if occurred_at:
            self._watermark_at = _db_occurred_at(occurred_at)
        self._watermark_event_id = str(raw.get("event_id") or "")

    def _persist_watermark_unlocked(self) -> None:
        if self._watermark_path is None:
            return
        self._watermark_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "occurred_at": self._watermark_at.isoformat(),
            "event_id": self._watermark_event_id,
        }
        tmp = self._watermark_path.with_name(f".{self._watermark_path.name}.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
        tmp.replace(self._watermark_path)
