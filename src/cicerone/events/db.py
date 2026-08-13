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
from cicerone.events.normalize import EventNormalizeError, normalize_event
from cicerone.io.db_store import DEFAULT_EVENTS_TABLE
from cicerone.io.options import require_option, sql_identifier

logger = logging.getLogger(__name__)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _as_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        ts = pd.to_datetime(value, utc=True)
        if pd.isna(ts):
            raise EventNormalizeError("occurred_at is invalid")
        dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


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
        self._watermark_at = _as_utc(initial) if initial not in (None, "") else _EPOCH
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

    def poll(self, max_events: int = 100) -> Sequence[NormalizedEvent]:
        if max_events < 1:
            return []
        with self._lock:
            if self._engine is None:
                raise RuntimeError("DbEventSource.connect() required before poll")
            watermark_at = self._watermark_at
            in_flight_ids = set(self._in_flight)
            engine = self._engine
        rows = self._fetch_rows(engine, watermark_at, limit=max_events + len(in_flight_ids) + 8)
        out: list[NormalizedEvent] = []
        with self._lock:
            for payload in rows:
                if len(out) >= max_events:
                    break
                try:
                    occurred_at = _as_utc(payload.get("occurred_at"))
                    payload = {
                        **payload,
                        "occurred_at": occurred_at.isoformat(),
                        "event_id": _stable_event_id(payload, occurred_at),
                    }
                    event = normalize_event(payload)
                except EventNormalizeError:
                    logger.warning("Skipping invalid DB event row: %s", payload, exc_info=True)
                    continue
                key = _cursor_key(event)
                if key <= (self._watermark_at, self._watermark_event_id):
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
            lag = len(self._in_flight)
            detail = f"db watermark={self._watermark_at.isoformat()}"
            connected = self._connected
            last_event_at = self._last_event_at
            engine = self._engine
            watermark_at = self._watermark_at
        if engine is not None:
            try:
                lag += self._count_after(engine, watermark_at)
            except Exception:
                logger.exception("Failed to estimate DB event lag")
        return EventSourceHealth(
            connected=connected,
            lag=lag,
            last_event_at=last_event_at,
            detail=detail,
        )

    def _base_from_sql(self) -> str:
        if self._events_query:
            return f"({self._events_query}) AS cicerone_events_src"
        return f'"{self._events_table}"'

    def _fetch_rows(self, engine: Engine, watermark_at: datetime, *, limit: int) -> list[dict[str, Any]]:
        sql = text(
            f"SELECT * FROM {self._base_from_sql()} "
            "WHERE occurred_at >= :watermark "
            "ORDER BY occurred_at ASC "
            "LIMIT :limit"
        )
        frame = pd.read_sql_query(sql, engine, params={"watermark": watermark_at, "limit": max(limit, 1)})
        if frame.empty:
            return []
        return frame.to_dict(orient="records")

    def _count_after(self, engine: Engine, watermark_at: datetime) -> int:
        sql = text(f"SELECT COUNT(*) AS n FROM {self._base_from_sql()} WHERE occurred_at > :watermark")
        frame = pd.read_sql_query(sql, engine, params={"watermark": watermark_at})
        return int(frame.iloc[0]["n"]) if not frame.empty else 0

    def _load_watermark_unlocked(self) -> None:
        if self._watermark_path is None or not self._watermark_path.is_file():
            return
        raw = json.loads(self._watermark_path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"invalid watermark file {self._watermark_path}")
        occurred_at = raw.get("occurred_at")
        if occurred_at:
            self._watermark_at = _as_utc(occurred_at)
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
