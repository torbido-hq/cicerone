"""In-memory webhook EventSource (HTTP push → poll/ack)."""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from cicerone.config.constants import DEFAULT_EVENTS_WEBHOOK_MAX_PENDING
from cicerone.events.base import EventBackpressureError, EventSourceHealth, NormalizedEvent
from cicerone.events.normalize import normalize_event, normalize_events


class WebhookEventSource:
    """Push sink for ``POST /events``; drained via ``poll`` / ``ack``."""

    def __init__(self, options: dict[str, Any] | None = None):
        options = dict(options or {})
        self._max_pending = int(options.get("max_pending", DEFAULT_EVENTS_WEBHOOK_MAX_PENDING))
        if self._max_pending < 1:
            raise ValueError("events.options.max_pending must be >= 1")
        self._lock = threading.Lock()
        self._pending: deque[NormalizedEvent] = deque()
        self._in_flight: OrderedDict[str, NormalizedEvent] = OrderedDict()
        self._connected = False
        self._last_event_at: datetime | None = None

    def connect(self) -> None:
        with self._lock:
            self._connected = True

    def ingest(self, payloads: Sequence[Any] | Mapping[str, Any] | Any) -> list[NormalizedEvent]:
        if isinstance(payloads, Mapping):
            events = [normalize_event(payloads)]
        elif isinstance(payloads, Sequence) and not isinstance(payloads, (str, bytes, bytearray)):
            events = normalize_events(list(payloads))
        else:
            events = [normalize_event(payloads)]
        with self._lock:
            self._connected = True
            known = {event.event_id for event in self._pending} | set(self._in_flight)
            novel: list[NormalizedEvent] = []
            for event in events:
                if event.event_id in known:
                    continue
                known.add(event.event_id)
                novel.append(event)
            backlog = len(self._pending) + len(self._in_flight)
            if novel and backlog + len(novel) > self._max_pending:
                raise EventBackpressureError(
                    f"event backlog full ({backlog}/{self._max_pending}); retry later"
                )
            for event in novel:
                self._pending.append(event)
                self._last_event_at = event.occurred_at
        return novel

    def poll(self, max_events: int = 100) -> Sequence[NormalizedEvent]:
        if max_events < 1:
            return []
        out: list[NormalizedEvent] = []
        with self._lock:
            while self._pending and len(out) < max_events:
                event = self._pending.popleft()
                if event.event_id not in self._in_flight:
                    self._in_flight[event.event_id] = event
                out.append(event)
        return out

    def nack(self, events: Sequence[NormalizedEvent]) -> None:
        """Return in-flight events to the pending queue (failed processing)."""
        with self._lock:
            for event in reversed(list(events)):
                self._in_flight.pop(event.event_id, None)
                self._pending.appendleft(event)

    def ack(self, event_ids: Sequence[str]) -> None:
        with self._lock:
            for event_id in event_ids:
                self._in_flight.pop(str(event_id), None)

    def health(self) -> EventSourceHealth:
        with self._lock:
            lag = len(self._pending) + len(self._in_flight)
            return EventSourceHealth(
                connected=self._connected,
                lag=lag,
                last_event_at=self._last_event_at,
                detail="webhook",
            )
