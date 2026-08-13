"""In-memory webhook EventSource (HTTP push → poll/ack)."""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from cicerone.events.base import EventSourceHealth, NormalizedEvent
from cicerone.events.normalize import normalize_event, normalize_events


class WebhookEventSource:
    """Push sink for ``POST /events``; drained via ``poll`` / ``ack``."""

    def __init__(self, options: dict[str, Any] | None = None):
        del options
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
            queued: list[NormalizedEvent] = []
            for event in events:
                if event.event_id in known:
                    continue
                known.add(event.event_id)
                self._pending.append(event)
                self._last_event_at = event.occurred_at
                queued.append(event)
        return queued

    def poll(self, max_events: int = 100) -> Sequence[NormalizedEvent]:
        if max_events < 1:
            return []
        out: list[NormalizedEvent] = []
        with self._lock:
            while self._pending and len(out) < max_events:
                event = self._pending.popleft()
                # Keep first in-flight row for a given id (retries already skipped at ingest).
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
