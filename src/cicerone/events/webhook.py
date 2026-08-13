"""In-memory webhook EventSource (HTTP push → poll/ack)."""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from cicerone.events.base import EventSourceHealth, NormalizedEvent
from cicerone.events.normalize import normalize_event, normalize_events
from cicerone.events.registry import register_event_source


class WebhookEventSource:
    """Push sink for ``POST /events``; drained via ``poll`` / ``ack``."""

    def __init__(self, options: dict[str, Any] | None = None):
        del options  # reserved for auth_token etc. (handled by serve route)
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
            for event in events:
                self._pending.append(event)
                self._last_event_at = event.occurred_at
        return events

    def poll(self, max_events: int = 100) -> Sequence[NormalizedEvent]:
        if max_events < 1:
            return []
        out: list[NormalizedEvent] = []
        with self._lock:
            while self._pending and len(out) < max_events:
                event = self._pending.popleft()
                self._in_flight[event.event_id] = event
                out.append(event)
        return out

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


def _build_webhook(options: dict[str, Any]) -> WebhookEventSource:
    return WebhookEventSource(options)


register_event_source("webhook", _build_webhook)
