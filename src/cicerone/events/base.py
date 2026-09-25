"""EventSource protocol, normalized event types, and shared consumer lifecycle."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass(frozen=True)
class NormalizedEvent:
    user_id: str
    item_id: str
    event_type: str
    quantity: int
    occurred_at: datetime
    event_id: str
    generated_event_id: bool = False


@dataclass(frozen=True)
class EventSourceHealth:
    connected: bool
    lag: int | None = None
    last_event_at: datetime | None = None
    detail: str | None = None


class EventBackpressureError(Exception):
    """Source queue is full; caller should retry later (HTTP 429)."""


class EventSourceError(RuntimeError):
    """Source is disconnected or a named backend call failed."""


class EventSource(Protocol):
    def connect(self) -> None:
        """Establish connections / start accepting work."""
        ...

    def poll(self, max_events: int = 100) -> Sequence[NormalizedEvent]:
        """Return up to ``max_events`` pending events (may be empty)."""
        ...

    def ack(self, event_ids: Sequence[str]) -> Sequence[str]:
        """Ack ids that still have a live delivery; return those ids."""
        ...

    def nack(self, events: Sequence[NormalizedEvent]) -> Sequence[NormalizedEvent]:
        """Requeue events; return those the source did not keep."""
        ...

    def health(self) -> EventSourceHealth:
        """Lag / connectivity for dashboard and metrics."""
        ...


class QueuedEventSource:
    """Pending/in-flight bookkeeping for persistent EventSource consumers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connected = False
        self._pending: deque[NormalizedEvent] = deque()
        self._pending_ids: set[str] = set()
        self._in_flight: set[str] = set()
        self._last_event_at: datetime | None = None

    def poll(self, max_events: int = 100) -> Sequence[NormalizedEvent]:
        if max_events < 1:
            return []
        backend = self._require_ready()
        out = self._drain_pending(max_events)
        remaining = max_events - len(out)
        if remaining > 0:
            out.extend(self._fetch_events(backend, remaining))
        self._record_last_event(out)
        return out

    def nack(self, events: Sequence[NormalizedEvent]) -> Sequence[NormalizedEvent]:
        if not events:
            return ()
        kept: set[int] = set()
        with self._lock:
            for event in reversed(list(events)):
                if self._delivery_handle(event.event_id) is None:
                    continue
                self._in_flight.discard(event.event_id)
                kept.add(id(event))
                if event.event_id in self._pending_ids:
                    continue
                self._pending.appendleft(event)
                self._pending_ids.add(event.event_id)
        return tuple(event for event in events if id(event) not in kept)

    def ack(self, event_ids: Sequence[str]) -> Sequence[str]:
        if not event_ids:
            return ()
        backend = self._require_ready()
        resolved = self._resolve_deliveries(event_ids)
        if not resolved:
            return ()
        return self._commit_acks(backend, resolved)

    def _require_ready(self) -> Any:
        with self._lock:
            backend = self._backend()
            if not self._connected or backend is None:
                raise EventSourceError(f"{type(self).__name__} is not connected")
            return backend

    def _backend(self) -> Any:
        raise NotImplementedError

    def _delivery_handle(self, event_id: str) -> Any | None:
        raise NotImplementedError

    def _fetch_events(self, backend: Any, max_events: int) -> Sequence[NormalizedEvent]:
        raise NotImplementedError

    def _commit_acks(self, backend: Any, resolved: Sequence[tuple[str, Any]]) -> Sequence[str]:
        raise NotImplementedError

    def _drain_pending(self, max_events: int) -> list[NormalizedEvent]:
        out: list[NormalizedEvent] = []
        with self._lock:
            while self._pending and len(out) < max_events:
                event = self._pending.popleft()
                self._pending_ids.discard(event.event_id)
                self._in_flight.add(event.event_id)
                out.append(event)
        return out

    def _record_last_event(self, events: Sequence[NormalizedEvent]) -> None:
        if not events:
            return
        newest = max(event.occurred_at for event in events)
        with self._lock:
            self._last_event_at = newest

    def _clear_lifecycle(self) -> None:
        self._pending.clear()
        self._pending_ids.clear()
        self._in_flight.clear()

    def _forget_ids_unlocked(self, event_id: str) -> None:
        self._in_flight.discard(event_id)
        self._pending_ids.discard(event_id)

    def _local_held_unlocked(self) -> int:
        return len(self._pending_ids) + len(self._in_flight)

    def _resolve_deliveries(self, event_ids: Sequence[str]) -> list[tuple[str, Any]]:
        resolved: list[tuple[str, Any]] = []
        with self._lock:
            for event_id in event_ids:
                eid = str(event_id)
                handle = self._delivery_handle(eid)
                if handle is not None:
                    resolved.append((eid, handle))
        return resolved
