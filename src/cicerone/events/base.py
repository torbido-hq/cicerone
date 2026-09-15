"""EventSource protocol and normalized event types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


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
