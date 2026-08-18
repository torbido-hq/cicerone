"""Micro-batch buffer: flush by count or time window."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from cicerone.events.base import NormalizedEvent
from cicerone.events.normalize import event_fingerprint


@dataclass(frozen=True)
class BufferExtendResult:
    """Outcome of ``MicroBatchBuffer.extend`` for source ack/nack bookkeeping."""

    kept: tuple[NormalizedEvent, ...]
    duplicates: tuple[NormalizedEvent, ...]
    overflow: tuple[NormalizedEvent, ...]

    @property
    def kept_count(self) -> int:
        return len(self.kept)


class MicroBatchBuffer:
    def __init__(
        self,
        *,
        batch_size: int,
        batch_window_seconds: float,
        dedupe: bool = True,
        max_events: int | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if batch_window_seconds <= 0:
            raise ValueError("batch_window_seconds must be > 0")
        # Cap in-flight buffer (+ dedupe sets) so long-running workers cannot grow without bound.
        cap = batch_size if max_events is None else max_events
        if cap < batch_size:
            raise ValueError("max_events must be >= batch_size")
        self._batch_size = batch_size
        self._max_events = cap
        self._batch_window_seconds = batch_window_seconds
        self._dedupe = dedupe
        self._events: list[NormalizedEvent] = []
        self._event_ids: set[str] = set()
        self._fingerprints: set[str] = set()
        self._window_started_at: float | None = None

    def __len__(self) -> int:
        return len(self._events)

    @property
    def remaining_capacity(self) -> int:
        return max(0, self._max_events - len(self._events))

    def extend(self, events: Sequence[NormalizedEvent]) -> BufferExtendResult:
        """Append events; classify kept vs duplicate vs capacity overflow for ack/nack."""
        kept: list[NormalizedEvent] = []
        duplicates: list[NormalizedEvent] = []
        overflow: list[NormalizedEvent] = []
        now = time.monotonic()
        for event in events:
            if len(self._events) >= self._max_events:
                overflow.append(event)
                continue
            if self._dedupe:
                if event.event_id in self._event_ids:
                    duplicates.append(event)
                    continue
                fingerprint = event_fingerprint(event)
                if fingerprint in self._fingerprints:
                    duplicates.append(event)
                    continue
                self._event_ids.add(event.event_id)
                self._fingerprints.add(fingerprint)
            if self._window_started_at is None:
                self._window_started_at = now
            self._events.append(event)
            kept.append(event)
        return BufferExtendResult(
            kept=tuple(kept),
            duplicates=tuple(duplicates),
            overflow=tuple(overflow),
        )

    def ready(self, *, now: float | None = None) -> bool:
        if not self._events:
            return False
        if len(self._events) >= self._batch_size:
            return True
        started = self._window_started_at
        if started is None:
            return False
        clock = time.monotonic() if now is None else now
        return (clock - started) >= self._batch_window_seconds

    def flush_if_ready(self, *, now: float | None = None) -> list[NormalizedEvent]:
        if not self.ready(now=now):
            return []
        return self.flush()

    def flush(self) -> list[NormalizedEvent]:
        events = self._events
        self._events = []
        self._event_ids.clear()
        self._fingerprints.clear()
        self._window_started_at = None
        return events
