"""Micro-batch buffer: flush by count or time window."""

from __future__ import annotations

import time
from collections.abc import Sequence

from cicerone.events.base import NormalizedEvent
from cicerone.events.normalize import event_fingerprint


class MicroBatchBuffer:
    def __init__(self, *, batch_size: int, batch_window_seconds: float, dedupe: bool = True):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if batch_window_seconds <= 0:
            raise ValueError("batch_window_seconds must be > 0")
        self._batch_size = batch_size
        self._batch_window_seconds = batch_window_seconds
        self._dedupe = dedupe
        self._events: list[NormalizedEvent] = []
        self._fingerprints: set[str] = set()
        self._window_started_at: float | None = None

    def __len__(self) -> int:
        return len(self._events)

    def extend(self, events: Sequence[NormalizedEvent]) -> int:
        """Append events; return how many were kept (after optional dedupe)."""
        kept = 0
        now = time.monotonic()
        for event in events:
            if self._dedupe:
                fingerprint = event_fingerprint(event)
                if fingerprint in self._fingerprints:
                    continue
                self._fingerprints.add(fingerprint)
            if self._window_started_at is None:
                self._window_started_at = now
            self._events.append(event)
            kept += 1
        return kept

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
        self._fingerprints.clear()
        self._window_started_at = None
        return events
