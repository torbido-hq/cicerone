"""Background worker: poll EventSource → micro-batch → incremental updater."""

from __future__ import annotations

import logging
import threading

from cicerone.events.base import EventSource
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.updater import IncrementalUpdater

logger = logging.getLogger(__name__)


class EventWorker:
    def __init__(
        self,
        source: EventSource,
        buffer: MicroBatchBuffer,
        updater: IncrementalUpdater,
        *,
        poll_interval_seconds: float = 1.0,
        poll_max_events: int = 100,
    ):
        self._source = source
        self._buffer = buffer
        self._updater = updater
        self._poll_interval_seconds = poll_interval_seconds
        self._poll_max_events = poll_max_events
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._source.connect()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="cicerone-events", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("Event worker tick failed")
            self._stop.wait(self._poll_interval_seconds)

    def tick(self) -> int:
        """One poll/flush cycle; returns events successfully applied."""
        polled = list(self._source.poll(self._poll_max_events))
        if polled:
            self._buffer.extend(polled)
        ready = self._buffer.flush_if_ready()
        if not ready:
            return 0
        applied = self._updater.apply(ready)
        if applied:
            self._source.ack([event.event_id for event in ready])
            return applied
        # Un-acked in-flight stays until a later successful apply.
        self._buffer.extend(ready)
        return 0
