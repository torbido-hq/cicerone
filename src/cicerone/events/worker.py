"""Background worker: poll EventSource → micro-batch → incremental updater."""

from __future__ import annotations

import logging
import threading

from cicerone.config.constants import DEFAULT_EVENTS_POLL_INTERVAL_SECONDS
from cicerone.events.base import EventSource
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.updater import IncrementalUpdater
from cicerone.serve.metrics import (
    record_events_flush,
    record_events_tick_error,
    update_events_source_health,
)

logger = logging.getLogger(__name__)


class EventWorker:
    def __init__(
        self,
        source: EventSource,
        buffer: MicroBatchBuffer,
        updater: IncrementalUpdater,
        *,
        poll_interval_seconds: float = DEFAULT_EVENTS_POLL_INTERVAL_SECONDS,
        poll_max_events: int = 100,
    ):
        self._source = source
        self._buffer = buffer
        self._updater = updater
        self._poll_interval_seconds = poll_interval_seconds
        self._poll_max_events = poll_max_events
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def source(self) -> EventSource:
        return self._source

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._source.connect()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="cicerone-events", daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout_seconds: float = 5.0) -> bool:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout_seconds)
            if thread.is_alive():
                logger.warning(
                    "Event worker thread %s still alive after %.2fs join timeout",
                    thread.name,
                    join_timeout_seconds,
                )
                return False
        close = getattr(self._source, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Event source close() failed during worker stop")
        return True

    def refresh_source_health_metrics(self) -> None:
        try:
            health = self._source.health()
        except Exception:
            logger.exception("Failed to read event source health for metrics")
            update_events_source_health(connected=False, lag=None)
            return
        update_events_source_health(connected=health.connected, lag=health.lag)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                record_events_tick_error()
                logger.exception("Event worker tick failed")
            self._stop.wait(self._poll_interval_seconds)

    def tick(self) -> int:
        """One poll/flush cycle; returns events successfully applied."""
        room = self._buffer.remaining_capacity
        if room > 0:
            polled = list(self._source.poll(min(self._poll_max_events, room)))
            if polled:
                self._buffer.extend(polled)
        ready = self._buffer.flush_if_ready()
        if not ready:
            return 0
        try:
            applied = self._updater.apply(ready)
        except Exception:
            record_events_flush(status="error")
            logger.exception("Incremental apply failed; returning %d event(s) to source", len(ready))
            self._source.nack(ready)
            raise
        if applied == 0:
            # Busy: return to source so lag stays accurate and another tick can retry.
            record_events_flush(status="busy")
            self._source.nack(ready)
            return 0
        if applied != len(ready):
            record_events_flush(status="error")
            logger.error(
                "Incremental apply returned %d for %d ready event(s); nacking batch",
                applied,
                len(ready),
            )
            self._source.nack(ready)
            raise RuntimeError(
                f"IncrementalUpdater.apply returned {applied} for batch of {len(ready)}; "
                "partial apply is not supported"
            )
        record_events_flush(status="success", events=applied)
        self._source.ack([event.event_id for event in ready])
        return applied
