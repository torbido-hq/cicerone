"""Background worker: poll EventSource → micro-batch → incremental updater."""

from __future__ import annotations

import logging
import threading

from cicerone.config.constants import DEFAULT_EVENTS_POLL_INTERVAL_SECONDS
from cicerone.events.base import EventSource, NormalizedEvent
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.updater import IncrementalUpdater
from cicerone.locks import LockBackend, LockLostError
from cicerone.serve.metrics import (
    record_events_apply_busy,
    record_events_flush,
    record_events_lock,
    record_events_tick_error,
    update_events_leader,
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
        apply_lock: LockBackend | None = None,
        poll_without_lock: bool = False,
    ):
        self._source = source
        self._buffer = buffer
        self._updater = updater
        self._poll_interval_seconds = poll_interval_seconds
        self._poll_max_events = poll_max_events
        self._apply_lock = apply_lock
        self._poll_without_lock = poll_without_lock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._source.connect()
        self.refresh_source_health_metrics()
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
        try:
            self._drain_buffer_on_stop()
        except Exception:
            logger.exception("Event worker drain on stop failed")
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
            finally:
                self.refresh_source_health_metrics()
            self._stop.wait(self._poll_interval_seconds)

    def tick(self) -> int:
        """One poll/flush cycle; returns events successfully applied."""
        got_lock = True
        if self._apply_lock is not None:
            got_lock = self._apply_lock.acquire()
            if got_lock:
                record_events_lock(status="acquired")
            else:
                record_events_lock(status="skip")
                if not self._poll_without_lock:
                    return 0
        else:
            update_events_leader(True)

        try:
            return self._tick_with_lease(got_lock)
        finally:
            if self._apply_lock is not None and got_lock:
                self._apply_lock.release()

    def _tick_with_lease(self, got_lock: bool) -> int:
        if got_lock or self._poll_without_lock:
            room = self._buffer.remaining_capacity
            if room > 0:
                # May receive more than ``room``; overflow is nacked for later redelivery.
                polled = list(self._source.poll(self._poll_max_events))
                if polled:
                    result = self._buffer.extend(polled)
                    # Duplicates are already represented in the buffer — ack so sources
                    # do not leave them stuck in-flight / PEL.
                    if result.duplicates:
                        self._source.ack([event.event_id for event in result.duplicates])
                    # Capacity rejects must be redelivered when there is room again.
                    if result.overflow:
                        self._source.nack(result.overflow)
        ready = self._buffer.flush_if_ready()
        if not ready:
            return 0
        if self._apply_lock is not None and not got_lock:
            record_events_flush(status="busy")
            record_events_apply_busy(reason="lock")
            self._source.nack(ready)
            return 0
        return self._flush_ready(ready)

    def _drain_buffer_on_stop(self) -> None:
        leftover = self._buffer.flush()
        if not leftover:
            return
        logger.info("Draining %d buffered event(s) on worker stop", len(leftover))
        if self._apply_lock is not None and not self._apply_lock.acquire():
            logger.info("Stop drain skipped: apply lease held by another replica")
            self._source.nack(leftover)
            return
        try:
            applied = self._updater.apply(leftover)
        except Exception:
            logger.exception("Stop drain apply failed; nacking %d event(s)", len(leftover))
            self._source.nack(leftover)
            return
        finally:
            if self._apply_lock is not None:
                self._apply_lock.release()
        if applied == len(leftover):
            try:
                self._source.ack([event.event_id for event in leftover])
            except Exception:
                logger.exception("Stop drain ack failed; nacking %d event(s)", len(leftover))
                self._source.nack(leftover)
            return
        logger.warning(
            "Stop drain applied %d/%d event(s); nacking batch for redelivery",
            applied,
            len(leftover),
        )
        self._source.nack(leftover)

    def _flush_ready(self, ready: list[NormalizedEvent]) -> int:
        try:
            applied = self._updater.apply(ready)
        except LockLostError:
            record_events_flush(status="error")
            logger.error(
                "Apply lease lost before write; nacking %d event(s)",
                len(ready),
            )
            self._source.nack(ready)
            return 0
        except Exception:
            record_events_flush(status="error")
            logger.exception("Incremental apply failed; returning %d event(s) to source", len(ready))
            self._source.nack(ready)
            return 0
        if applied == 0:
            record_events_flush(status="busy")
            record_events_apply_busy(reason="retrain")
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
            return 0
        try:
            self._source.ack([event.event_id for event in ready])
        except Exception:
            record_events_flush(status="error")
            logger.exception("Event source ack failed after successful apply; nacking batch")
            self._source.nack(ready)
            raise
        record_events_flush(status="success", events=applied)
        return applied
