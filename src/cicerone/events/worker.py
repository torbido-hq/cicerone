"""Background worker: poll EventSource → micro-batch → incremental updater."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any

from cicerone.config.constants import (
    DEFAULT_EVENTS_HEARTBEAT_SECONDS,
    DEFAULT_EVENTS_POLL_INTERVAL_SECONDS,
)
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

_ONLINE_PERSIST_ATTEMPTS = 3


def _call_heartbeat(beat: Callable[..., Any], events: Sequence[NormalizedEvent]) -> None:
    try:
        beat(events)
    except Exception:
        logger.exception("Event source heartbeat failed")


@contextmanager
def inflight_heartbeat(
    source: EventSource,
    events: Sequence[NormalizedEvent],
    interval_seconds: float,
):
    """Beat at start of apply and again every ``interval_seconds`` until exit."""
    beat = getattr(source, "heartbeat", None)
    if not callable(beat):
        yield
        return
    _call_heartbeat(beat, events)
    if interval_seconds <= 0:
        yield
        return
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(interval_seconds):
            _call_heartbeat(beat, events)

    thread = threading.Thread(target=_loop, name="cicerone-events-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=max(1.0, interval_seconds))


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
        heartbeat_interval_seconds: float = DEFAULT_EVENTS_HEARTBEAT_SECONDS,
    ):
        self._source = source
        self._buffer = buffer
        self._updater = updater
        self._poll_interval_seconds = poll_interval_seconds
        self._poll_max_events = poll_max_events
        self._apply_lock = apply_lock
        self._poll_without_lock = poll_without_lock
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
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
        if self._apply_lock is None:
            self._poll_into_buffer()
            ready = self._buffer.flush_if_ready()
            return self._flush_ready(ready) if ready else 0

        if self._poll_without_lock:
            self._poll_into_buffer()
            ready = self._buffer.flush_if_ready()
            if not ready:
                return 0
            return self._with_apply_lock(ready)

        if not self._acquire_apply_lock():
            return 0
        try:
            self._poll_into_buffer()
            ready = self._buffer.flush_if_ready()
            return self._flush_ready(ready) if ready else 0
        finally:
            self._release_apply_lock()

    def _poll_into_buffer(self) -> None:
        room = self._buffer.remaining_capacity
        if room <= 0:
            return
        # May receive more than ``room``; overflow is nacked for later redelivery.
        polled = list(self._source.poll(self._poll_max_events))
        if not polled:
            return
        result = self._buffer.extend(polled)
        # Duplicates are already represented in the buffer — ack so sources
        # do not leave them stuck in-flight / PEL.
        if result.duplicates:
            self._source.ack([event.event_id for event in result.duplicates])
        # Capacity rejects must be redelivered when there is room again.
        if result.overflow:
            self._source.nack(result.overflow)

    def _acquire_apply_lock(self) -> bool:
        lock = self._apply_lock
        if lock is None:
            return True
        if not lock.acquire():
            record_events_lock(status="skip")
            update_events_leader(False)
            return False
        record_events_lock(status="acquired")
        update_events_leader(lock.owned())
        return True

    def _release_apply_lock(self) -> None:
        lock = self._apply_lock
        if lock is None:
            return
        lock.release()
        update_events_leader(False)

    def _with_apply_lock(self, ready: list[NormalizedEvent]) -> int:
        if not self._acquire_apply_lock():
            record_events_flush(status="busy")
            record_events_apply_busy(reason="lock")
            self._source.nack(ready)
            return 0
        try:
            return self._flush_ready(ready)
        finally:
            self._release_apply_lock()

    def _drain_buffer_on_stop(self) -> None:
        leftover = self._buffer.flush()
        if not leftover:
            return
        logger.info("Draining %d buffered event(s) on worker stop", len(leftover))
        if not self._acquire_apply_lock():
            logger.info("Stop drain skipped: apply lease held by another replica")
            self._source.nack(leftover)
            return
        try:
            self._flush_ready(leftover)
        finally:
            self._release_apply_lock()

    def _flush_ready(self, ready: list[NormalizedEvent]) -> int:
        try:
            with inflight_heartbeat(self._source, ready, self._heartbeat_interval_seconds):
                applied = self._updater.apply(ready, persist_online=False)
        except LockLostError:
            record_events_flush(status="error")
            update_events_leader(False)
            logger.error(
                "Apply lease lost before write; nacking %d event(s)",
                len(ready),
            )
            self._updater.abort_online()
            self._source.nack(ready)
            return 0
        except Exception:
            record_events_flush(status="error")
            logger.exception("Incremental apply failed; returning %d event(s) to source", len(ready))
            self._updater.abort_online()
            self._source.nack(ready)
            return 0
        if applied == 0:
            record_events_flush(status="busy")
            record_events_apply_busy(reason="retrain")
            self._updater.abort_online()
            self._source.nack(ready)
            return 0
        if applied != len(ready):
            record_events_flush(status="error")
            logger.error(
                "Incremental apply returned %d for %d ready event(s); nacking batch",
                applied,
                len(ready),
            )
            self._updater.abort_online()
            self._source.nack(ready)
            return 0
        try:
            self._source.ack([event.event_id for event in ready])
        except Exception:
            record_events_flush(status="error")
            logger.exception("Event source ack failed after successful apply; nacking batch")
            self._updater.abort_online()
            self._source.nack(ready)
            raise
        self._persist_online_after_ack()
        record_events_flush(status="success", events=applied)
        return applied

    def _persist_online_after_ack(self) -> None:
        last_error: BaseException | None = None
        for attempt in range(1, _ONLINE_PERSIST_ATTEMPTS + 1):
            try:
                self._updater.persist_online()
                return
            except LockLostError:
                logger.error("Apply lease lost before online persist; dropping pending artifact")
                self._updater.abort_online()
                return
            except Exception as exc:
                last_error = exc
                logger.exception(
                    "Online artifact persist failed after ack (attempt %d/%d)",
                    attempt,
                    _ONLINE_PERSIST_ATTEMPTS,
                )
        self._updater.abort_online()
        if last_error is not None:
            logger.error("Online artifact persist gave up after ack; pending fit dropped")
