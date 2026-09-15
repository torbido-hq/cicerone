"""Background worker: poll EventSource → micro-batch → incremental updater."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any

from cicerone.config.constants import (
    DEFAULT_EVENTS_HEARTBEAT_SECONDS,
    DEFAULT_EVENTS_POLL_INTERVAL_SECONDS,
)
from cicerone.events.base import EventSource, NormalizedEvent
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.normalize import event_fingerprint
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
_APPLIED_EVENT_ID_CAP = 10_000


class HeartbeatError(RuntimeError):
    """Raised when an in-flight heartbeat fails (apply must not ack)."""


def _call_heartbeat(
    beat: Callable[..., Any],
    events: Sequence[NormalizedEvent],
    *,
    fail_closed: bool = False,
) -> None:
    try:
        beat(events)
    except Exception as exc:
        logger.exception("Event source heartbeat failed")
        if fail_closed:
            raise HeartbeatError("Event source heartbeat failed") from exc


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
    _call_heartbeat(beat, events, fail_closed=True)
    if interval_seconds <= 0:
        yield
        return
    stop = threading.Event()
    failed = threading.Event()

    def _loop() -> None:
        while not stop.wait(interval_seconds):
            try:
                beat(events)
            except Exception:
                logger.exception("Event source heartbeat failed")
                failed.set()
                return

    thread = threading.Thread(target=_loop, name="cicerone-events-heartbeat", daemon=True)
    thread.start()
    completed = False
    try:
        yield
        completed = True
    finally:
        stop.set()
        thread.join(timeout=max(1.0, interval_seconds))
    if completed and (failed.is_set() or thread.is_alive()):
        raise HeartbeatError("Event source heartbeat failed")


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
        self._source_guard = threading.Lock()
        self._tick_guard = threading.Lock()
        self._finalized = False
        self._source_unhealthy = False
        self._stop_epoch = 0
        self._starting = False
        self._thread: threading.Thread | None = None
        self._held: list[NormalizedEvent] = []
        self._deferred_acks: list[NormalizedEvent] = []
        self._retry_acks: list[NormalizedEvent] = []
        self._applied_event_ids: set[str] = set()
        self._applied_event_id_order: deque[str] = deque()
        self._applied_fingerprints: set[str] = set()
        self._applied_fingerprint_order: deque[str] = deque()
        self._ephemeral_event_ids = bool(getattr(source, "ephemeral_event_ids", False))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        epoch = self._stop_epoch
        connect_error: BaseException | None = None
        with self._source_guard:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._starting or self._stop_epoch != epoch:
                return
            self._starting = True
            self._stop.clear()
            self._finalized = False
            try:
                self._source.connect()
            except Exception as exc:
                connect_error = exc
        try:
            if self._start_aborted(epoch):
                self._drain_stopped_start()
                if connect_error is not None:
                    raise connect_error
                return
            if connect_error is not None:
                raise connect_error
            with self._tick_guard, self._source_guard:
                if self._start_aborted(epoch):
                    self._drain_and_close()
                    return
                self._source_unhealthy = not self.refresh_source_health_metrics()
                if self._start_aborted(epoch):
                    self._drain_and_close()
                    return
            with self._source_guard:
                if not self._start_aborted(epoch):
                    self._thread = threading.Thread(target=self._loop, name="cicerone-events", daemon=True)
                    self._thread.start()
                    if self._start_aborted(epoch):
                        return
                    return
            self._drain_stopped_start()
        finally:
            self._starting = False

    def _start_aborted(self, epoch: int) -> bool:
        return self._stop.is_set() or self._stop_epoch != epoch

    def _drain_stopped_start(self) -> None:
        with self._tick_guard, self._source_guard:
            self._drain_and_close()

    def stop(self, *, join_timeout_seconds: float = 5.0) -> bool:
        self._stop.set()
        self._stop_epoch += 1
        thread = self._thread
        was_alive = thread is not None and thread.is_alive()
        joined = True
        if thread is not None and was_alive:
            thread.join(timeout=join_timeout_seconds)
            if thread.is_alive():
                logger.warning(
                    "Event worker thread %s still alive after %.2fs join timeout",
                    thread.name,
                    join_timeout_seconds,
                )
                joined = False
        if joined and was_alive:
            acquired = self._tick_guard.acquire(blocking=True, timeout=join_timeout_seconds)
            if acquired:
                try:
                    if self._source_guard.acquire(blocking=False):
                        try:
                            if self._stop.is_set() and self._thread is thread:
                                self._drain_and_close()
                        finally:
                            self._source_guard.release()
                finally:
                    self._tick_guard.release()
            else:
                return False
        elif not joined:
            acquired = self._tick_guard.acquire(blocking=True, timeout=join_timeout_seconds)
            if acquired:
                try:
                    if self._source_guard.acquire(blocking=False):
                        try:
                            if self._stop.is_set() and self._thread is thread:
                                self._drain_and_close()
                        finally:
                            self._source_guard.release()
                finally:
                    self._tick_guard.release()
        elif self._tick_guard.acquire(blocking=False):
            try:
                if self._source_guard.acquire(blocking=False):
                    try:
                        if self._stop.is_set() and self._thread is thread:
                            self._drain_and_close()
                    finally:
                        self._source_guard.release()
                else:
                    return False
            finally:
                self._tick_guard.release()
        else:
            return False
        return joined

    def refresh_source_health_metrics(self) -> bool:
        try:
            health = self._source.health()
        except Exception:
            logger.exception("Failed to read event source health for metrics")
            update_events_source_health(connected=False, lag=None)
            return False
        update_events_source_health(connected=health.connected, lag=health.lag)
        return bool(health.connected)

    def _close_source(self) -> None:
        close = getattr(self._source, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            logger.exception("Event source close() failed during worker stop")

    def _drain_and_close(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        try:
            self._drain_buffer_on_stop()
        except Exception:
            logger.exception("Event worker drain on stop failed")
        self._close_source()

    def _restore_buffer(self, events: list[NormalizedEvent]) -> None:
        events = [event for event in events if not self._is_applied(event)]
        if not events:
            return
        result = self._buffer.extend(events)
        if result.overflow:
            self._held.extend(result.overflow)

    def _take_buffered(self) -> list[NormalizedEvent]:
        leftover = self._buffer.flush()
        leftover.extend(self._held)
        self._held = []
        return leftover

    def _take_pending_acks(self) -> list[NormalizedEvent]:
        pending = self._deferred_acks + self._retry_acks
        self._deferred_acks = []
        self._retry_acks = []
        return pending

    def _remember_token(self, token: str, seen: set[str], order: deque[str]) -> None:
        if token in seen:
            return
        seen.add(token)
        order.append(token)
        while len(order) > _APPLIED_EVENT_ID_CAP:
            seen.discard(order.popleft())

    def _remember_applied(self, events: Sequence[NormalizedEvent]) -> None:
        for event in events:
            self._remember_token(event.event_id, self._applied_event_ids, self._applied_event_id_order)
            if self._ephemeral_event_ids:
                self._remember_token(
                    event_fingerprint(event), self._applied_fingerprints, self._applied_fingerprint_order
                )

    def _is_applied(self, event: NormalizedEvent) -> bool:
        if event.event_id in self._applied_event_ids:
            return True
        return self._ephemeral_event_ids and event_fingerprint(event) in self._applied_fingerprints

    def _still_unapplied(self, event: NormalizedEvent) -> bool:
        if self._buffer.contains_event_id(event.event_id):
            return True
        fingerprint = event_fingerprint(event)
        if self._buffer.contains_fingerprint(fingerprint):
            return True
        return any(
            held.event_id == event.event_id or event_fingerprint(held) == fingerprint for held in self._held
        )

    def _ack_unbuffered(self, events: Sequence[NormalizedEvent]) -> None:
        to_ack: list[NormalizedEvent] = []
        for event in events:
            if self._still_unapplied(event):
                self._deferred_acks.append(event)
            else:
                to_ack.append(event)
        if not to_ack:
            return
        try:
            self._source.ack([event.event_id for event in to_ack])
        except Exception:
            self._retry_acks.extend(to_ack)
            raise

    def _ack_deferred_matching(self, applied: Sequence[NormalizedEvent]) -> None:
        if not self._deferred_acks:
            return
        applied_ids = {event.event_id for event in applied}
        applied_fps = {event_fingerprint(event) for event in applied}
        keep: list[NormalizedEvent] = []
        matched: list[NormalizedEvent] = []
        for event in self._deferred_acks:
            if (
                event.event_id in applied_ids or event_fingerprint(event) in applied_fps
            ) and not self._still_unapplied(event):
                matched.append(event)
            else:
                keep.append(event)
        self._deferred_acks = keep
        if not matched:
            return
        try:
            self._source.ack([event.event_id for event in matched])
        except Exception:
            self._retry_acks.extend(matched)
            raise

    def _flush_retry_acks(self) -> None:
        if not self._retry_acks:
            return
        batch = self._retry_acks
        self._retry_acks = []
        try:
            self._source.ack([event.event_id for event in batch])
            self._ack_deferred_matching(batch)
        except Exception:
            self._retry_acks.extend(batch)
            raise

    def _rejected_nacks(
        self, leftover: list[NormalizedEvent], result: Sequence[NormalizedEvent] | None
    ) -> list[NormalizedEvent]:
        if not result:
            return []
        rejected_ids = {id(event) for event in result}
        return [event for event in leftover if id(event) in rejected_ids]

    def _return_events(self, events: Sequence[NormalizedEvent]) -> None:
        leftover = list(events)
        if not leftover:
            return
        try:
            rejected = self._rejected_nacks(leftover, self._source.nack(leftover))
        except Exception:
            logger.exception(
                "Event worker failed to return %d event(s) to the source",
                len(leftover),
            )
            self._restore_buffer(leftover)
            return
        if rejected:
            self._restore_buffer(rejected)

    def _requeue_buffer_after_reconnect(self, leftover: list[NormalizedEvent]) -> None:
        self._return_events(leftover)

    def _reconnect_source(self) -> bool:
        with self._tick_guard:
            with self._source_guard:
                if self._stop.is_set():
                    self._drain_and_close()
                    return False
                leftover = self._take_buffered()
                leftover.extend(self._take_pending_acks())
                try:
                    self._source.connect()
                except Exception:
                    logger.exception("Event source reconnect failed")
                    self._restore_buffer(leftover)
                    if self._stop.is_set():
                        self._drain_and_close()
                    return False
                self._requeue_buffer_after_reconnect(leftover)
                if self._stop.is_set():
                    self._drain_and_close()
                    return False
            if self._stop.is_set():
                with self._source_guard:
                    self._drain_and_close()
                return False
            return True

    def _loop(self) -> None:
        disconnected = self._source_unhealthy
        current = threading.current_thread()
        try:
            while not self._stop.is_set():
                if disconnected and not self._reconnect_source():
                    if self._stop.is_set():
                        break
                    self._stop.wait(self._poll_interval_seconds)
                    continue
                try:
                    self.tick()
                except Exception:
                    record_events_tick_error()
                    logger.exception("Event worker tick failed")
                disconnected = self._source_unhealthy
                self._stop.wait(self._poll_interval_seconds)
        finally:
            if self._stop.is_set():
                with self._tick_guard, self._source_guard:
                    if self._thread is current:
                        self._drain_and_close()

    def tick(self) -> int:
        """One poll/flush cycle; returns events successfully applied."""
        with self._tick_guard:
            if self._stop.is_set():
                return 0
            try:
                return self._tick_locked()
            finally:
                self._source_unhealthy = not self.refresh_source_health_metrics()

    def _tick_locked(self) -> int:
        self._flush_retry_acks()
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
        if self._held:
            held_result = self._buffer.extend(self._held)
            self._held = list(held_result.overflow)
        room = self._buffer.remaining_capacity
        if room <= 0:
            return
        # May receive more than ``room``; overflow is nacked for later redelivery.
        polled = list(self._source.poll(self._poll_max_events))
        if not polled:
            return
        already = [event for event in polled if self._is_applied(event)]
        fresh = [event for event in polled if not self._is_applied(event)]
        result = self._buffer.extend(fresh) if fresh else None
        if result is not None and result.overflow:
            self._return_events(result.overflow)
        if already:
            self._ack_unbuffered(already)
        if result is not None and result.duplicates:
            self._ack_unbuffered(result.duplicates)

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
            self._return_events(ready)
            return 0
        try:
            return self._flush_ready(ready)
        finally:
            self._release_apply_lock()

    def _drain_buffer_on_stop(self) -> None:
        leftover = self._take_buffered()
        if leftover:
            logger.info("Draining %d buffered event(s) on worker stop", len(leftover))
            if not self._acquire_apply_lock():
                logger.info("Stop drain skipped: apply lease held by another replica")
                leftover.extend(self._take_pending_acks())
                self._return_events(leftover)
                return
            try:
                self._flush_ready(leftover)
            finally:
                self._release_apply_lock()
        pending = self._take_pending_acks()
        if pending:
            self._return_events(pending)

    def _flush_ready(self, ready: list[NormalizedEvent]) -> int:
        try:
            with inflight_heartbeat(self._source, ready, self._heartbeat_interval_seconds):
                applied = self._updater.apply(ready, persist_online=False)
        except HeartbeatError:
            record_events_flush(status="error")
            logger.error("In-flight heartbeat failed; returning %d event(s) to source", len(ready))
            self._updater.abort_online()
            self._return_events(ready)
            return 0
        except LockLostError:
            record_events_flush(status="error")
            update_events_leader(False)
            logger.error(
                "Apply lease lost before write; nacking %d event(s)",
                len(ready),
            )
            self._updater.abort_online()
            self._return_events(ready)
            return 0
        except Exception:
            record_events_flush(status="error")
            logger.exception("Incremental apply failed; returning %d event(s) to source", len(ready))
            self._updater.abort_online()
            self._return_events(ready)
            return 0
        if applied == 0:
            record_events_flush(status="busy")
            record_events_apply_busy(reason="retrain")
            self._updater.abort_online()
            self._return_events(ready)
            return 0
        if applied != len(ready):
            record_events_flush(status="error")
            logger.error(
                "Incremental apply returned %d for %d ready event(s); nacking batch",
                applied,
                len(ready),
            )
            self._updater.abort_online()
            self._return_events(ready)
            return 0
        self._remember_applied(ready)
        try:
            self._source.ack([event.event_id for event in ready])
        except TimeoutError:
            record_events_flush(status="error")
            logger.exception("Event source ack timed out after successful apply; persisting without nack")
            self._retry_acks.extend(ready)
            self._persist_online_after_ack()
            return applied
        except Exception:
            record_events_flush(status="error")
            logger.exception("Event source ack failed after successful apply; persisting without nack")
            self._retry_acks.extend(ready)
            self._persist_online_after_ack()
            raise
        try:
            self._ack_deferred_matching(ready)
        except Exception:
            record_events_flush(status="error")
            logger.exception(
                "Event source deferred ack failed after successful apply; persisting without nack"
            )
            self._persist_online_after_ack()
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
