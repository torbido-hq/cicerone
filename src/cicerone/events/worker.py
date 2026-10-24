"""Background worker: poll EventSource → micro-batch → incremental updater."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Sequence

from cicerone.config.constants import (
    DEFAULT_EVENTS_HEARTBEAT_SECONDS,
    DEFAULT_EVENTS_POLL_INTERVAL_SECONDS,
)
from cicerone.events.base import ClosableSource, EventSource, NormalizedEvent, has_ephemeral_event_ids
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.errors import (
    EVENT_APPLY_ERRORS,
    EVENT_SOURCE_ERRORS,
    EVENT_WORKER_ERRORS,
)
from cicerone.events.normalize import event_fingerprint
from cicerone.events.updater import IncrementalUpdater, event_replay_key
from cicerone.events.worker_heartbeat import HeartbeatError, inflight_heartbeat
from cicerone.locks import LockBackend, LockLostError, WriterLockBusyError
from cicerone.metrics import (
    record_events_apply_busy,
    record_events_flush,
    record_events_lock,
    record_events_replayed,
    record_events_tick_error,
    update_events_leader,
    update_events_source_health,
)

logger = logging.getLogger(__name__)

_ONLINE_PERSIST_ATTEMPTS = 3
# Failures after a committed write that are acked and logged; anything else is re-raised after the ack.
_POST_WRITE_ERRORS: tuple[type[BaseException], ...] = (
    *EVENT_APPLY_ERRORS,
    LockLostError,
    WriterLockBusyError,
    HeartbeatError,
)
_REPAIR_ERRORS: tuple[type[BaseException], ...] = (*EVENT_APPLY_ERRORS, WriterLockBusyError)
_APPLIED_EVENT_ID_CAP = 10_000


class _BoundedSeen:
    """Insertion-ordered set that forgets its oldest tokens past ``cap``."""

    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._seen: set[str] = set()
        self._order: deque[str] = deque()

    def __contains__(self, token: object) -> bool:
        return token in self._seen

    def add(self, token: str) -> None:
        if token in self._seen:
            return
        self._seen.add(token)
        self._order.append(token)
        while len(self._order) > self._cap:
            self._seen.discard(self._order.popleft())


__all__ = ["EventWorker", "HeartbeatError", "inflight_heartbeat"]


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
        if has_ephemeral_event_ids(source) and not buffer.fingerprint_dedupe:
            raise ValueError(
                "source has ephemeral event ids; build its MicroBatchBuffer with fingerprint_dedupe=True"
            )
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
        self._applied_event_ids = _BoundedSeen(_APPLIED_EVENT_ID_CAP)
        self._applied_fingerprints = _BoundedSeen(_APPLIED_EVENT_ID_CAP)
        self._ephemeral_event_ids = has_ephemeral_event_ids(source)
        updater.use_replay_key(self._replay_key)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        epoch = self._stop_epoch
        connect_error: BaseException | None = None
        starting = False
        try:
            with self._source_guard:
                if self._thread is not None and self._thread.is_alive():
                    return
                if self._starting or self._stop_epoch != epoch:
                    return
                self._starting = True
                starting = True
                self._stop.clear()
                self._finalized = False
                try:
                    self._source.connect()
                except EVENT_SOURCE_ERRORS as exc:
                    connect_error = exc
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
                    return
            self._drain_stopped_start()
        except Exception:
            if starting and not self._finalized:
                self._drain_stopped_start()
            raise
        finally:
            if starting:
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
        if not self._try_finalize(thread, timeout=join_timeout_seconds):
            return False
        return joined

    def _try_finalize(self, thread: threading.Thread | None, *, timeout: float) -> bool:
        if not self._tick_guard.acquire(blocking=True, timeout=timeout):
            return False
        try:
            if not self._source_guard.acquire(blocking=False):
                return False
            try:
                if self._stop.is_set() and self._thread is thread:
                    self._drain_and_close()
                return True
            finally:
                self._source_guard.release()
        finally:
            self._tick_guard.release()

    def _holder_should_finalize(self) -> bool:
        thread = self._thread
        return thread is None or thread is threading.current_thread() or not thread.is_alive()

    def refresh_source_health_metrics(self) -> bool:
        try:
            health = self._source.health()
        except EVENT_SOURCE_ERRORS:
            logger.exception("Failed to read event source health for metrics")
            update_events_source_health(connected=False, lag=None)
            return False
        update_events_source_health(connected=health.connected, lag=health.lag)
        return bool(health.connected)

    def _close_source(self) -> None:
        if not isinstance(self._source, ClosableSource):
            return
        try:
            self._source.close()
        except EVENT_SOURCE_ERRORS:
            logger.exception("Event source close() failed during worker stop")

    def _drain_and_close(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        try:
            self._drain_buffer_on_stop()
        except EVENT_WORKER_ERRORS:
            logger.exception("Event worker drain on stop failed")
        finally:
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

    def _take_deferred_acks(self) -> list[NormalizedEvent]:
        pending = self._deferred_acks
        self._deferred_acks = []
        return pending

    def _take_pending_acks(self) -> list[NormalizedEvent]:
        pending = self._take_deferred_acks() + self._retry_acks
        self._retry_acks = []
        return pending

    def _ack_live_ids(self, event_ids: Sequence[str]) -> set[str]:
        result = self._source.ack(event_ids)
        if result is None:
            return {str(event_id) for event_id in event_ids}
        return {str(event_id) for event_id in result}

    def _drop_retry_acks(self, live: Sequence[NormalizedEvent]) -> None:
        if not self._retry_acks or not live:
            return
        live_ids = {event.event_id for event in live}
        live_fps = {event_fingerprint(event) for event in live if self._fingerprint_dedupe(event)}
        self._retry_acks = [
            event
            for event in self._retry_acks
            if event.event_id not in live_ids
            and not (self._fingerprint_dedupe(event) and event_fingerprint(event) in live_fps)
        ]

    def _fingerprint_dedupe(self, event: NormalizedEvent) -> bool:
        return self._ephemeral_event_ids and event.generated_event_id

    def _replay_key(self, event: NormalizedEvent) -> str | None:
        # A generated id is not stable and identical payloads can be distinct events, so those stay
        # at-least-once: no durable key, only the in-memory fingerprint dedupe.
        if self._fingerprint_dedupe(event):
            return None
        return event_replay_key(event)

    def _remember_applied(self, events: Sequence[NormalizedEvent]) -> None:
        for event in events:
            self._applied_event_ids.add(event.event_id)
            if self._fingerprint_dedupe(event):
                self._applied_fingerprints.add(event_fingerprint(event))

    def _is_applied(self, event: NormalizedEvent) -> bool:
        if event.event_id in self._applied_event_ids:
            return True
        return self._fingerprint_dedupe(event) and event_fingerprint(event) in self._applied_fingerprints

    def _still_unapplied(self, event: NormalizedEvent) -> bool:
        if self._buffer.contains_event_id(event.event_id):
            return True
        if self._fingerprint_dedupe(event):
            fingerprint = event_fingerprint(event)
            if self._buffer.contains_fingerprint(fingerprint):
                return True
            return any(
                held.event_id == event.event_id
                or (self._fingerprint_dedupe(held) and event_fingerprint(held) == fingerprint)
                for held in self._held
            )
        return any(held.event_id == event.event_id for held in self._held)

    def _ack_unbuffered(self, events: Sequence[NormalizedEvent]) -> None:
        to_ack: list[NormalizedEvent] = []
        for event in events:
            if self._still_unapplied(event):
                self._deferred_acks.append(event)
            else:
                to_ack.append(event)
        if to_ack:
            self._drop_retry_acks(self._ack_tracked(to_ack))

    def _ack_deferred_matching(self, applied: Sequence[NormalizedEvent]) -> None:
        if not self._deferred_acks:
            return
        applied_ids = {event.event_id for event in applied}
        applied_fps = {event_fingerprint(event) for event in applied if self._fingerprint_dedupe(event)}
        keep: list[NormalizedEvent] = []
        matched: list[NormalizedEvent] = []
        for event in self._deferred_acks:
            matched_id = event.event_id in applied_ids
            matched_fp = self._fingerprint_dedupe(event) and event_fingerprint(event) in applied_fps
            if (matched_id or matched_fp) and not self._still_unapplied(event):
                matched.append(event)
            else:
                keep.append(event)
        self._deferred_acks = keep
        if matched:
            self._drop_retry_acks(self._ack_tracked(matched))

    def _flush_retry_acks(self) -> None:
        if not self._retry_acks:
            return
        batch = self._retry_acks
        self._retry_acks = []
        self._settle_live(self._ack_tracked(batch))

    def _ack_tracked(self, events: Sequence[NormalizedEvent]) -> list[NormalizedEvent]:
        """Ack ``events`` and return those still live; failed or missing acks are kept for retry."""
        try:
            live_ids = self._ack_live_ids([event.event_id for event in events])
        except BaseException:
            self._retry_acks.extend(events)
            raise
        missing = [event for event in events if event.event_id not in live_ids]
        # The source no longer holds these deliveries, so cap them; a failed ack above retries without limit.
        dropped = len(missing) - max(0, _APPLIED_EVENT_ID_CAP - len(self._retry_acks))
        if dropped > 0:
            logger.warning("Dropping %d event ack(s) for deliveries the source no longer holds", dropped)
            del missing[:dropped]
        self._retry_acks.extend(missing)
        return [event for event in events if event.event_id in live_ids]

    def _settle_live(self, live: Sequence[NormalizedEvent]) -> None:
        if live:
            self._ack_deferred_matching(live)
            self._drop_retry_acks(live)

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
        completed = False
        try:
            rejected = self._rejected_nacks(leftover, self._source.nack(leftover))
            completed = True
        except EVENT_SOURCE_ERRORS:
            logger.exception(
                "Event worker failed to return %d event(s) to the source",
                len(leftover),
            )
            return
        finally:
            if not completed:
                self._restore_buffer(leftover)
        if rejected:
            self._restore_buffer(rejected)

    def _reconnect_source(self) -> bool:
        with self._tick_guard:
            with self._source_guard:
                if self._stop.is_set():
                    self._drain_and_close()
                    return False
                leftover = self._take_buffered()
                leftover.extend(self._take_deferred_acks())
                try:
                    self._source.connect()
                except EVENT_SOURCE_ERRORS:
                    logger.exception("Event source reconnect failed")
                    self._restore_buffer(leftover)
                    if self._stop.is_set():
                        self._drain_and_close()
                    return False
                self._return_events(leftover)
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
                except EVENT_WORKER_ERRORS:
                    record_events_tick_error()
                    logger.exception("Event worker tick failed")
                disconnected = self._source_unhealthy
                self._stop.wait(self._poll_interval_seconds)
        finally:
            with self._tick_guard, self._source_guard:
                if self._thread is current:
                    self._drain_and_close()

    def tick(self) -> int:
        """One poll/flush cycle; returns events successfully applied."""
        with self._tick_guard:
            try:
                if self._stop.is_set():
                    return 0
                return self._tick_locked()
            finally:
                if self._stop.is_set() and self._holder_should_finalize():
                    with self._source_guard:
                        self._drain_and_close()
                elif not self._stop.is_set():
                    self._source_unhealthy = not self.refresh_source_health_metrics()

    def _tick_locked(self) -> int:
        self._flush_retry_acks()
        if self._apply_lock is None:
            self._poll_into_buffer()
            ready = self._buffer.flush_if_ready()
            return self._repair_then_flush(ready)

        if self._poll_without_lock:
            self._poll_into_buffer()
            ready = self._buffer.flush_if_ready()
            if not ready:
                if self._updater.post_write_pending and self._acquire_apply_lock():
                    try:
                        self._repair_post_write()
                    finally:
                        self._release_apply_lock()
                return 0
            return self._with_apply_lock(ready)

        if not self._acquire_apply_lock():
            return 0
        try:
            self._poll_into_buffer()
            ready = self._buffer.flush_if_ready()
            return self._repair_then_flush(ready)
        finally:
            self._release_apply_lock()

    def _repair_then_flush(self, ready: list[NormalizedEvent]) -> int:
        repaired = False
        try:
            self._repair_post_write()
            repaired = True
        finally:
            if not repaired and ready:
                self._return_events(ready)
        return self._flush_ready(ready) if ready else 0

    def _repair_post_write(self) -> None:
        if not self._updater.post_write_pending:
            return
        try:
            self._updater.repair_post_write()
        except LockLostError as exc:
            record_events_tick_error()
            if exc.kind == "apply":
                update_events_leader(False)
            logger.error("%s; post-write repair retried later", exc)
        except _REPAIR_ERRORS:
            record_events_tick_error()
            logger.exception("Post-write repair failed; retrying later")

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
            return self._repair_then_flush(ready)
        finally:
            self._release_apply_lock()

    def _drain_buffer_on_stop(self) -> None:
        leftover = self._take_buffered()
        try:
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
        finally:
            pending = self._take_pending_acks()
            if pending:
                self._return_events(pending)

    def _flush_ready(self, ready: list[NormalizedEvent]) -> int:
        # Rejected events are never applied; they ride along with the batch to its ack.
        valid = [event for event in ready if event.rejected is None]
        try:
            replayed = self._updater.applied_before(valid)
        except Exception as exc:
            record_events_flush(status="error")
            logger.exception("Applied-event lookup failed; returning %d event(s) to source", len(ready))
            self._return_events(ready)
            if isinstance(exc, EVENT_APPLY_ERRORS):
                return 0
            raise
        if replayed:
            ready = self._ack_replayed(ready, replayed)
            valid = [event for event in ready if event.rejected is None]
        if not valid:
            self._log_rejected(ready)
            self._ack_unbuffered(ready)
            return 0
        applied: int | None = None
        post_write_failed = False
        # Re-raised after the ack; it is logged first, so an online-persist error may take precedence.
        unexpected: Exception | None = None
        apply_started = False
        try:
            try:
                with inflight_heartbeat(self._source, ready, self._heartbeat_interval_seconds):
                    apply_started = True
                    applied = self._updater.apply(valid, persist_online=False)
            except Exception as exc:
                # Once the rows are written any error must still ack them (nacking applies the events
                # twice), so this cannot be narrowed; an unexpected one is re-raised after the ack.
                # last_apply_written describes the previous batch until this batch's apply starts.
                if not applied and not (apply_started and self._updater.last_apply_written):
                    raise
                if not isinstance(exc, _POST_WRITE_ERRORS):
                    unexpected = exc
                # The worker loop counts a re-raised worker error itself.
                counted_by_loop = unexpected is not None and isinstance(unexpected, EVENT_WORKER_ERRORS)
                self._log_post_write_failure(ready, exc, count=not counted_by_loop)
                if isinstance(exc, LockLostError) and exc.kind == "apply":
                    update_events_leader(False)
                applied = applied or len(valid)
                post_write_failed = True
        except HeartbeatError:
            logger.error("In-flight heartbeat failed; returning %d event(s) to source", len(ready))
            return self._reject(ready, "error")
        except LockLostError as exc:
            if exc.kind == "apply":
                update_events_leader(False)
            logger.error("%s; nacking %d event(s)", exc, len(ready))
            return self._reject(ready, "error")
        except WriterLockBusyError as exc:
            record_events_apply_busy(reason="lock")
            logger.info("%s; nacking %d event(s)", exc, len(ready))
            return self._reject(ready, "busy")
        except Exception as exc:
            logger.exception("Incremental apply failed; returning %d event(s) to source", len(ready))
            self._reject(ready, "error")
            if isinstance(exc, EVENT_APPLY_ERRORS):
                return 0
            raise
        if applied and not post_write_failed and self._updater.last_publish_failed:
            # Written and acked; the publish is queued for repair, but this flush did not complete.
            record_events_tick_error()
            post_write_failed = True
        if applied == 0:
            record_events_apply_busy(reason="retrain")
            return self._reject(ready, "busy")
        if applied != len(valid):
            logger.error(
                "Incremental apply returned %d for %d ready event(s); nacking batch",
                applied,
                len(valid),
            )
            return self._reject(ready, "error")
        self._remember_applied(valid)
        self._log_rejected(ready)
        try:
            live = self._ack_tracked(ready)
        except TimeoutError:
            record_events_flush(status="error")
            logger.exception("Event source ack timed out after successful apply; persisting without nack")
            self._persist_online_after_ack()
            if unexpected is not None:
                raise unexpected from None
            return applied
        except Exception:
            record_events_flush(status="error")
            logger.exception("Event source ack failed after successful apply; persisting without nack")
            self._persist_online_after_ack()
            raise
        try:
            self._settle_live(live)
        except Exception:
            record_events_flush(status="error")
            logger.exception(
                "Event source deferred ack failed after successful apply; persisting without nack"
            )
            self._persist_online_after_ack()
            raise
        if post_write_failed:
            record_events_flush(status="error")
            self._persist_online_after_ack()
            if unexpected is not None:
                raise unexpected
            return applied
        self._persist_online_after_ack()
        record_events_flush(status="success", events=applied)
        return applied

    def _reject(self, ready: list[NormalizedEvent], status: str) -> int:
        record_events_flush(status=status)
        self._updater.abort_online()
        self._return_events(ready)
        return 0

    def _ack_replayed(
        self, ready: list[NormalizedEvent], replayed: list[NormalizedEvent]
    ) -> list[NormalizedEvent]:
        # Recorded by an earlier write (e.g. before a restart): ack instead of applying twice.
        logger.info("Acking %d event(s) already applied by an earlier write", len(replayed))
        record_events_replayed(len(replayed))
        self._remember_applied(replayed)
        skip = {id(event) for event in replayed}
        fresh = [event for event in ready if id(event) not in skip]
        try:
            self._ack_unbuffered(replayed)
        except EVENT_SOURCE_ERRORS:
            logger.exception("Event source ack failed for already-applied event(s); retrying later")
        except Exception:
            record_events_flush(status="error")
            self._return_events(fresh)
            raise
        return fresh

    def _log_rejected(self, events: Sequence[NormalizedEvent]) -> None:
        for event in events:
            if event.rejected is not None:
                logger.warning("Skipping invalid event row %s", event.rejected)

    def _log_post_write_failure(
        self, ready: list[NormalizedEvent], exc: BaseException, *, count: bool = True
    ) -> None:
        if count:
            record_events_tick_error()
        logger.error("%s; acking %d written event(s)", exc, len(ready), exc_info=exc)

    def _persist_online_after_ack(self) -> None:
        persisted = False
        try:
            for attempt in range(1, _ONLINE_PERSIST_ATTEMPTS + 1):
                try:
                    self._updater.persist_online()
                    persisted = True
                    return
                except LockLostError as exc:
                    logger.error("%s; dropping pending artifact", exc)
                    return
                except WriterLockBusyError as exc:
                    logger.info(
                        "%s; online persist after ack (attempt %d/%d)",
                        exc,
                        attempt,
                        _ONLINE_PERSIST_ATTEMPTS,
                    )
                except EVENT_APPLY_ERRORS:
                    logger.exception(
                        "Online artifact persist failed after ack (attempt %d/%d)",
                        attempt,
                        _ONLINE_PERSIST_ATTEMPTS,
                    )
            logger.error("Online artifact persist gave up after ack; pending fit dropped")
        finally:
            if not persisted:
                self._updater.abort_online()
