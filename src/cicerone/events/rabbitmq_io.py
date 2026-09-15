"""Single-thread pika I/O worker for RabbitMQEventSource."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)

_IO_STOP = object()
_IO_IDLE_SECONDS = 0.5
_JOB_QUEUED = "queued"
_JOB_CLAIMED = "claimed"
_JOB_RUNNING = "running"
_JOB_STARTED = "started"
_JOB_INVOKING = "invoking"
_JOB_DISPATCHED = "dispatched"
_JOB_ABANDONED = "abandoned"


class _IoJob:
    __slots__ = ("fn", "reply", "state", "run_lock", "permit")

    def __init__(self, fn: Callable[[], Any], reply: queue.Queue[tuple[str, Any]]) -> None:
        self.fn = fn
        self.reply = reply
        self.state = _JOB_QUEUED
        self.run_lock = threading.Lock()
        self.permit = True


class _PikaIo:
    """Run BlockingConnection calls on one thread (pika is not thread-safe)."""

    def __init__(self, timeout_seconds: float) -> None:
        self._jobs: queue.Queue[Any] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="cicerone-amqp-io", daemon=True)
        self._connection: Any | None = None
        self._channel: Any | None = None
        self._timeout_seconds = timeout_seconds
        self._failed = False
        self._closing = False
        self._state_lock = threading.Lock()
        self._busy = 0
        self._abandon_channel: Any | None = None
        self._abandon_connection: Any | None = None

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def closing(self) -> bool:
        return self._closing

    @property
    def busy(self) -> bool:
        with self._state_lock:
            return self._busy > 0

    def _clear_busy(self) -> None:
        with self._state_lock:
            if self._busy > 0:
                self._busy -= 1

    def _try_begin_shutdown(self) -> bool:
        with self._state_lock:
            if self._failed or self._closing or self._busy > 0:
                return False
            self._closing = True
            return True

    def start(self) -> None:
        self._thread.start()

    def submit(self, fn: Callable[[], Any], *, allow_closing: bool = False) -> Any:
        reply: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        job = _IoJob(fn, reply)

        def _guarded() -> Any:
            with self._state_lock:
                if self._failed or not job.permit or job.state != _JOB_STARTED:
                    raise RuntimeError("RabbitMQ I/O worker abandoned")
                job.state = _JOB_INVOKING
            job.run_lock.acquire()
            try:
                with self._state_lock:
                    if self._failed or not job.permit or job.state != _JOB_INVOKING:
                        raise RuntimeError("RabbitMQ I/O worker abandoned")
                    job.state = _JOB_DISPATCHED
                    if self._failed or not job.permit:
                        job.state = _JOB_ABANDONED
                        raise RuntimeError("RabbitMQ I/O worker abandoned")
                with self._state_lock:
                    if self._failed or not job.permit:
                        job.state = _JOB_ABANDONED
                        raise RuntimeError("RabbitMQ I/O worker abandoned")
                return fn()
            finally:
                job.run_lock.release()

        job.fn = _guarded
        with self._state_lock:
            if self._failed or not self._thread.is_alive() or (self._closing and not allow_closing):
                raise RuntimeError("RabbitMQ I/O thread is not running")
            self._busy += 1
            self._jobs.put(job)
        try:
            status, payload = reply.get(timeout=self._timeout_seconds)
        except queue.Empty as exc:
            self._abandon_unclaimed(job)
            raise TimeoutError(f"RabbitMQ I/O call timed out after {self._timeout_seconds}s") from exc
        finally:
            if not self._failed:
                self._clear_busy()
        if status == "err":
            raise payload
        if self._failed:
            raise RuntimeError("RabbitMQ I/O worker abandoned")
        return payload

    def _mark_failed(self) -> None:
        with self._state_lock:
            self._failed = True
            self._detach_handles()

    def _detach_handles(self) -> None:
        if self._abandon_channel is None:
            self._abandon_channel = self._channel
        if self._abandon_connection is None:
            self._abandon_connection = self._connection
        self._channel = None
        self._connection = None

    def broker_channel(self) -> Any:
        with self._state_lock:
            if self._failed or self._channel is None:
                raise RuntimeError("RabbitMQ I/O worker abandoned")
            return self._channel

    def broker_connection(self) -> Any:
        with self._state_lock:
            if self._failed or self._connection is None:
                raise RuntimeError("RabbitMQ I/O worker abandoned")
            return self._connection

    def _bind_handles(self, *, connection: Any | None = None, channel: Any | None = None) -> None:
        with self._state_lock:
            if self._failed:
                raise RuntimeError("RabbitMQ I/O worker abandoned")
            if connection is not None:
                self._connection = connection
            if channel is not None:
                self._channel = channel

    def _abandon_unclaimed(self, job: _IoJob) -> None:
        with self._state_lock:
            self._failed = True
            job.permit = False
            job.state = _JOB_ABANDONED
            self._detach_handles()

    def _take_job(self, job: _IoJob) -> bool:
        with self._state_lock:
            if self._failed or job.state != _JOB_QUEUED:
                job.state = _JOB_ABANDONED
                return False
            job.state = _JOB_CLAIMED
            return True

    def _enter_job(self, job: _IoJob) -> bool:
        with self._state_lock:
            if self._failed or job.state != _JOB_CLAIMED:
                job.state = _JOB_ABANDONED
                return False
            job.state = _JOB_RUNNING
            return True

    def _should_run(self, job: _IoJob) -> bool:
        with self._state_lock:
            if self._failed or job.state != _JOB_RUNNING:
                job.state = _JOB_ABANDONED
                return False
            job.state = _JOB_STARTED
            return True

    def abandon(self, channel: Any, connection: Any) -> None:
        self._mark_failed()
        if channel is not None:
            self._abandon_channel = channel
        if connection is not None:
            self._abandon_connection = connection
        self.stop()

    def stop(self) -> None:
        self._jobs.put(_IO_STOP)
        self._thread.join(timeout=0.1 if self._failed else 5.0)

    def _cleanup_abandoned(self) -> None:
        channel = self._abandon_channel if self._abandon_channel is not None else self._channel
        connection = self._abandon_connection if self._abandon_connection is not None else self._connection
        leftover_channel = self._channel
        leftover_connection = self._connection
        self._connection = None
        self._channel = None
        self._abandon_channel = None
        self._abandon_connection = None
        _close_handles(channel, connection)
        if leftover_channel is not None and leftover_channel is not channel:
            _close_quietly(leftover_channel, "channel")
        if leftover_connection is not None and leftover_connection is not connection:
            _close_quietly(leftover_connection, "connection")

    def _loop(self) -> None:
        while True:
            try:
                job = self._jobs.get(timeout=_IO_IDLE_SECONDS)
            except queue.Empty:
                if self._failed:
                    self._exit_failed()
                    return
                with self._state_lock:
                    if self._failed:
                        self._exit_failed()
                        return
                    if self._closing:
                        continue
                    self._busy += 1
                try:
                    self._pump()
                finally:
                    if not self._failed:
                        self._clear_busy()
                if self._failed:
                    self._exit_failed()
                    return
                continue
            if job is _IO_STOP:
                if self._failed:
                    self._exit_failed()
                return
            if not self._take_job(job):
                with suppress(queue.Full):
                    job.reply.put_nowait(("err", RuntimeError("RabbitMQ I/O worker abandoned")))
                self._exit_failed()
                return
            if not self._enter_job(job):
                with suppress(queue.Full):
                    job.reply.put_nowait(("err", RuntimeError("RabbitMQ I/O worker abandoned")))
                self._exit_failed()
                return
            if not self._should_run(job):
                with suppress(queue.Full):
                    job.reply.put_nowait(("err", RuntimeError("RabbitMQ I/O worker abandoned")))
                self._exit_failed()
                return
            try:
                result = job.fn()
            except Exception as exc:
                payload: tuple[str, Any] = ("err", exc)
            else:
                payload = ("ok", result)
            if self._failed:
                payload = ("err", RuntimeError("RabbitMQ I/O worker abandoned"))
            with suppress(queue.Full):
                job.reply.put_nowait(payload)
            if self._failed:
                self._exit_failed()
                return

    def _pump(self) -> None:
        connection = self._connection
        if connection is None:
            return
        try:
            connection.process_data_events(time_limit=0)
        except Exception:
            self._mark_failed()
            logger.exception("RabbitMQ I/O thread process_data_events failed")

    def _fail_pending(self, exc: BaseException) -> None:
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                return
            if job is _IO_STOP:
                continue
            with suppress(queue.Full):
                job.reply.put_nowait(("err", exc))

    def _exit_failed(self) -> None:
        self._fail_pending(RuntimeError("RabbitMQ I/O worker abandoned"))
        self._cleanup_abandoned()


def _release_io(io: _PikaIo, channel: Any, connection: Any) -> None:
    if io.failed or not io._try_begin_shutdown():
        io.abandon(channel, connection)
        if not io._thread.is_alive():
            _close_handles(channel, connection)
        return
    try:

        def _shutdown() -> None:
            io._connection = None
            _close_handles(channel, connection)

        io.submit(_shutdown, allow_closing=True)
    except Exception:
        logger.exception("Failed to close RabbitMQ connection on I/O thread")
        io.abandon(channel, connection)
        return
    io.stop()


def _close_handles(channel: Any, connection: Any) -> None:
    _close_quietly(channel, "channel")
    _close_quietly(connection, "connection")


def _close_quietly(handle: Any, label: str) -> None:
    if handle is None:
        return
    closer = getattr(handle, "close", None)
    if not callable(closer):
        return
    try:
        closer()
    except Exception:
        logger.exception("Failed to close RabbitMQ %s", label)
