"""RabbitMQ EventSource (queue consume + manual ack)."""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from collections.abc import Callable, Sequence
from datetime import datetime
from functools import partial
from typing import Any

from cicerone.amqp_options import prefetch_count, require_amqp_url, require_queue
from cicerone.config.constants import ConfigError
from cicerone.events.base import EventSource, EventSourceHealth, NormalizedEvent
from cicerone.events.json_payload import decode_json_object
from cicerone.events.normalize import EventNormalizeError, normalize_event

logger = logging.getLogger(__name__)

_EVENTS_PREFIX = "events.options"
_IO_STOP = object()
_IO_IDLE_SECONDS = 0.5


def validate_rabbitmq_event_options(options: dict[str, Any]) -> None:
    require_amqp_url(options, prefix=_EVENTS_PREFIX)
    require_queue(options, prefix=_EVENTS_PREFIX)
    prefetch_count(options, prefix=_EVENTS_PREFIX)


def _missing_extra() -> ConfigError:
    return ConfigError(
        'events.kind = "rabbitmq" requires the pika package; '
        "install with: pip install 'cicerone-recommender[rabbitmq]'"
    )


class _PikaIo:
    """Run BlockingConnection calls on one thread (pika is not thread-safe)."""

    def __init__(self) -> None:
        self._jobs: queue.Queue[Any] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="cicerone-amqp-io", daemon=True)
        self._connection: Any | None = None

    def start(self) -> None:
        self._thread.start()

    def submit(self, fn: Callable[[], Any]) -> Any:
        if not self._thread.is_alive():
            raise RuntimeError("RabbitMQ I/O thread is not running")
        reply: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        self._jobs.put((fn, reply))
        status, payload = reply.get()
        if status == "err":
            raise payload
        return payload

    def stop(self) -> None:
        self._jobs.put(_IO_STOP)
        self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while True:
            try:
                job = self._jobs.get(timeout=_IO_IDLE_SECONDS)
            except queue.Empty:
                self._pump()
                continue
            if job is _IO_STOP:
                return
            fn, reply = job
            try:
                reply.put(("ok", fn()))
            except Exception as exc:
                reply.put(("err", exc))

    def _pump(self) -> None:
        connection = self._connection
        if connection is None:
            return
        try:
            connection.process_data_events(time_limit=0)
        except Exception:
            logger.exception("RabbitMQ I/O thread process_data_events failed")


class RabbitMQEventSource(EventSource):
    """Consume JSON events from one queue; ack with ``basic_ack``."""

    def __init__(self, options: dict[str, Any]):
        validate_rabbitmq_event_options(options)
        self._amqp_url = require_amqp_url(options, prefix=_EVENTS_PREFIX)
        self._queue = require_queue(options, prefix=_EVENTS_PREFIX)
        self._prefetch = prefetch_count(options, prefix=_EVENTS_PREFIX)

        self._io: _PikaIo | None = None
        self._connection: Any | None = None
        self._channel: Any | None = None
        self._connected = False
        self._lock = threading.Lock()
        self._pending: deque[NormalizedEvent] = deque()
        self._pending_ids: set[str] = set()
        self._in_flight: set[str] = set()
        self._delivery_tags: dict[str, int] = {}
        self._held_tags: set[int] = set()
        self._last_event_at: datetime | None = None

    def connect(self) -> None:
        try:
            import pika
        except ImportError as exc:
            raise _missing_extra() from exc

        io = _PikaIo()
        io.start()
        try:
            connection, channel = io.submit(partial(self._open, pika, io))
        except Exception as exc:
            io.stop()
            raise ConfigError(f"events.options.amqp_url is unreachable: {exc}") from exc

        with self._lock:
            previous_io = self._io
            previous_channel = self._channel
            previous_connection = self._connection
            self._io = io
            self._connection = connection
            self._channel = channel
            self._connected = True
        if previous_io is not None:
            try:

                def _close_previous() -> None:
                    previous_io._connection = None
                    _close_handles(previous_channel, previous_connection)

                previous_io.submit(_close_previous)
            except Exception:
                logger.exception("Failed to close previous RabbitMQ connection")
            previous_io.stop()

    def close(self) -> None:
        with self._lock:
            io = self._io
            channel = self._channel
            connection = self._connection
            self._io = None
            self._channel = None
            self._connection = None
            self._connected = False
            self._pending.clear()
            self._pending_ids.clear()
            self._in_flight.clear()
            self._delivery_tags.clear()
            self._held_tags.clear()
        if io is None:
            return
        try:

            def _shutdown() -> None:
                io._connection = None
                _close_handles(channel, connection)

            io.submit(_shutdown)
        except Exception:
            logger.exception("RabbitMQ close on I/O thread failed")
        io.stop()

    def poll(self, max_events: int = 100) -> Sequence[NormalizedEvent]:
        if max_events < 1:
            return []
        io = self._require_io()
        out: list[NormalizedEvent] = []
        with self._lock:
            while self._pending and len(out) < max_events:
                event = self._pending.popleft()
                self._pending_ids.discard(event.event_id)
                self._in_flight.add(event.event_id)
                out.append(event)

        remaining = max_events - len(out)
        while remaining > 0:
            try:
                method, _properties, body = io.submit(self._basic_get)
            except Exception:
                logger.exception("RabbitMQ basic_get failed")
                break
            if method is None:
                break
            incoming = self._delivery_to_event(io, method, body)
            if incoming is None:
                continue
            out.append(incoming)
            remaining -= 1

        if out:
            newest = max(event.occurred_at for event in out)
            with self._lock:
                self._last_event_at = newest
        return out

    def ack(self, event_ids: Sequence[str]) -> None:
        if not event_ids:
            return
        io = self._require_io()
        with self._lock:
            resolved: list[tuple[str, int]] = []
            for event_id in event_ids:
                eid = str(event_id)
                tag = self._delivery_tags.get(eid)
                if tag is not None:
                    resolved.append((eid, tag))
        if not resolved:
            return
        for _, tag in resolved:
            io.submit(partial(self._basic_ack, tag))
        with self._lock:
            for eid, tag in resolved:
                self._delivery_tags.pop(eid, None)
                self._held_tags.discard(tag)
                self._in_flight.discard(eid)
                self._pending_ids.discard(eid)

    def nack(self, events: Sequence[NormalizedEvent]) -> None:
        if not events:
            return
        with self._lock:
            for event in reversed(list(events)):
                if event.event_id not in self._delivery_tags:
                    continue
                self._in_flight.discard(event.event_id)
                if event.event_id in self._pending_ids:
                    continue
                self._pending.appendleft(event)
                self._pending_ids.add(event.event_id)

    def heartbeat(self, events: Sequence[NormalizedEvent]) -> None:
        del events
        io = self._io
        if io is None:
            return
        try:
            io.submit(self._pump_connection)
        except Exception:
            logger.exception("RabbitMQ heartbeat process_data_events failed")

    def health(self) -> EventSourceHealth:
        with self._lock:
            connected = self._connected
            io = self._io
            channel = self._channel
            local_held = len(self._pending_ids) + len(self._in_flight)
            last_event_at = self._last_event_at
        if not connected or io is None or channel is None:
            return EventSourceHealth(connected=False, lag=None, last_event_at=last_event_at)
        ready = 0
        try:
            declared = io.submit(self._passive_declare)
            ready = int(declared.method.message_count)
        except Exception:
            logger.exception("RabbitMQ queue_declare (passive) failed")
            ready = 0
        lag = ready + local_held
        return EventSourceHealth(
            connected=True,
            lag=lag,
            last_event_at=last_event_at,
            detail=f"queue={self._queue}",
        )

    def _basic_get(self) -> Any:
        channel = self._channel
        if channel is None:
            return None, None, None
        return channel.basic_get(self._queue, auto_ack=False)

    def _basic_ack(self, tag: int) -> None:
        channel = self._channel
        if channel is None:
            return
        channel.basic_ack(delivery_tag=tag)

    def _passive_declare(self) -> Any:
        channel = self._channel
        if channel is None:
            raise RuntimeError("RabbitMQEventSource is not connected")
        return channel.queue_declare(queue=self._queue, durable=True, passive=True)

    def _open(self, pika: Any, io: _PikaIo) -> tuple[Any, Any]:
        connection = pika.BlockingConnection(pika.URLParameters(self._amqp_url))
        io._connection = connection
        channel = connection.channel()
        channel.basic_qos(prefetch_count=self._prefetch)
        channel.queue_declare(queue=self._queue, durable=True)
        return connection, channel

    def _pump_connection(self) -> None:
        connection = self._connection
        if connection is None:
            return
        connection.process_data_events(time_limit=0)

    def _require_io(self) -> _PikaIo:
        with self._lock:
            if not self._connected or self._io is None:
                raise RuntimeError("RabbitMQEventSource is not connected")
            return self._io

    def _delivery_to_event(self, io: _PikaIo, method: Any, body: Any) -> NormalizedEvent | None:
        tag = int(method.delivery_tag)
        with self._lock:
            if tag in self._held_tags:
                return None
        try:
            payload = decode_json_object(body)
        except EventNormalizeError as exc:
            logger.warning("Skipping invalid RabbitMQ message %s: %s", tag, exc)
            self._ack_discard(io, tag)
            return None
        if payload.get("event_id") in (None, "") and payload.get("idempotency_key") in (None, ""):
            payload["event_id"] = str(tag)
        try:
            event = normalize_event(payload)
        except EventNormalizeError as exc:
            logger.warning("Skipping invalid RabbitMQ message %s: %s", tag, exc)
            self._ack_discard(io, tag)
            return None
        with self._lock:
            if event.event_id in self._delivery_tags:
                logger.warning(
                    "Duplicate event_id %r on RabbitMQ delivery %s; acking duplicate",
                    event.event_id,
                    tag,
                )
                drop = True
            else:
                drop = False
                self._delivery_tags[event.event_id] = tag
                self._held_tags.add(tag)
                self._in_flight.add(event.event_id)
        if drop:
            self._ack_discard(io, tag)
            return None
        return event

    def _ack_discard(self, io: _PikaIo, tag: int) -> None:
        try:
            io.submit(partial(self._basic_ack, tag))
        except Exception:
            logger.exception("Failed to ack discarded RabbitMQ message")


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
