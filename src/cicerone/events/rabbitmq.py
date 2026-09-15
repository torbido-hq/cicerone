"""RabbitMQ EventSource (queue consume + manual ack)."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Sequence
from contextlib import suppress
from datetime import datetime
from functools import partial
from typing import Any

from cicerone.amqp_options import (
    amqp_timeout_seconds,
    apply_amqp_timeouts,
    prefetch_count,
    require_amqp_url,
    require_queue,
)
from cicerone.config.constants import ConfigError
from cicerone.events.base import EventSource, EventSourceHealth, NormalizedEvent
from cicerone.events.json_payload import decode_json_object
from cicerone.events.normalize import EventNormalizeError, normalize_event
from cicerone.events.rabbitmq_io import (
    _close_handles,
    _PikaIo,
    _release_io,
)

logger = logging.getLogger(__name__)

_EVENTS_PREFIX = "events.options"


def validate_rabbitmq_event_options(options: dict[str, Any]) -> None:
    require_amqp_url(options, prefix=_EVENTS_PREFIX)
    require_queue(options, prefix=_EVENTS_PREFIX)
    prefetch_count(options, prefix=_EVENTS_PREFIX)
    amqp_timeout_seconds(options, prefix=_EVENTS_PREFIX)


def _missing_extra() -> ConfigError:
    return ConfigError(
        'events.kind = "rabbitmq" requires the pika package; '
        "install with: pip install 'cicerone-recommender[rabbitmq]'"
    )


class RabbitMQEventSource(EventSource):
    """Consume JSON events from one queue; ack with ``basic_ack``."""

    ephemeral_event_ids = True

    def __init__(self, options: dict[str, Any]):
        validate_rabbitmq_event_options(options)
        self._amqp_url = require_amqp_url(options, prefix=_EVENTS_PREFIX)
        self._queue = require_queue(options, prefix=_EVENTS_PREFIX)
        self._prefetch = prefetch_count(options, prefix=_EVENTS_PREFIX)
        self._timeout_seconds = amqp_timeout_seconds(options, prefix=_EVENTS_PREFIX)

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
        self._event_io: dict[int, tuple[_PikaIo, str]] = {}
        self._last_event_at: datetime | None = None

    def connect(self) -> None:
        try:
            import pika
        except ImportError as exc:
            raise _missing_extra() from exc

        io = _PikaIo(self._timeout_seconds)
        io.start()
        try:
            connection, channel = io.submit(partial(self._open, pika, io))
        except Exception as exc:
            io.abandon(None, io._connection)
            raise ConfigError(f"events.options.amqp_url is unreachable: {exc}") from exc

        with self._lock:
            previous_io = self._io
            previous_channel = self._channel
            previous_connection = self._connection
            carried = list(self._pending)
            self._io = io
            self._connection = connection
            self._channel = channel
            self._connected = True
            self._pending.clear()
            self._pending_ids.clear()
            self._in_flight.clear()
            self._delivery_tags.clear()
            self._held_tags.clear()
            self._event_io.clear()
            for event in carried:
                self._pending.append(event)
                self._pending_ids.add(event.event_id)
        if previous_io is not None:
            _release_io(previous_io, previous_channel, previous_connection)

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
            self._event_io.clear()
        if io is None:
            return
        _release_io(io, channel, connection)

    def poll(self, max_events: int = 100) -> Sequence[NormalizedEvent]:
        if max_events < 1:
            return []
        io = self._require_io()
        claimed: list[tuple[NormalizedEvent, int | None]] = []
        with self._lock:
            while self._pending and len(claimed) < max_events:
                event = self._pending.popleft()
                self._pending_ids.discard(event.event_id)
                tag = self._delivery_tags.get(event.event_id)
                self._in_flight.add(event.event_id)
                claimed.append((event, tag))

        remaining = max_events - len(claimed)
        while remaining > 0:
            if not self._owns_io(io):
                break
            try:
                method, _properties, body = io.submit(partial(self._basic_get, io))
            except Exception:
                logger.exception("RabbitMQ basic_get failed")
                io._mark_failed()
                break
            if method is None:
                break
            incoming = self._delivery_to_event(io, method, body)
            if incoming is None:
                if not self._owns_io(io):
                    break
                continue
            with self._lock:
                tag = self._delivery_tags.get(incoming.event_id)
            if tag is None:
                continue
            claimed.append((incoming, tag))
            remaining -= 1

        with self._lock:
            out = [
                event
                for event, tag in claimed
                if self._io is io and (tag is None or self._delivery_tags.get(event.event_id) == tag)
            ]
            if out:
                self._last_event_at = max(event.occurred_at for event in out)
            for event in out:
                self._event_io[id(event)] = (io, event.event_id)
        return out

    def ack(self, event_ids: Sequence[str]) -> Sequence[str]:
        if not event_ids:
            return ()
        io = self._require_io()
        confirmed: list[str] = []
        with self._lock:
            if self._io is not io:
                return ()
            resolved: list[tuple[str, int]] = []
            local_only: list[str] = []
            for event_id in event_ids:
                eid = str(event_id)
                tag = self._delivery_tags.get(eid)
                if tag is not None:
                    resolved.append((eid, tag))
                elif eid in self._in_flight or eid in self._pending_ids:
                    local_only.append(eid)
            for eid in local_only:
                self._forget_event(eid)
                confirmed.append(eid)
        if not resolved:
            return tuple(confirmed)
        for eid, tag in resolved:
            if not self._owns_io(io):
                return tuple(confirmed)
            io.submit(partial(self._basic_ack, io, tag))
            with self._lock:
                if self._io is not io or self._delivery_tags.get(eid) != tag:
                    continue
                self._delivery_tags.pop(eid, None)
                self._held_tags.discard(tag)
                self._forget_event(eid)
                confirmed.append(eid)
        return tuple(confirmed)

    def _forget_event(self, eid: str) -> None:
        self._in_flight.discard(eid)
        self._pending_ids.discard(eid)
        stale = [key for key, (_owner, event_id) in self._event_io.items() if event_id == eid]
        for key in stale:
            self._event_io.pop(key, None)

    def nack(self, events: Sequence[NormalizedEvent]) -> Sequence[NormalizedEvent]:
        if not events:
            return ()
        with self._lock:
            io = self._io
            if io is None or io.failed or io.closing:
                return tuple(events)
            retained: list[NormalizedEvent] = []
            for event in reversed(list(events)):
                owner = self._event_io.get(id(event))
                if owner is None or owner[0] is not io:
                    continue
                if event.event_id not in self._delivery_tags:
                    continue
                retained.append(event)
            if self._io is not io or io.failed or io.closing:
                return tuple(events)
            kept: set[int] = set()
            added: list[NormalizedEvent] = []
            for event in retained:
                self._in_flight.discard(event.event_id)
                kept.add(id(event))
                if event.event_id in self._pending_ids:
                    continue
                self._pending.appendleft(event)
                self._pending_ids.add(event.event_id)
                added.append(event)
            if self._io is not io or io.failed or io.closing:
                for event in added:
                    with suppress(ValueError):
                        self._pending.remove(event)
                    self._pending_ids.discard(event.event_id)
                return tuple(events)
        return tuple(event for event in events if id(event) not in kept)

    def heartbeat(self, events: Sequence[NormalizedEvent]) -> None:
        del events
        io = self._io
        if io is None:
            return
        try:
            io.submit(partial(self._pump_connection, io))
        except Exception:
            logger.exception("RabbitMQ heartbeat process_data_events failed")
            raise

    def health(self) -> EventSourceHealth:
        with self._lock:
            connected = self._connected
            io = self._io
            channel = self._channel
            local_held = len(self._pending_ids) + len(self._in_flight)
            last_event_at = self._last_event_at
        if not connected or io is None or channel is None or io.failed or io.closing:
            return EventSourceHealth(connected=False, lag=None, last_event_at=last_event_at)
        ready = 0
        try:
            declared = io.submit(partial(self._passive_declare, io))
            ready = int(declared.method.message_count)
        except Exception:
            logger.exception("RabbitMQ queue_declare (passive) failed")
            if io.failed or io.closing or not self._owns_io(io):
                return EventSourceHealth(connected=False, lag=None, last_event_at=last_event_at)
            ready = 0
        if io.failed or io.closing or not self._owns_io(io):
            return EventSourceHealth(connected=False, lag=None, last_event_at=last_event_at)
        lag = ready + local_held
        return EventSourceHealth(
            connected=True,
            lag=lag,
            last_event_at=last_event_at,
            detail=f"queue={self._queue}",
        )

    def _basic_get(self, io: _PikaIo) -> Any:
        return io.broker_channel().basic_get(self._queue, auto_ack=False)

    def _basic_ack(self, io: _PikaIo, tag: int) -> None:
        io.broker_channel().basic_ack(delivery_tag=tag)

    def _passive_declare(self, io: _PikaIo) -> Any:
        return io.broker_channel().queue_declare(queue=self._queue, durable=True, passive=True)

    def _open(self, pika: Any, io: _PikaIo) -> tuple[Any, Any]:
        if io.failed:
            raise RuntimeError("RabbitMQ I/O worker abandoned")
        connection = pika.BlockingConnection(
            apply_amqp_timeouts(pika.URLParameters(self._amqp_url), self._timeout_seconds)
        )
        channel = None
        try:
            io._bind_handles(connection=connection)
            channel = connection.channel()
            io._bind_handles(channel=channel)
            channel.basic_qos(prefetch_count=self._prefetch)
            channel.queue_declare(queue=self._queue, durable=True)
        except Exception:
            io._channel = None
            io._connection = None
            _close_handles(channel, connection)
            raise
        return connection, channel

    def _pump_connection(self, io: _PikaIo) -> None:
        connection = io.broker_connection()
        try:
            connection.process_data_events(time_limit=0)
        except Exception:
            io._mark_failed()
            raise

    def _require_io(self) -> _PikaIo:
        with self._lock:
            if not self._connected or self._io is None:
                raise RuntimeError("RabbitMQEventSource is not connected")
            return self._io

    def _owns_io(self, io: _PikaIo) -> bool:
        with self._lock:
            return self._io is io and not io.failed

    def _delivery_to_event(self, io: _PikaIo, method: Any, body: Any) -> NormalizedEvent | None:
        tag = int(method.delivery_tag)
        with self._lock:
            if self._io is not io or tag in self._held_tags:
                return None
        try:
            payload = decode_json_object(body)
        except EventNormalizeError as exc:
            logger.warning("Skipping invalid RabbitMQ message %s: %s", tag, exc)
            self._ack_discard(io, tag)
            return None
        try:
            event = normalize_event(payload)
        except EventNormalizeError as exc:
            logger.warning("Skipping invalid RabbitMQ message %s: %s", tag, exc)
            self._ack_discard(io, tag)
            return None
        with self._lock:
            if self._io is not io:
                return None
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
            io.submit(partial(self._basic_ack, io, tag))
        except Exception:
            logger.exception("Failed to ack discarded RabbitMQ message")
