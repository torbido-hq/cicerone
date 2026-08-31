"""Kafka EventSource (consumer group, manual offset commits)."""

from __future__ import annotations

import logging
import socket
import threading
from collections import deque
from collections.abc import Sequence, Set
from datetime import datetime
from typing import Any

from cicerone.config.constants import ConfigError
from cicerone.events.base import EventSource, EventSourceHealth, NormalizedEvent
from cicerone.events.json_payload import decode_json_object
from cicerone.events.normalize import EventNormalizeError, normalize_event
from cicerone.kafka_options import kafka_client_config, optional_nonempty_str, require_nonempty_str

logger = logging.getLogger(__name__)

_EVENTS_PREFIX = "events.options"


def validate_kafka_event_options(options: dict[str, Any]) -> None:
    kafka_client_config(options, prefix=_EVENTS_PREFIX)
    require_nonempty_str(options, "topic", prefix=_EVENTS_PREFIX)
    require_nonempty_str(options, "group_id", prefix=_EVENTS_PREFIX)
    optional_nonempty_str(options, "consumer_name", prefix=_EVENTS_PREFIX)


def _missing_extra() -> ConfigError:
    return ConfigError(
        'events.kind = "kafka" requires the confluent-kafka package; '
        "install with: pip install 'cicerone-recommender[kafka]'"
    )


class KafkaEventSource(EventSource):
    """Consume JSON events from one topic; ack advances the commit watermark."""

    def __init__(self, options: dict[str, Any]):
        validate_kafka_event_options(options)
        self._conf = kafka_client_config(options, prefix=_EVENTS_PREFIX)
        self._topic = require_nonempty_str(options, "topic", prefix=_EVENTS_PREFIX)
        self._group_id = require_nonempty_str(options, "group_id", prefix=_EVENTS_PREFIX)
        raw_name = optional_nonempty_str(options, "consumer_name", prefix=_EVENTS_PREFIX)
        self._consumer_name = raw_name or socket.gethostname() or "cicerone"

        self._consumer: Any | None = None
        self._connected = False
        self._lock = threading.Lock()
        self._pending: deque[NormalizedEvent] = deque()
        self._pending_ids: set[str] = set()
        self._in_flight: set[str] = set()
        self._messages: dict[str, Any] = {}
        self._held_offsets: set[tuple[int, int]] = set()
        self._max_offset: dict[int, int] = {}
        self._topic_partition: Any | None = None
        self._last_event_at: datetime | None = None

    def connect(self) -> None:
        try:
            from confluent_kafka import Consumer, TopicPartition
        except ImportError as exc:
            raise _missing_extra() from exc

        conf = {
            **self._conf,
            "group.id": self._group_id,
            "client.id": self._consumer_name,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
        }
        consumer = Consumer(conf)
        try:
            consumer.list_topics(topic=self._topic, timeout=10)
        except Exception as exc:
            try:
                consumer.close()
            except Exception:
                logger.exception("Failed to close Kafka consumer after connect error")
            raise ConfigError(f"events.options.bootstrap_servers is unreachable: {exc}") from exc
        consumer.subscribe([self._topic])

        with self._lock:
            previous = self._consumer
            self._consumer = consumer
            self._topic_partition = TopicPartition
            self._connected = True
        if previous is not None and previous is not consumer:
            try:
                previous.close()
            except Exception:
                logger.exception("Failed to close previous Kafka consumer")

    def close(self) -> None:
        with self._lock:
            consumer = self._consumer
            self._consumer = None
            self._connected = False
            self._pending.clear()
            self._pending_ids.clear()
            self._in_flight.clear()
            self._messages.clear()
            self._held_offsets.clear()
            self._max_offset.clear()
            self._topic_partition = None
        if consumer is not None:
            try:
                consumer.close()
            except Exception:
                logger.exception("Kafka consumer close failed")

    def poll(self, max_events: int = 100) -> Sequence[NormalizedEvent]:
        if max_events < 1:
            return []
        consumer = self._require_client()
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
                message = consumer.poll(0.0)
            except Exception:
                logger.exception("Kafka poll failed")
                break
            if message is None:
                break
            incoming = self._message_to_event(consumer, message)
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
        consumer = self._require_client()
        with self._lock:
            resolved: list[tuple[str, Any]] = []
            for event_id in event_ids:
                eid = str(event_id)
                message = self._messages.get(eid)
                if message is not None:
                    resolved.append((eid, message))
        if not resolved:
            return
        with self._lock:
            done: dict[int, set[int]] = {}
            for _, message in resolved:
                partition = int(message.partition())
                done.setdefault(partition, set()).add(int(message.offset()))
            watermarks = {
                partition: self._next_commit_offset(partition, extra_done=offsets)
                for partition, offsets in done.items()
            }
            for eid, message in resolved:
                self._messages.pop(eid, None)
                partition = int(message.partition())
                offset = int(message.offset())
                self._held_offsets.discard((partition, offset))
                self._max_offset[partition] = max(self._max_offset.get(partition, -1), offset)
                self._in_flight.discard(eid)
                self._pending_ids.discard(eid)
        self._commit_watermarks(consumer, watermarks)

    def nack(self, events: Sequence[NormalizedEvent]) -> None:
        if not events:
            return
        with self._lock:
            for event in reversed(list(events)):
                if event.event_id not in self._messages:
                    continue
                self._in_flight.discard(event.event_id)
                if event.event_id in self._pending_ids:
                    continue
                self._pending.appendleft(event)
                self._pending_ids.add(event.event_id)

    def heartbeat(self, events: Sequence[NormalizedEvent]) -> None:
        del events

    def health(self) -> EventSourceHealth:
        with self._lock:
            connected = self._connected
            local_held = len(self._pending_ids) + len(self._in_flight)
            last_event_at = self._last_event_at
        if not connected:
            return EventSourceHealth(connected=False, lag=None, last_event_at=last_event_at)
        return EventSourceHealth(
            connected=True,
            lag=local_held if local_held else 0,
            last_event_at=last_event_at,
            detail=f"topic={self._topic} group={self._group_id} consumer={self._consumer_name}",
        )

    def _require_client(self) -> Any:
        with self._lock:
            if not self._connected or self._consumer is None:
                raise RuntimeError("KafkaEventSource is not connected")
            return self._consumer

    def _message_to_event(self, consumer: Any, message: Any) -> NormalizedEvent | None:
        error = message.error() if hasattr(message, "error") else None
        if error:
            logger.warning("Skipping Kafka message with error: %s", error)
            return None
        partition = int(message.partition())
        offset = int(message.offset())
        held_key = (partition, offset)
        with self._lock:
            if held_key in self._held_offsets:
                return None
            self._max_offset[partition] = max(self._max_offset.get(partition, -1), offset)
        try:
            payload = decode_json_object(message.value())
        except EventNormalizeError as exc:
            logger.warning("Skipping invalid Kafka message %s-%s: %s", partition, offset, exc)
            self._commit_discard(consumer, message)
            return None
        if payload.get("event_id") in (None, "") and payload.get("idempotency_key") in (None, ""):
            payload["event_id"] = f"{partition}-{offset}"
        try:
            event = normalize_event(payload)
        except EventNormalizeError as exc:
            logger.warning("Skipping invalid Kafka message %s-%s: %s", partition, offset, exc)
            self._commit_discard(consumer, message)
            return None
        with self._lock:
            if event.event_id in self._messages:
                logger.warning(
                    "Duplicate event_id %r on Kafka %s-%s; committing duplicate",
                    event.event_id,
                    partition,
                    offset,
                )
                drop = True
            else:
                drop = False
                self._messages[event.event_id] = message
                self._held_offsets.add(held_key)
                self._in_flight.add(event.event_id)
        if drop:
            self._commit_discard(consumer, message)
            return None
        return event

    def _next_commit_offset(self, partition: int, *, extra_done: Set[int] | None = None) -> int | None:
        done = extra_done or frozenset()
        held = {offset for part, offset in self._held_offsets if part == partition} - done
        max_seen = self._max_offset.get(partition, -1)
        if done:
            max_seen = max(max_seen, max(done))
        if held:
            nxt = min(held)
            return None if nxt == 0 else nxt
        if max_seen < 0:
            return None
        return max_seen + 1

    def _commit_watermarks(self, consumer: Any, watermarks: dict[int, int | None]) -> None:
        ctor = self._topic_partition
        if ctor is None:
            raise RuntimeError("KafkaEventSource is not connected")
        for partition, nxt in watermarks.items():
            if nxt is None:
                continue
            consumer.commit(offsets=[ctor(self._topic, partition, nxt)], asynchronous=False)

    def _commit_discard(self, consumer: Any, message: Any) -> None:
        partition = int(message.partition())
        offset = int(message.offset())
        with self._lock:
            nxt = self._next_commit_offset(partition, extra_done={offset})
        try:
            self._commit_watermarks(consumer, {partition: nxt})
        except Exception:
            logger.exception("Failed to commit discarded Kafka message")
