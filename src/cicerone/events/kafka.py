"""Kafka EventSource (consumer group, manual offset commits)."""

from __future__ import annotations

import logging
import socket
from collections import deque
from collections.abc import Sequence, Set
from typing import Any

from cicerone.config.constants import import_extra
from cicerone.events.base import EventSourceError, EventSourceHealth, NormalizedEvent, QueuedEventSource
from cicerone.events.errors import KafkaException
from cicerone.events.json_payload import decode_json_object
from cicerone.events.normalize import EventNormalizeError, normalize_event, supplied_event_id
from cicerone.kafka_options import kafka_client_config, kafka_timeout_seconds
from cicerone.option_parse import optional_nonempty_str, require_nonempty_str

logger = logging.getLogger(__name__)

_EVENTS_PREFIX = "events.options"


def validate_kafka_event_options(options: dict[str, Any]) -> None:
    kafka_client_config(options, prefix=_EVENTS_PREFIX)
    require_nonempty_str(options, "topic", prefix=_EVENTS_PREFIX)
    require_nonempty_str(options, "group_id", prefix=_EVENTS_PREFIX)
    optional_nonempty_str(options, "consumer_name", prefix=_EVENTS_PREFIX)


def _close_quietly(consumer: Any, message: str) -> None:
    try:
        consumer.close()
    except KafkaException:
        logger.exception(message)


class KafkaEventSource(QueuedEventSource):
    """Consume JSON events from one topic; ack advances the commit watermark."""

    def __init__(self, options: dict[str, Any]):
        validate_kafka_event_options(options)
        super().__init__()
        self._conf = kafka_client_config(options, prefix=_EVENTS_PREFIX)
        self._timeout_seconds = kafka_timeout_seconds(options, prefix=_EVENTS_PREFIX)
        self._topic = require_nonempty_str(options, "topic", prefix=_EVENTS_PREFIX)
        self._group_id = require_nonempty_str(options, "group_id", prefix=_EVENTS_PREFIX)
        raw_name = optional_nonempty_str(options, "consumer_name", prefix=_EVENTS_PREFIX)
        self._consumer_name = raw_name or socket.gethostname() or "cicerone"

        self._consumer: Any | None = None
        self._messages: dict[str, Any] = {}
        self._held_offsets: set[tuple[int, int]] = set()
        self._max_offset: dict[int, int] = {}
        self._topic_partition: Any | None = None

    def connect(self) -> None:
        confluent_kafka = import_extra("confluent_kafka", "kafka", 'events.kind = "kafka"')
        conf = {
            **self._conf,
            "group.id": self._group_id,
            "client.id": self._consumer_name,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
        }
        consumer = confluent_kafka.Consumer(conf)
        connected = False
        try:
            consumer.list_topics(topic=self._topic, timeout=self._timeout_seconds)
            connected = True
        except KafkaException as exc:
            raise EventSourceError(f"events.options.bootstrap_servers is unreachable: {exc}") from exc
        finally:
            if not connected:
                _close_quietly(consumer, "Failed to close Kafka consumer after connect error")
        consumer.subscribe([self._topic], on_revoke=self._on_revoke, on_lost=self._on_revoke)

        with self._lock:
            previous = self._detach_unlocked()
            self._consumer = consumer
            self._topic_partition = confluent_kafka.TopicPartition
            self._connected = True
            self._clear_lifecycle()
        if previous is not None:
            self._release(previous)

    def _detach_unlocked(self) -> Any | None:
        consumer = self._consumer
        self._consumer = None
        self._messages.clear()
        self._held_offsets.clear()
        self._max_offset.clear()
        self._topic_partition = None
        return consumer

    def _release(self, consumer: Any) -> None:
        _close_quietly(consumer, "Kafka consumer close failed")

    def _on_revoke(self, consumer: Any, partitions: Sequence[Any]) -> None:
        # Another member now owns these partitions: their deliveries are no longer ours to commit.
        revoked = {int(tp.partition) for tp in partitions}
        with self._lock:
            if consumer is not self._consumer:
                return  # a replaced consumer revokes on close
            dropped = {eid for eid, message in self._messages.items() if int(message.partition()) in revoked}
            for eid in dropped:
                del self._messages[eid]
                self._forget_ids_unlocked(eid)
            self._pending = deque(event for event in self._pending if event.event_id not in dropped)
            self._held_offsets = {
                (part, offset) for part, offset in self._held_offsets if part not in revoked
            }
            for partition in revoked:
                self._max_offset.pop(partition, None)

    def _commit_acks(self, consumer: Any, resolved: Sequence[tuple[str, Any]]) -> Sequence[str]:
        with self._lock:
            done: dict[int, set[int]] = {}
            for _eid, message in resolved:
                partition = int(message.partition())
                done.setdefault(partition, set()).add(int(message.offset()))
            watermarks = {
                partition: self._next_commit_offset(partition, extra_done=offsets)
                for partition, offsets in done.items()
            }
        finished: list[tuple[str, Any]] = []
        try:
            for partition, nxt in watermarks.items():
                if nxt is None:
                    continue
                self._commit_offset(consumer, partition, nxt)
                with self._lock:
                    for eid, message in self._messages.items():
                        if int(message.partition()) == partition and int(message.offset()) < nxt:
                            finished.append((eid, message))
        finally:
            with self._lock:
                for eid, message in finished:
                    self._messages.pop(eid, None)
                    self._release_offset_unlocked(message)
                    self._forget_ids_unlocked(eid)
                for eid, message in resolved:
                    if eid in self._messages:
                        self._release_offset_unlocked(message)
        return tuple(eid for eid, _message in finished)

    def _release_offset_unlocked(self, message: Any) -> None:
        partition = int(message.partition())
        offset = int(message.offset())
        self._held_offsets.discard((partition, offset))
        self._max_offset[partition] = max(self._max_offset.get(partition, -1), offset)

    def health(self) -> EventSourceHealth:
        with self._lock:
            connected = self._connected
            local_held = self._local_held_unlocked()
            last_event_at = self._last_event_at
        if not connected:
            return EventSourceHealth(connected=False, lag=None, last_event_at=last_event_at)
        return EventSourceHealth(
            connected=True,
            lag=local_held if local_held else 0,
            last_event_at=last_event_at,
            detail=f"topic={self._topic} group={self._group_id} consumer={self._consumer_name}",
        )

    def _backend(self) -> Any:
        return self._consumer

    def _delivery_handle(self, event_id: str) -> Any | None:
        return self._messages.get(event_id)

    def _fetch_events(self, consumer: Any, max_events: int) -> list[NormalizedEvent]:
        out: list[NormalizedEvent] = []
        remaining = max_events
        while remaining > 0:
            try:
                message = consumer.poll(0.0)
            except KafkaException:
                logger.exception("Kafka poll failed")
                break
            if message is None:
                break
            incoming = self._message_to_event(consumer, message)
            if incoming is None:
                continue
            out.append(incoming)
            remaining -= 1
        return out

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
        if supplied_event_id(payload) is None:
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

    def _commit_offset(self, consumer: Any, partition: int, nxt: int | None) -> None:
        ctor = self._topic_partition
        if ctor is None:
            raise EventSourceError("KafkaEventSource is not connected")
        if nxt is not None:
            consumer.commit(offsets=[ctor(self._topic, partition, nxt)], asynchronous=False)

    def _commit_discard(self, consumer: Any, message: Any) -> None:
        partition = int(message.partition())
        offset = int(message.offset())
        with self._lock:
            nxt = self._next_commit_offset(partition, extra_done={offset})
        try:
            self._commit_offset(consumer, partition, nxt)
        except KafkaException:
            logger.exception("Failed to commit discarded Kafka message")
