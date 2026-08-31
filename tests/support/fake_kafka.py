"""In-memory confluent-kafka stand-in for tests."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest


class FakeKafkaMessage:
    def __init__(
        self,
        value: bytes | str | dict[str, Any] | None,
        *,
        topic: str = "cicerone.events",
        partition: int = 0,
        offset: int = 0,
        key: bytes | None = None,
        error: object | None = None,
    ) -> None:
        self._value = value
        self._topic = topic
        self._partition = partition
        self._offset = offset
        self._key = key
        self._error = error

    def error(self) -> object | None:
        return self._error

    def value(self) -> bytes | str | dict[str, Any] | None:
        return self._value

    def key(self) -> bytes | None:
        return self._key

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset

    def topic(self) -> str:
        return self._topic


class FakeTopicPartition:
    def __init__(self, topic: str, partition: int, offset: int) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset


class FakeKafkaBroker:
    def __init__(self) -> None:
        self.topics: dict[str, list[FakeKafkaMessage]] = {}
        self.committed: list[tuple[int, int]] = []
        self.produced: list[tuple[str, bytes | None, bytes | str | None]] = []
        self.list_topics_error: Exception | None = None
        self._seq = 0

    def add(
        self,
        topic: str,
        value: bytes | str | dict[str, Any],
        *,
        error: object | None = None,
    ) -> FakeKafkaMessage:
        message = FakeKafkaMessage(value, topic=topic, offset=self._seq, error=error)
        self._seq += 1
        self.topics.setdefault(topic, []).append(message)
        return message

    def produce(self, topic: str, value: bytes | str | None, key: bytes | None = None) -> None:
        self.produced.append((topic, key, value))
        if value is not None:
            self.add(topic, value)


class FakeConsumer:
    def __init__(self, broker: FakeKafkaBroker, config: dict[str, Any]) -> None:
        self.broker = broker
        self.config = config
        self._topics: list[str] = []
        self._index: dict[str, int] = {}
        self.closed = False

    def subscribe(self, topics: list[str]) -> None:
        self._topics = list(topics)

    def poll(self, timeout: float) -> FakeKafkaMessage | None:
        del timeout
        for topic in self._topics:
            messages = self.broker.topics.get(topic, [])
            cursor = self._index.get(topic, 0)
            if cursor < len(messages):
                self._index[topic] = cursor + 1
                return messages[cursor]
        return None

    def commit(
        self,
        message: FakeKafkaMessage | None = None,
        offsets: list[Any] | None = None,
        asynchronous: bool = False,
    ) -> None:
        del asynchronous
        if offsets:
            for tp in offsets:
                self.broker.committed.append((int(tp.partition), int(tp.offset)))
            return
        if message is not None:
            self.broker.committed.append((message.partition(), message.offset()))

    def list_topics(self, topic: str | None = None, timeout: float | None = None) -> None:
        del topic, timeout
        if self.broker.list_topics_error is not None:
            raise self.broker.list_topics_error

    def close(self) -> None:
        self.closed = True


class FakeProducer:
    def __init__(self, broker: FakeKafkaBroker, config: dict[str, Any]) -> None:
        self.broker = broker
        self.config = config
        self.flush_remaining = 0

    def produce(
        self,
        topic: str,
        value: bytes | str | None = None,
        key: bytes | None = None,
        **_kwargs: Any,
    ) -> None:
        self.broker.produce(topic, value, key)

    def flush(self, timeout: float | None = None) -> int:
        del timeout
        return self.flush_remaining

    def list_topics(self, timeout: float | None = None) -> None:
        del timeout
        if self.broker.list_topics_error is not None:
            raise self.broker.list_topics_error


def install_fake_kafka(
    monkeypatch: pytest.MonkeyPatch, broker: FakeKafkaBroker | None = None
) -> FakeKafkaBroker:
    broker = broker or FakeKafkaBroker()
    module = ModuleType("confluent_kafka")

    def _consumer(config: dict[str, Any]) -> FakeConsumer:
        return FakeConsumer(broker, config)

    def _producer(config: dict[str, Any]) -> FakeProducer:
        return FakeProducer(broker, config)

    module.Consumer = _consumer  # type: ignore[attr-defined]
    module.Producer = _producer  # type: ignore[attr-defined]
    module.TopicPartition = FakeTopicPartition  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "confluent_kafka", module)
    return broker
