"""In-memory pika stand-in for tests."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


class FakeRabbitMessage:
    def __init__(self, body: bytes | str | dict[str, Any], delivery_tag: int) -> None:
        self.body = body
        self.delivery_tag = delivery_tag


class FakeChannel:
    def __init__(self, broker: FakeRabbitBroker) -> None:
        self.broker = broker
        self.prefetch: int | None = None
        self.acked: list[int] = []
        self.nacked: list[tuple[int, bool]] = []
        self.closed = False
        self._unacked: dict[int, FakeRabbitMessage] = {}

    def basic_qos(self, prefetch_count: int = 0) -> None:
        self.prefetch = prefetch_count

    def queue_declare(self, queue: str, durable: bool = True, passive: bool = False) -> SimpleNamespace:
        del durable
        if self.broker.queue_declare_error is not None:
            raise self.broker.queue_declare_error
        if not passive:
            self.broker.queues.setdefault(queue, [])
        ready = len(self.broker.queues.get(queue, []))
        return SimpleNamespace(method=SimpleNamespace(message_count=ready))

    def basic_get(self, queue: str, auto_ack: bool = False) -> tuple[Any, None, Any]:
        del auto_ack
        messages = self.broker.queues.get(queue, [])
        if not messages:
            return None, None, None
        message = messages.pop(0)
        self._unacked[message.delivery_tag] = message
        method = SimpleNamespace(delivery_tag=message.delivery_tag)
        return method, None, message.body

    def basic_ack(self, delivery_tag: int) -> None:
        self.acked.append(delivery_tag)
        self._unacked.pop(delivery_tag, None)

    def basic_nack(self, delivery_tag: int, requeue: bool = True) -> None:
        self.nacked.append((delivery_tag, requeue))
        held = self._unacked.pop(delivery_tag, None)
        if requeue and held is not None:
            queue = next(iter(self.broker.queues), "default")
            self.broker.queues.setdefault(queue, []).insert(0, held)

    def basic_publish(
        self,
        exchange: str,
        routing_key: str,
        body: bytes,
        properties: object | None = None,
    ) -> None:
        del properties
        self.broker.published.append((exchange, routing_key, body))
        target = routing_key or next(iter(self.broker.queues), "default")
        self.broker.enqueue(target, body)

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, broker: FakeRabbitBroker) -> None:
        self.broker = broker
        self.channel_obj = FakeChannel(broker)
        self.closed = False
        self.heartbeats = 0
        self.process_error: Exception | None = None

    def channel(self) -> FakeChannel:
        return self.channel_obj

    def process_data_events(self, time_limit: float | int = 0) -> None:
        del time_limit
        if self.process_error is not None:
            raise self.process_error
        self.heartbeats += 1

    def close(self) -> None:
        self.closed = True


class FakeRabbitBroker:
    def __init__(self) -> None:
        self.queues: dict[str, list[FakeRabbitMessage]] = {}
        self.published: list[tuple[str, str, bytes]] = []
        self.connect_error: Exception | None = None
        self.queue_declare_error: Exception | None = None
        self._tag = 0

    def enqueue(self, queue: str, body: bytes | str | dict[str, Any]) -> FakeRabbitMessage:
        self._tag += 1
        message = FakeRabbitMessage(body, self._tag)
        self.queues.setdefault(queue, []).append(message)
        return message


def install_fake_rabbitmq(
    monkeypatch: pytest.MonkeyPatch, broker: FakeRabbitBroker | None = None
) -> FakeRabbitBroker:
    broker = broker or FakeRabbitBroker()
    module = ModuleType("pika")

    def _params(url: str) -> str:
        return url

    def _connection(_params: str) -> FakeConnection:
        if broker.connect_error is not None:
            raise broker.connect_error
        connection = FakeConnection(broker)
        broker.connection = connection  # type: ignore[attr-defined]
        return connection

    module.URLParameters = _params  # type: ignore[attr-defined]
    module.BlockingConnection = _connection  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pika", module)
    return broker
