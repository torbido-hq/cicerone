from __future__ import annotations

import json
from typing import Any

import pytest
from support.events import event_payload
from support.fake_rabbitmq import install_fake_rabbitmq

from cicerone.config import ConfigError
from cicerone.events.ha import ingest_is_fanout, poll_without_apply_lock
from cicerone.events.rabbitmq import RabbitMQEventSource, validate_rabbitmq_event_options
from cicerone.events.registry import build_event_source, registered_event_source_kinds


def _options(**extra: Any) -> dict[str, Any]:
    return {
        "amqp_url": "amqp://guest:guest@localhost:5672/",
        "queue": "cicerone.events",
        **extra,
    }


def test_rabbitmq_registered():
    assert "rabbitmq" in registered_event_source_kinds()
    source = build_event_source("rabbitmq", _options())
    assert isinstance(source, RabbitMQEventSource)
    assert ingest_is_fanout("rabbitmq") is True
    assert poll_without_apply_lock("rabbitmq") is True


def test_validate_requires_core_options():
    with pytest.raises(ConfigError, match="amqp_url"):
        validate_rabbitmq_event_options({"queue": "q"})
    with pytest.raises(ConfigError, match="queue"):
        validate_rabbitmq_event_options({"amqp_url": "amqp://localhost/"})
    with pytest.raises(ConfigError, match="prefetch"):
        validate_rabbitmq_event_options(_options(prefetch=0))


def test_poll_ack_and_health(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    broker.enqueue("cicerone.events", event_payload(event_id="e1", item_id="i1"))
    broker.enqueue("cicerone.events", event_payload(event_id="e2", item_id="i2"))
    source = RabbitMQEventSource(_options())
    source.connect()
    first = list(source.poll(1))
    assert [event.event_id for event in first] == ["e1"]
    assert source.health().connected is True
    source.ack([first[0].event_id])
    second = list(source.poll(10))
    assert [event.event_id for event in second] == ["e2"]
    source.ack([second[0].event_id])
    assert list(source.poll(10)) == []
    channel = broker.connection.channel_obj
    assert channel.acked == [1, 2]


def test_nack_allows_repoll(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    broker.enqueue("cicerone.events", event_payload(event_id="e1"))
    source = RabbitMQEventSource(_options())
    source.connect()
    first = list(source.poll(10))
    assert len(first) == 1
    source.nack(first)
    source.nack(first)
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["e1"]
    source.ack(["missing", again[0].event_id])
    assert list(source.poll(10)) == []
    assert broker.connection.channel_obj.nacked == []


def test_ack_forgets_succeeded_tags_when_later_ack_fails(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    broker.enqueue("cicerone.events", event_payload(event_id="e1"))
    broker.enqueue("cicerone.events", event_payload(event_id="e2"))
    source = RabbitMQEventSource(_options())
    source.connect()
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["e1", "e2"]
    original = broker.connection.channel_obj.basic_ack

    def _ack(*, delivery_tag: int) -> None:
        if delivery_tag == 2:
            raise RuntimeError("ack 2")
        original(delivery_tag=delivery_tag)

    broker.connection.channel_obj.basic_ack = _ack  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="ack 2"):
        source.ack([event.event_id for event in events])
    source.nack(events)
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["e2"]
    source.close()


def test_missing_event_id_uses_delivery_tag(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    payload = event_payload()
    payload.pop("event_id")
    broker.enqueue("cicerone.events", payload)
    source = RabbitMQEventSource(_options())
    source.connect()
    events = list(source.poll(10))
    assert len(events) == 1
    assert events[0].event_id == "1"


def test_poison_entry_is_acked(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    broker.enqueue("cicerone.events", {"user_id": "u1"})
    broker.enqueue("cicerone.events", event_payload(event_id="ok"))
    source = RabbitMQEventSource(_options())
    source.connect()
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["ok"]
    assert 1 in broker.connection.channel_obj.acked


def test_ack_discard_tolerates_ack_failure(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    broker.enqueue("cicerone.events", b"not-json")
    broker.enqueue("cicerone.events", event_payload(event_id="ok"))
    source = RabbitMQEventSource(_options())
    source.connect()

    def _boom(**_kwargs):
        raise RuntimeError("ack fail")

    broker.connection.channel_obj.basic_ack = _boom  # type: ignore[method-assign]
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["ok"]


def test_bytes_json_payload(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    broker.enqueue("cicerone.events", json.dumps(event_payload(event_id="e1")).encode())
    source = RabbitMQEventSource(_options())
    source.connect()
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["e1"]


def test_heartbeat_pumps_connection(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    broker.enqueue("cicerone.events", event_payload(event_id="e1"))
    source = RabbitMQEventSource(_options())
    source.connect()
    events = list(source.poll(10))
    source.heartbeat(events)
    assert broker.connection.heartbeats == 1
    source.ack([events[0].event_id])


def test_heartbeat_logs_process_failure(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options())
    source.connect()
    broker.connection.process_error = RuntimeError("hb")
    source.heartbeat([])


def test_heartbeat_when_disconnected(monkeypatch):
    install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options())
    source.heartbeat([])
    source.connect()
    source.close()
    source.heartbeat([])


def test_missing_pika_package(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "pika":
            raise ImportError("no pika")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    source = RabbitMQEventSource(_options())
    with pytest.raises(ConfigError, match=r"cicerone-recommender\[rabbitmq\]"):
        source.connect()


def test_poll_before_connect_raises():
    source = RabbitMQEventSource(_options())
    with pytest.raises(RuntimeError, match="not connected"):
        source.poll(1)


def test_poll_zero_and_close(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options(prefetch=10))
    source.connect()
    assert broker.connection.channel_obj.prefetch == 10
    assert list(source.poll(0)) == []
    source.ack([])
    source.nack([])
    source.close()
    assert source.health().connected is False


def test_duplicate_event_id_is_acked(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    broker.enqueue("cicerone.events", event_payload(event_id="e1"))
    source = RabbitMQEventSource(_options())
    source.connect()
    first = list(source.poll(1))
    broker.enqueue("cicerone.events", event_payload(event_id="e1", item_id="other"))
    again = list(source.poll(10))
    assert first[0].event_id == "e1"
    assert again == []
    source.ack([first[0].event_id])


def test_health_tolerates_queue_probe_failure(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options())
    source.connect()

    def _boom(**_kwargs):
        raise RuntimeError("no queue")

    broker.connection.channel_obj.queue_declare = _boom  # type: ignore[method-assign]
    health = source.health()
    assert health.connected is True


def test_basic_get_failure_returns_partial(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    broker.enqueue("cicerone.events", event_payload(event_id="e1"))
    source = RabbitMQEventSource(_options())
    source.connect()
    first = list(source.poll(1))
    source.nack(first)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("get fail")

    broker.connection.channel_obj.basic_get = _boom  # type: ignore[method-assign]
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["e1"]


def test_connect_failure(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    broker.connect_error = RuntimeError("down")
    source = RabbitMQEventSource(_options())
    with pytest.raises(ConfigError, match="unreachable"):
        source.connect()


def test_connect_closes_connection_when_declare_fails(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    broker.queue_declare_error = RuntimeError("no queue")
    source = RabbitMQEventSource(_options())
    with pytest.raises(ConfigError, match="unreachable"):
        source.connect()
    assert broker.connection.closed is True


def test_heartbeat_runs_on_io_thread(monkeypatch):
    import threading

    broker = install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options())
    source.connect()
    ids: list[int] = []
    original_get = broker.connection.channel_obj.basic_get
    original_pump = broker.connection.process_data_events

    def _get(*args, **kwargs):
        ids.append(threading.get_ident())
        return original_get(*args, **kwargs)

    def _pump(*args, **kwargs):
        ids.append(threading.get_ident())
        return original_pump(*args, **kwargs)

    broker.connection.channel_obj.basic_get = _get  # type: ignore[method-assign]
    broker.connection.process_data_events = _pump  # type: ignore[method-assign]
    source.poll(1)
    done = threading.Event()

    def _beat() -> None:
        source.heartbeat([])
        done.set()

    threading.Thread(target=_beat, name="test-heartbeat").start()
    assert done.wait(2.0)
    io_ident = source._io._thread.ident if source._io is not None else None
    source.close()
    assert io_ident is not None
    assert ids
    assert all(ident == io_ident for ident in ids)


def test_reconnect_closes_previous(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options())
    source.connect()
    first = broker.connection
    source.connect()
    assert first.closed is True
    source.close()
