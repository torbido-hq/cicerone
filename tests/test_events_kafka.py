from __future__ import annotations

import json
from typing import Any

import pytest
from support.events import event_payload
from support.fake_kafka import install_fake_kafka

from cicerone.config import ConfigError
from cicerone.events.ha import ingest_is_fanout
from cicerone.events.kafka import KafkaEventSource, validate_kafka_event_options
from cicerone.events.registry import build_event_source, registered_event_source_kinds


def _options(**extra: Any) -> dict[str, Any]:
    return {
        "bootstrap_servers": "localhost:9092",
        "topic": "cicerone.events",
        "group_id": "cicerone",
        "consumer_name": "test-consumer",
        **extra,
    }


def test_kafka_registered():
    assert "kafka" in registered_event_source_kinds()
    source = build_event_source("kafka", _options())
    assert isinstance(source, KafkaEventSource)
    assert ingest_is_fanout("kafka") is True


def test_validate_requires_core_options():
    with pytest.raises(ConfigError, match="bootstrap_servers"):
        validate_kafka_event_options({"topic": "t", "group_id": "g"})
    with pytest.raises(ConfigError, match="topic"):
        validate_kafka_event_options({"bootstrap_servers": "h:9092", "group_id": "g"})
    with pytest.raises(ConfigError, match="group_id"):
        validate_kafka_event_options({"bootstrap_servers": "h:9092", "topic": "t"})
    with pytest.raises(ConfigError, match="consumer_name"):
        validate_kafka_event_options(_options(consumer_name="  "))
    with pytest.raises(ConfigError, match="security_protocol"):
        validate_kafka_event_options(_options(security_protocol="nope"))


def test_optional_int_validation():
    from cicerone.kafka_options import optional_int

    assert optional_int({}, "n", 3, prefix="x", minimum=1) == 3
    with pytest.raises(ConfigError, match="integer"):
        optional_int({"n": "nope"}, "n", 3, prefix="x", minimum=1)
    with pytest.raises(ConfigError, match=">= 1"):
        optional_int({"n": 0}, "n", 3, prefix="x", minimum=1)


def test_poll_ack_and_health(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    broker.add("cicerone.events", event_payload(event_id="e1", item_id="i1"))
    broker.add("cicerone.events", event_payload(event_id="e2", item_id="i2"))
    source = KafkaEventSource(_options())
    source.connect()
    first = list(source.poll(1))
    assert [event.event_id for event in first] == ["e1"]
    assert source.health().connected is True
    assert source.health().lag is not None and source.health().lag >= 0
    source.ack([first[0].event_id])
    second = list(source.poll(10))
    assert [event.event_id for event in second] == ["e2"]
    source.ack([second[0].event_id])
    assert list(source.poll(10)) == []
    assert broker.committed == [(0, 0), (0, 1)]


def test_nack_allows_repoll(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    broker.add("cicerone.events", event_payload(event_id="e1"))
    source = KafkaEventSource(_options())
    source.connect()
    first = list(source.poll(10))
    assert len(first) == 1
    source.nack(first)
    source.nack(first)
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["e1"]
    source.ack(["missing", again[0].event_id])
    assert list(source.poll(10)) == []


def test_missing_event_id_uses_partition_offset(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    payload = event_payload()
    payload.pop("event_id")
    broker.add("cicerone.events", payload)
    source = KafkaEventSource(_options())
    source.connect()
    events = list(source.poll(10))
    assert len(events) == 1
    assert events[0].event_id == "0-0"


def test_poison_entry_is_committed(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    broker.add("cicerone.events", {"user_id": "u1"})
    broker.add("cicerone.events", event_payload(event_id="ok"))
    source = KafkaEventSource(_options())
    source.connect()
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["ok"]
    assert (0, 0) in broker.committed


def test_bytes_json_payload(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    broker.add("cicerone.events", json.dumps(event_payload(event_id="e1")).encode())
    source = KafkaEventSource(_options())
    source.connect()
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["e1"]


def test_message_error_is_skipped(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    broker.add("cicerone.events", event_payload(event_id="bad"), error="boom")
    broker.add("cicerone.events", event_payload(event_id="ok"))
    source = KafkaEventSource(_options())
    source.connect()
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["ok"]


def test_duplicate_event_id_is_committed(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    broker.add("cicerone.events", event_payload(event_id="e1"))
    source = KafkaEventSource(_options())
    source.connect()
    first = list(source.poll(1))
    broker.add("cicerone.events", event_payload(event_id="e1", item_id="other"))
    again = list(source.poll(10))
    assert first[0].event_id == "e1"
    assert again == []
    source.ack([first[0].event_id])


def test_sasl_options_in_consumer_config(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    source = KafkaEventSource(
        _options(
            security_protocol="SASL_SSL",
            sasl_mechanism="PLAIN",
            sasl_username="user",
            sasl_password="pass",
        )
    )
    source.connect()
    consumer = source._consumer
    assert consumer.config["security.protocol"] == "SASL_SSL"
    assert consumer.config["sasl.mechanisms"] == "PLAIN"
    assert consumer.config["sasl.username"] == "user"
    assert broker.topics == {}


def test_missing_confluent_kafka_package(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "confluent_kafka":
            raise ImportError("no kafka")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    source = KafkaEventSource(_options())
    with pytest.raises(ConfigError, match=r"cicerone-recommender\[kafka\]"):
        source.connect()


def test_poll_before_connect_raises():
    source = KafkaEventSource(_options())
    with pytest.raises(RuntimeError, match="not connected"):
        source.poll(1)


def test_poll_zero_and_empty_ack_nack(monkeypatch):
    install_fake_kafka(monkeypatch)
    source = KafkaEventSource(_options())
    source.connect()
    assert list(source.poll(0)) == []
    source.ack([])
    source.nack([])
    source.heartbeat([])
    assert source.health().connected is True
    source.close()
    assert source.health().connected is False


def test_invalid_json_is_committed(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    broker.add("cicerone.events", b"not-json")
    broker.add("cicerone.events", event_payload(event_id="ok"))
    source = KafkaEventSource(_options())
    source.connect()
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["ok"]


def test_reconnect_closes_previous(monkeypatch):
    install_fake_kafka(monkeypatch)
    source = KafkaEventSource(_options())
    source.connect()
    first = source._consumer
    source.connect()
    assert first.closed is True
    source.close()


def test_poll_exception_returns_partial(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    broker.add("cicerone.events", event_payload(event_id="e1"))
    source = KafkaEventSource(_options())
    source.connect()
    first = list(source.poll(1))
    source.nack(first)

    def _boom(_timeout):
        raise RuntimeError("poll fail")

    source._consumer.poll = _boom  # type: ignore[method-assign]
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["e1"]


def test_connect_list_topics_failure(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    broker.list_topics_error = RuntimeError("down")
    source = KafkaEventSource(_options())
    with pytest.raises(ConfigError, match="unreachable"):
        source.connect()


def test_commit_discard_tolerates_commit_failure(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    broker.add("cicerone.events", b"not-json")
    broker.add("cicerone.events", event_payload(event_id="ok"))
    source = KafkaEventSource(_options())
    source.connect()

    def _boom(**_kwargs):
        raise RuntimeError("commit fail")

    source._consumer.commit = _boom  # type: ignore[method-assign]
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["ok"]
