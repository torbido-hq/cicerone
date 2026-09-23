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
    from cicerone.option_parse import optional_int

    assert optional_int({}, "n", 3, prefix="x", minimum=1) == 3
    with pytest.raises(ConfigError, match="integer"):
        optional_int({"n": "nope"}, "n", 3, prefix="x", minimum=1)
    with pytest.raises(ConfigError, match=">= 1"):
        optional_int({"n": 0}, "n", 3, prefix="x", minimum=1)
    with pytest.raises(ConfigError, match="integer"):
        optional_int({"n": float("inf")}, "n", 3, prefix="x", minimum=1)

    class _OverflowInt:
        def __int__(self) -> int:
            raise OverflowError("too big")

    with pytest.raises(ConfigError, match="integer"):
        optional_int({"n": _OverflowInt()}, "n", 3, prefix="x", minimum=1)


def test_optional_float_validation():
    from cicerone.option_parse import optional_float

    assert optional_float({}, "n", 10.0, prefix="x") == 10.0
    assert optional_float({"n": "2.5"}, "n", 10.0, prefix="x") == 2.5
    with pytest.raises(ConfigError, match="number"):
        optional_float({"n": "nope"}, "n", 10.0, prefix="x")
    with pytest.raises(ConfigError, match="> 0"):
        optional_float({"n": 0}, "n", 10.0, prefix="x")
    with pytest.raises(ConfigError, match="finite"):
        optional_float({"n": float("nan")}, "n", 10.0, prefix="x")
    with pytest.raises(ConfigError, match="finite"):
        optional_float({"n": float("inf")}, "n", 10.0, prefix="x")
    with pytest.raises(ConfigError, match="number"):
        optional_float({"n": 10**400}, "n", 10.0, prefix="x")

    class _OverflowNoRepr:
        def __float__(self) -> float:
            raise OverflowError("too big")

        def __repr__(self) -> str:
            raise RuntimeError("repr failed")

    with pytest.raises(ConfigError, match="must be a number$"):
        optional_float({"n": _OverflowNoRepr()}, "n", 10.0, prefix="x")
    with pytest.raises(ConfigError, match="<= 5"):
        optional_float({"n": 5.1}, "n", 10.0, prefix="x", maximum=5.0)
    assert optional_float({"n": 5.0}, "n", 10.0, prefix="x", maximum=5.0) == 5.0
    with pytest.raises(ConfigError, match=">= 0.01"):
        optional_float({"n": 0.005}, "n", 10.0, prefix="x", minimum=0.01)


def test_kafka_client_timeouts_default_and_override(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    source = KafkaEventSource(_options())
    source.connect()
    assert source._consumer.config["socket.timeout.ms"] == 10000
    assert source._consumer.config["request.timeout.ms"] == 10000
    assert broker.list_topics_timeouts[-1] == 10.0

    other = KafkaEventSource(_options(timeout_seconds=3))
    other.connect()
    assert other._consumer.config["socket.timeout.ms"] == 3000
    assert other._consumer.config["request.timeout.ms"] == 3000
    assert broker.list_topics_timeouts[-1] == 3.0


def test_consumer_poll_interval_options(monkeypatch):
    install_fake_kafka(monkeypatch)
    source = KafkaEventSource(_options(max_poll_interval_ms=600_000, session_timeout_ms=45_000))
    source.connect()
    assert source._consumer.config["max.poll.interval.ms"] == 600_000
    assert source._consumer.config["session.timeout.ms"] == 45_000
    assert source._consumer.config["heartbeat.interval.ms"] == 15_000
    floor = KafkaEventSource(_options(session_timeout_ms=2))
    floor.connect()
    assert floor._consumer.config["session.timeout.ms"] == 2
    assert floor._consumer.config["heartbeat.interval.ms"] == 1
    default = KafkaEventSource(_options())
    default.connect()
    assert "max.poll.interval.ms" not in default._consumer.config
    assert "session.timeout.ms" not in default._consumer.config
    assert "heartbeat.interval.ms" not in default._consumer.config


def test_validate_rejects_bad_poll_interval():
    with pytest.raises(ConfigError, match="max_poll_interval_ms"):
        validate_kafka_event_options(_options(max_poll_interval_ms=0))
    with pytest.raises(ConfigError, match="session_timeout_ms"):
        validate_kafka_event_options(_options(session_timeout_ms="nope"))
    with pytest.raises(ConfigError, match="session_timeout_ms"):
        validate_kafka_event_options(_options(session_timeout_ms=1))
    with pytest.raises(ConfigError, match="max_poll_interval_ms must be >="):
        validate_kafka_event_options(_options(max_poll_interval_ms=1000, session_timeout_ms=2000))
    with pytest.raises(ConfigError, match="session_timeout_ms"):
        validate_kafka_event_options(_options(max_poll_interval_ms=10_000))
    with pytest.raises(ConfigError, match="max_poll_interval_ms"):
        validate_kafka_event_options(_options(session_timeout_ms=400_000))
    from cicerone.kafka_options import (
        MAX_MAX_POLL_INTERVAL_MS,
        MAX_SESSION_TIMEOUT_MS,
        kafka_consumer_config,
    )

    with pytest.raises(ConfigError, match="session_timeout_ms"):
        validate_kafka_event_options(
            _options(
                max_poll_interval_ms=MAX_SESSION_TIMEOUT_MS + 1, session_timeout_ms=MAX_SESSION_TIMEOUT_MS + 1
            )
        )
    with pytest.raises(ConfigError, match="max_poll_interval_ms"):
        validate_kafka_event_options(_options(max_poll_interval_ms=MAX_MAX_POLL_INTERVAL_MS + 1))
    at_session_max = kafka_consumer_config(
        _options(max_poll_interval_ms=MAX_SESSION_TIMEOUT_MS, session_timeout_ms=MAX_SESSION_TIMEOUT_MS),
        prefix="events.options",
    )
    assert at_session_max["session.timeout.ms"] == MAX_SESSION_TIMEOUT_MS
    from cicerone.kafka_options import _heartbeat_interval_ms

    with pytest.raises(ConfigError, match="heartbeat"):
        _heartbeat_interval_ms(1)
    from cicerone.kafka_options import MAX_TIMEOUT_MS

    with pytest.raises(ConfigError, match="max_poll_interval_ms"):
        validate_kafka_event_options(_options(max_poll_interval_ms=MAX_TIMEOUT_MS + 1))
    with pytest.raises(ConfigError, match="max_poll_interval_ms"):
        validate_kafka_event_options(_options(max_poll_interval_ms=600000.9))
    with pytest.raises(ConfigError, match="session_timeout_ms"):
        validate_kafka_event_options(_options(session_timeout_ms=True))
    with pytest.raises(ConfigError, match="max_poll_interval_ms"):
        validate_kafka_event_options(_options(max_poll_interval_ms=float("inf")))
    with pytest.raises(ConfigError, match="max_poll_interval_ms"):
        validate_kafka_event_options(_options(max_poll_interval_ms=1e308))
    conf = kafka_consumer_config(_options(max_poll_interval_ms=600000.0), prefix="events.options")
    assert conf["max.poll.interval.ms"] == 600000


def test_validate_rejects_bad_timeout():
    with pytest.raises(ConfigError, match="timeout_seconds"):
        validate_kafka_event_options(_options(timeout_seconds=0))
    with pytest.raises(ConfigError, match="timeout_seconds"):
        validate_kafka_event_options(_options(timeout_seconds=1e308))
    with pytest.raises(ConfigError, match="timeout_seconds"):
        validate_kafka_event_options(_options(timeout_seconds=0.005))
    from cicerone.kafka_options import (
        MAX_TIMEOUT_MS,
        MIN_TIMEOUT_MS,
        kafka_client_config,
        kafka_timeout_ms,
    )
    from cicerone.option_parse import MAX_BROKER_TIMEOUT_SECONDS

    assert kafka_timeout_ms(_options(timeout_seconds=0.01), prefix="x") == MIN_TIMEOUT_MS
    at_max = _options(timeout_seconds=MAX_BROKER_TIMEOUT_SECONDS)
    assert kafka_timeout_ms(at_max, prefix="x") == MAX_TIMEOUT_MS
    with pytest.raises(ConfigError, match="timeout_seconds"):
        kafka_client_config(_options(timeout_seconds=1e308), prefix="events.options")


def test_kafka_timeout_ms_normalizes_conversion_overflow(monkeypatch):
    from cicerone import kafka_options

    monkeypatch.setattr(kafka_options, "kafka_timeout_seconds", lambda options, *, prefix: 1e308)
    with pytest.raises(ConfigError, match="timeout_seconds"):
        kafka_options.kafka_timeout_ms(_options(), prefix="events.options")
    monkeypatch.setattr(kafka_options, "kafka_timeout_seconds", lambda options, *, prefix: 0.005)
    with pytest.raises(ConfigError, match="timeout_seconds"):
        kafka_options.kafka_timeout_ms(_options(), prefix="events.options")


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
    assert broker.committed == [(0, 1), (0, 2)]


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
    assert (0, 1) in broker.committed


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


def test_ack_does_not_skip_earlier_offset(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    broker.add("cicerone.events", event_payload(event_id="e1"))
    broker.add("cicerone.events", event_payload(event_id="e2"))
    source = KafkaEventSource(_options())
    source.connect()
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["e1", "e2"]
    source.ack([events[1].event_id])
    assert broker.committed == []
    source.ack([events[0].event_id])
    assert broker.committed == [(0, 2)]


def test_ack_keeps_local_state_when_watermark_cannot_advance(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    broker.add("cicerone.events", event_payload(event_id="e1"))
    broker.add("cicerone.events", event_payload(event_id="e2"))
    source = KafkaEventSource(_options())
    source.connect()
    events = list(source.poll(10))
    source.ack([events[1].event_id])
    assert broker.committed == []
    assert events[1].event_id in source._messages
    assert source.nack([events[1]]) == ()
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["e2"]
    source.close()


def test_ack_keeps_later_offset_when_earlier_offset_is_held(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    for index in range(4):
        broker.add("cicerone.events", event_payload(event_id=f"e{index}"))
    source = KafkaEventSource(_options())
    source.connect()
    events = list(source.poll(10))
    source.ack([events[0].event_id, events[1].event_id])
    assert broker.committed == [(0, 2)]
    source.ack([events[3].event_id])
    assert events[3].event_id in source._messages
    assert source.nack([events[3]]) == ()
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["e3"]
    source.close()


def test_ack_keeps_local_state_when_commit_fails(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    broker.add("cicerone.events", event_payload(event_id="e1"))
    source = KafkaEventSource(_options())
    source.connect()
    events = list(source.poll(10))
    broker.commit_error = RuntimeError("commit fail")
    with pytest.raises(RuntimeError, match="commit fail"):
        source.ack([events[0].event_id])
    assert source.nack(events) == ()
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["e1"]
    source.close()


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


def test_reconnect_resets_ack_maps(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    broker.add("cicerone.events", event_payload(event_id="old"))
    source = KafkaEventSource(_options())
    source.connect()
    first = list(source.poll(1))
    assert [event.event_id for event in first] == ["old"]
    assert source._held_offsets
    source.connect()
    assert source._held_offsets == set()
    assert source._messages == {}
    again = list(source.poll(1))
    assert [event.event_id for event in again] == ["old"]
    source.ack([again[0].event_id])
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
