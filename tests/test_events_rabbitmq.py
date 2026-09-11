from __future__ import annotations

import json
import threading
import time
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
    with pytest.raises(ConfigError, match="timeout_seconds"):
        validate_rabbitmq_event_options(_options(timeout_seconds=0))
    with pytest.raises(ConfigError, match="timeout_seconds"):
        validate_rabbitmq_event_options(_options(timeout_seconds=1e308))


def test_amqp_timeouts_applied_on_connect(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options(timeout_seconds=2.5))
    source.connect()
    params = broker.last_url_params
    assert params is not None
    assert params.socket_timeout == 2.5
    assert params.blocked_connection_timeout == 2.5
    assert params.stack_timeout == 2.5
    assert source._io is not None
    assert source._io._timeout_seconds == 2.5
    source.close()


def test_pika_io_skips_job_queued_during_timed_out_pump():
    from cicerone.events.rabbitmq import _PikaIo

    entered = threading.Event()
    released = threading.Event()
    executed = threading.Event()

    class _Conn:
        def process_data_events(self, time_limit: float | int = 0) -> None:
            del time_limit
            entered.set()
            released.wait(timeout=2)

    io = _PikaIo(timeout_seconds=0.05)
    io._connection = _Conn()
    io.start()
    try:
        assert entered.wait(timeout=2)
        with pytest.raises(TimeoutError, match="timed out"):
            io.submit(executed.set)
        released.set()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and io._thread.is_alive():
            time.sleep(0.01)
        assert io.failed is True
        assert io._thread.is_alive() is False
        assert executed.is_set() is False
    finally:
        released.set()
        io.stop()


def test_pika_io_submit_times_out():
    from cicerone.events.rabbitmq import _PikaIo

    io = _PikaIo(timeout_seconds=0.05)
    io.start()
    try:
        with pytest.raises(TimeoutError, match="timed out"):
            io.submit(lambda: time.sleep(5))
        assert io.failed is True
        started = time.monotonic()
        io.stop()
        assert time.monotonic() - started < 1.0
        with pytest.raises(RuntimeError, match="not running"):
            io.submit(lambda: None)
    finally:
        io.stop()


def test_close_abandons_hung_idle_pump(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options())
    source.connect()
    connection = broker.connection
    channel = connection.channel_obj
    entered = threading.Event()
    released = threading.Event()

    def _hang_pump(time_limit: float | int = 0) -> None:
        del time_limit
        entered.set()
        released.wait(timeout=2)

    connection.process_data_events = _hang_pump  # type: ignore[method-assign]
    assert entered.wait(timeout=2)
    began = time.monotonic()
    source.close()
    assert time.monotonic() - began < 1.0
    released.set()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not (channel.closed and connection.closed):
        time.sleep(0.01)
    assert channel.closed is True
    assert connection.closed is True


def test_pika_io_busy_survives_overlapping_pump():
    from cicerone.events.rabbitmq import _PikaIo

    entered = threading.Event()
    released = threading.Event()
    job_started = threading.Event()
    hold_job = threading.Event()

    class _Conn:
        def process_data_events(self, time_limit: float | int = 0) -> None:
            del time_limit
            entered.set()
            released.wait(timeout=2)

    io = _PikaIo(timeout_seconds=2)
    io._connection = _Conn()
    io.start()
    try:
        assert entered.wait(timeout=2)
        assert io.busy is True

        def _job() -> None:
            job_started.set()
            hold_job.wait(timeout=2)

        waiter = threading.Thread(target=lambda: io.submit(_job))
        waiter.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and io._busy < 2:
            time.sleep(0.01)
        assert io._busy >= 2
        released.set()
        assert job_started.wait(timeout=2)
        assert io.busy is True
        hold_job.set()
        waiter.join(timeout=2)
        io._connection = None
        idle_deadline = time.monotonic() + 2.0
        while time.monotonic() < idle_deadline and io.busy:
            time.sleep(0.01)
        assert io.busy is False
    finally:
        released.set()
        hold_job.set()
        io.stop()


def test_close_abandons_when_submit_overlaps_pump(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options(timeout_seconds=2))
    source.connect()
    connection = broker.connection
    channel = connection.channel_obj
    pump_entered = threading.Event()
    pump_released = threading.Event()
    get_entered = threading.Event()

    def _hang_pump(time_limit: float | int = 0) -> None:
        del time_limit
        pump_entered.set()
        pump_released.wait(timeout=2)

    def _hang_get(*_args: Any, **_kwargs: Any) -> tuple[Any, None, Any]:
        get_entered.set()
        time.sleep(2)
        return None, None, None

    connection.process_data_events = _hang_pump  # type: ignore[method-assign]
    assert pump_entered.wait(timeout=2)
    source._basic_get = _hang_get  # type: ignore[method-assign]
    poller = threading.Thread(target=lambda: list(source.poll(1)))
    poller.start()
    queued = time.monotonic() + 2.0
    io = source._io
    while time.monotonic() < queued:
        io = source._io
        if io is not None and io._busy >= 2:
            break
        time.sleep(0.01)
    assert io is not None and io._busy >= 2
    pump_released.set()
    assert get_entered.wait(timeout=2)
    began = time.monotonic()
    source.close()
    assert time.monotonic() - began < 1.0
    poller.join(timeout=2)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not (channel.closed and connection.closed):
        time.sleep(0.01)
    assert channel.closed is True
    assert connection.closed is True


def test_amqp_callbacks_use_io_handles(monkeypatch):
    from types import SimpleNamespace

    broker = install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options())
    source.connect()
    io = source._io
    assert io is not None
    used: list[str] = []
    io_channel = io._channel
    io_connection = io._connection
    assert io_channel is not None
    assert io_connection is not None

    def _get(queue: str, auto_ack: bool = False) -> tuple[Any, None, Any]:
        del auto_ack
        used.append(f"get:{queue}")
        return None, None, None

    def _ack(delivery_tag: int) -> None:
        used.append(f"ack:{delivery_tag}")

    def _declare(*, queue: str, durable: bool = True, passive: bool = False) -> Any:
        del durable
        used.append(f"declare:{queue}:{passive}")
        return SimpleNamespace(method=SimpleNamespace(message_count=0))

    def _pump(time_limit: float | int = 0) -> None:
        used.append(f"pump:{time_limit}")

    def _fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("callback used source handles")

    io_channel.basic_get = _get  # type: ignore[method-assign]
    io_channel.basic_ack = _ack  # type: ignore[method-assign]
    io_channel.queue_declare = _declare  # type: ignore[method-assign]
    io_connection.process_data_events = _pump  # type: ignore[method-assign]
    source._channel = SimpleNamespace(basic_get=_fail, basic_ack=_fail, queue_declare=_fail)
    source._connection = SimpleNamespace(process_data_events=_fail)

    assert source._basic_get(io) == (None, None, None)
    source._basic_ack(io, 9)
    declared = source._passive_declare(io)
    assert declared.method.message_count == 0
    source._pump_connection(io)
    assert used == [f"get:{source._queue}", "ack:9", f"declare:{source._queue}:True", "pump:0"]
    source.close()


def test_reconnect_does_not_run_old_callback_on_new_channel(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options(timeout_seconds=0.4))
    source.connect()
    old_connection = broker.connection
    old_channel = old_connection.channel_obj
    pump_entered = threading.Event()
    pump_released = threading.Event()
    new_gets: list[int] = []

    def _hang_pump(time_limit: float | int = 0) -> None:
        del time_limit
        pump_entered.set()
        pump_released.wait(timeout=2)

    old_connection.process_data_events = _hang_pump  # type: ignore[method-assign]
    assert pump_entered.wait(timeout=2)
    poller = threading.Thread(target=lambda: list(source.poll(1)))
    poller.start()
    queued = time.monotonic() + 2.0
    old_io = source._io
    while time.monotonic() < queued:
        old_io = source._io
        if old_io is not None and old_io._busy >= 2:
            break
        time.sleep(0.01)
    assert old_io is not None and old_io._busy >= 2
    source.connect()
    new_channel = broker.connection.channel_obj
    assert new_channel is not old_channel
    original_get = new_channel.basic_get

    def _spy_get(queue: str, auto_ack: bool = False) -> tuple[Any, None, Any]:
        new_gets.append(1)
        return original_get(queue, auto_ack=auto_ack)

    new_channel.basic_get = _spy_get  # type: ignore[method-assign]
    pump_released.set()
    poller.join(timeout=2)
    assert new_gets == []
    assert old_channel.closed is True
    source.close()


def test_close_abandons_in_flight_submit(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options(timeout_seconds=2))
    source.connect()
    connection = broker.connection
    channel = connection.channel_obj
    started = threading.Event()

    def _hang(*_args: Any, **_kwargs: Any) -> tuple[Any, None, Any]:
        started.set()
        time.sleep(0.4)
        return None, None, None

    source._basic_get = _hang  # type: ignore[method-assign]
    poller = threading.Thread(target=lambda: list(source.poll(1)))
    poller.start()
    assert started.wait(timeout=2)
    began = time.monotonic()
    source.close()
    assert time.monotonic() - began < 1.0
    poller.join(timeout=2)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not (channel.closed and connection.closed):
        time.sleep(0.01)
    assert channel.closed is True
    assert connection.closed is True


def test_close_after_io_timeout_does_not_block(monkeypatch):
    install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options(timeout_seconds=0.05))
    source.connect()

    def _hang(*_args: Any, **_kwargs: Any) -> tuple[Any, None, Any]:
        time.sleep(5)
        return None, None, None

    source._basic_get = _hang  # type: ignore[method-assign]
    assert list(source.poll(1)) == []
    assert source.health().connected is False
    started = time.monotonic()
    source.close()
    assert time.monotonic() - started < 1.0


def test_close_after_io_timeout_closes_handles_when_call_unwinds(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options(timeout_seconds=0.05))
    source.connect()
    connection = broker.connection
    channel = connection.channel_obj
    done = threading.Event()

    def _hang(*_args: Any, **_kwargs: Any) -> tuple[Any, None, Any]:
        time.sleep(0.2)
        done.set()
        return None, None, None

    source._basic_get = _hang  # type: ignore[method-assign]
    assert list(source.poll(1)) == []
    source.close()
    assert done.wait(timeout=2)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not (channel.closed and connection.closed):
        time.sleep(0.01)
    assert channel.closed is True
    assert connection.closed is True


def test_close_after_timed_out_worker_exits_closes_channel(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options(timeout_seconds=0.05))
    source.connect()
    connection = broker.connection
    channel = connection.channel_obj
    io = source._io
    assert io is not None

    def _hang(*_args: Any, **_kwargs: Any) -> tuple[Any, None, Any]:
        time.sleep(0.15)
        return None, None, None

    source._basic_get = _hang  # type: ignore[method-assign]
    assert list(source.poll(1)) == []
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and io._thread.is_alive():
        time.sleep(0.01)
    assert io._thread.is_alive() is False
    source.close()
    assert channel.closed is True
    assert connection.closed is True


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


def test_health_disconnected_when_probe_times_out(monkeypatch):
    install_fake_rabbitmq(monkeypatch)
    source = RabbitMQEventSource(_options(timeout_seconds=0.05))
    source.connect()

    def _hang(*_args: Any, **_kwargs: Any) -> Any:
        time.sleep(5)
        raise RuntimeError("unreachable")

    source._passive_declare = _hang  # type: ignore[method-assign]
    health = source.health()
    assert health.connected is False
    source.close()


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


def test_connect_timeout_during_open_closes_connection(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    broker.channel_hang_seconds = 0.3
    source = RabbitMQEventSource(_options(timeout_seconds=0.05))
    with pytest.raises(ConfigError, match="unreachable"):
        source.connect()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not getattr(broker, "connection", None):
        time.sleep(0.01)
    connection = getattr(broker, "connection", None)
    assert connection is not None
    while time.monotonic() < deadline and not connection.closed:
        time.sleep(0.01)
    assert connection.closed is True
    assert connection.channel_obj.closed is True


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


def test_reconnect_resets_ack_maps(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    broker.enqueue("cicerone.events", event_payload(event_id="old"))
    source = RabbitMQEventSource(_options())
    source.connect()
    first = list(source.poll(1))
    assert [event.event_id for event in first] == ["old"]
    assert source._held_tags
    source.connect()
    assert source._held_tags == set()
    assert source._delivery_tags == {}
    broker.enqueue("cicerone.events", event_payload(event_id="new"))
    second = list(source.poll(10))
    assert [event.event_id for event in second] == ["new"]
    source.ack([first[0].event_id])
    source.ack([second[0].event_id])
    source.close()
