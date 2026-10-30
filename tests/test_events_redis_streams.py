from __future__ import annotations

import inspect
import sys
from typing import Any

import pytest
from support.events import dataset_updater, event_payload
from support.fakes import hide_module

from cicerone.config import ConfigError
from cicerone.events.base import EventSourceError
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.errors import RedisError
from cicerone.events.redis_streams import RedisStreamsEventSource, validate_redis_stream_options
from cicerone.events.registry import build_event_source, registered_event_source_kinds
from cicerone.events.worker import EventWorker
from cicerone.feature_config import FeatureConfig


class _ResponseError(Exception):
    pass


class FakeRedis:
    """Minimal in-memory Redis Streams stand-in for tests."""

    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.groups: dict[tuple[str, str], dict[str, Any]] = {}
        self._seq = 0
        self.closed = False
        self.ResponseError = _ResponseError

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True

    def xadd(self, name: str, fields: dict[str, Any], id: str = "*") -> str:  # noqa: A002
        self._seq += 1
        entry_id = f"{self._seq}-0" if id == "*" else id
        self.streams.setdefault(name, []).append((entry_id, {str(k): str(v) for k, v in fields.items()}))
        return entry_id

    def xgroup_create(self, name: str, groupname: str, id: str = "0-0", mkstream: bool = False) -> bool:
        if mkstream:
            self.streams.setdefault(name, [])
        key = (name, groupname)
        if key in self.groups:
            raise _ResponseError("BUSYGROUP Consumer Group name already exists")
        self.groups[key] = {
            "last_delivered": id,
            "consumers": {},
            "pending": {},  # entry_id -> {consumer, fields, idle_ms}
        }
        return True

    def xreadgroup(
        self,
        groupname: str,
        consumername: str,
        streams: dict[str, str],
        count: int | None = None,
        block: int | None = None,
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        del block
        out: list[tuple[str, list[tuple[str, dict[str, str]]]]] = []
        for stream_name, cursor in streams.items():
            group = self.groups[(stream_name, groupname)]
            messages: list[tuple[str, dict[str, str]]] = []
            if cursor != ">":
                continue
            for entry_id, fields in self.streams.get(stream_name, []):
                if entry_id in group["pending"]:
                    continue
                if not self._id_gt(entry_id, group["last_delivered"]):
                    continue
                group["pending"][entry_id] = {
                    "consumer": consumername,
                    "fields": fields,
                    "idle_ms": 0,
                }
                group["last_delivered"] = entry_id
                messages.append((entry_id, dict(fields)))
                if count is not None and len(messages) >= count:
                    break
            if messages:
                out.append((stream_name, messages))
        return out

    def xack(self, name: str, groupname: str, *ids: str) -> int:
        group = self.groups[(name, groupname)]
        acked = 0
        for entry_id in ids:
            if group["pending"].pop(entry_id, None) is not None:
                acked += 1
        return acked

    def xclaim(
        self,
        name: str,
        groupname: str,
        consumername: str,
        min_idle_time: int,
        message_ids: list[str],
    ) -> list[tuple[str, dict[str, str]]]:
        group = self.groups[(name, groupname)]
        claimed: list[tuple[str, dict[str, str]]] = []
        for entry_id in message_ids:
            meta = group["pending"].get(entry_id)
            if meta is None or meta["idle_ms"] < min_idle_time:
                continue
            meta["consumer"] = consumername
            meta["idle_ms"] = 0
            claimed.append((entry_id, dict(meta["fields"])))
        return claimed

    def xpending(self, name: str, groupname: str) -> dict[str, int]:
        group = self.groups[(name, groupname)]
        return {"pending": len(group["pending"]), "min": None, "max": None, "consumers": []}

    def xinfo_groups(self, name: str) -> list[dict[str, Any]]:
        result = []
        for (stream_name, group_name), group in self.groups.items():
            if stream_name != name:
                continue
            pending = set(group["pending"])
            lag = 0
            for entry_id, _ in self.streams.get(name, []):
                if entry_id in pending:
                    continue
                if not self._id_gt(entry_id, group["last_delivered"]):
                    continue
                lag += 1
            result.append({"name": group_name, "lag": lag, "pending": len(pending)})
        return result

    def xautoclaim(
        self,
        name: str,
        groupname: str,
        consumername: str,
        min_idle_time: int,
        start_id: str = "0-0",
        count: int | None = None,
    ) -> tuple[str, list[tuple[str, dict[str, str]]]]:
        group = self.groups[(name, groupname)]
        claimed: list[tuple[str, dict[str, str]]] = []
        next_id = "0-0"
        for entry_id, meta in list(group["pending"].items()):
            if not self._id_gte(entry_id, start_id):
                continue
            if meta["idle_ms"] < min_idle_time:
                continue
            meta["consumer"] = consumername
            meta["idle_ms"] = 0
            claimed.append((entry_id, dict(meta["fields"])))
            next_id = entry_id
            if count is not None and len(claimed) >= count:
                break
        return next_id, claimed

    def age_pending(self, name: str, groupname: str, entry_id: str, idle_ms: int) -> None:
        self.groups[(name, groupname)]["pending"][entry_id]["idle_ms"] = idle_ms

    @staticmethod
    def _id_gt(left: str, right: str) -> bool:
        return FakeRedis._parse_id(left) > FakeRedis._parse_id(right)

    @staticmethod
    def _id_gte(left: str, right: str) -> bool:
        return FakeRedis._parse_id(left) >= FakeRedis._parse_id(right)

    @staticmethod
    def _parse_id(value: str) -> tuple[int, int]:
        if value in {"0-0", "0", ">"}:
            return (0, 0)
        major, minor = value.split("-", 1)
        return int(major), int(minor)


def _install_fake_redis(monkeypatch: pytest.MonkeyPatch, client: FakeRedis) -> FakeRedis:
    class _Redis:
        ResponseError = _ResponseError

        @staticmethod
        def from_url(*_args: Any, **_kwargs: Any) -> FakeRedis:
            return client

    fake_mod = type(sys)("redis")
    fake_mod.Redis = _Redis
    fake_mod.ResponseError = _ResponseError
    monkeypatch.setitem(sys.modules, "redis", fake_mod)
    return client


def _options(**extra: Any) -> dict[str, Any]:
    return {
        "redis_url": "redis://localhost:6379/0",
        "stream": "cicerone:events",
        "consumer_group": "cicerone",
        "consumer_name": "test-consumer",
        "claim_idle_ms": 1000,
        **extra,
    }


def test_redis_streams_registered():
    assert "redis_streams" in registered_event_source_kinds()
    source = build_event_source("redis_streams", _options())
    assert isinstance(source, RedisStreamsEventSource)


def test_validate_requires_core_options():
    with pytest.raises(ConfigError, match="redis_url"):
        validate_redis_stream_options({"stream": "s", "consumer_group": "g"})
    with pytest.raises(ConfigError, match="stream"):
        validate_redis_stream_options({"redis_url": "redis://x", "consumer_group": "g"})
    with pytest.raises(ConfigError, match="consumer_group"):
        validate_redis_stream_options({"redis_url": "redis://x", "stream": "s"})


def test_validate_rejects_bad_tuning():
    with pytest.raises(ConfigError, match="block_ms"):
        validate_redis_stream_options(_options(block_ms=-1))
    with pytest.raises(ConfigError, match="claim_idle_ms"):
        validate_redis_stream_options(_options(claim_idle_ms=0))
    with pytest.raises(ConfigError, match="consumer_name"):
        validate_redis_stream_options(_options(consumer_name="  "))
    with pytest.raises(ConfigError, match="block_ms must be an integer, got inf"):
        validate_redis_stream_options(_options(block_ms=float("inf")))
    with pytest.raises(ConfigError, match="claim_idle_ms must be an integer, got 'soon'"):
        validate_redis_stream_options(_options(claim_idle_ms="soon"))


def test_poll_ack_and_health(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    client.xadd("cicerone:events", event_payload(event_id="e1", item_id="i1"))
    client.xadd("cicerone:events", event_payload(event_id="e2", item_id="i2"))

    first = list(source.poll(1))
    assert [event.event_id for event in first] == ["e1"]
    assert source.health().connected is True
    assert source.health().lag is not None and source.health().lag >= 1
    source.ack([first[0].event_id])

    second = list(source.poll(10))
    assert [event.event_id for event in second] == ["e2"]
    source.ack([second[0].event_id])
    assert list(source.poll(10)) == []
    assert client.xpending("cicerone:events", "cicerone")["pending"] == 0


def test_nack_allows_repoll(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    client.xadd("cicerone:events", event_payload(event_id="e1"))
    first = list(source.poll(10))
    assert len(first) == 1
    source.nack(first)
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["e1"]
    source.ack([again[0].event_id])
    assert list(source.poll(10)) == []


@pytest.mark.parametrize("event_id", [None, 0, ""], ids=["absent", "zero", "empty"])
def test_missing_event_id_uses_stream_entry_id(monkeypatch, event_id):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    payload = event_payload()
    payload.pop("event_id")
    if event_id is not None:
        payload["event_id"] = event_id
    entry_id = client.xadd("cicerone:events", payload)
    events = list(source.poll(10))
    assert len(events) == 1
    assert events[0].event_id == entry_id


def test_poison_entry_is_acked(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    client.xadd("cicerone:events", {"user_id": "u1"})  # missing required fields
    client.xadd("cicerone:events", event_payload(event_id="ok"))
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["ok"]
    assert client.xpending("cicerone:events", "cicerone")["pending"] == 1


def test_autoclaim_idle_from_other_consumer(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options(claim_idle_ms=50))
    source.connect()
    entry_id = client.xadd("cicerone:events", event_payload(event_id="abandoned"))
    # Deliver to another consumer, then age it.
    other = client.xreadgroup("cicerone", "dead-consumer", streams={"cicerone:events": ">"}, count=1)
    assert other
    client.age_pending("cicerone:events", "cicerone", entry_id, idle_ms=100)
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["abandoned"]
    source.ack([events[0].event_id])


def test_restarted_consumer_reclaims_its_own_unacked_entry(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    first = RedisStreamsEventSource(_options(claim_idle_ms=50))
    first.connect()
    entry_id = client.xadd("cicerone:events", event_payload(event_id="unacked"))
    assert [event.event_id for event in first.poll(10)] == ["unacked"]
    first.close()
    restarted = RedisStreamsEventSource(_options(claim_idle_ms=50))
    restarted.connect()
    client.age_pending("cicerone:events", "cicerone", entry_id, idle_ms=100)
    assert [event.event_id for event in restarted.poll(10)] == ["unacked"]


def test_busygroup_is_ok(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    source.close()
    again = RedisStreamsEventSource(_options())
    again.connect()  # group already exists
    assert again.health().connected is True
    again.close()
    assert client.closed is True


def test_missing_redis_package(monkeypatch):
    hide_module(monkeypatch, "redis")
    source = RedisStreamsEventSource(_options())
    with pytest.raises(ConfigError, match=r"cicerone-recommender\[redis\]"):
        source.connect()


def test_poll_before_connect_raises():
    source = RedisStreamsEventSource(_options())
    with pytest.raises(RuntimeError, match="not connected"):
        source.poll(1)


def test_poll_zero_and_empty_ack_nack(monkeypatch):
    _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    assert list(source.poll(0)) == []
    source.ack([])
    source.nack([])
    assert source.health().connected is True
    source.close()
    assert source.health().connected is False


def test_connect_ping_failure(monkeypatch):
    client = FakeRedis()
    client.ping = lambda: (_ for _ in ()).throw(RedisError("down"))  # type: ignore[method-assign]
    _install_fake_redis(monkeypatch, client)
    source = RedisStreamsEventSource(_options())
    with pytest.raises(EventSourceError, match="unreachable"):
        source.connect()


def test_connect_ping_unexpected_error_propagates(monkeypatch):
    client = FakeRedis()
    client.ping = lambda: (_ for _ in ()).throw(RuntimeError("bug"))  # type: ignore[method-assign]
    _install_fake_redis(monkeypatch, client)
    source = RedisStreamsEventSource(_options())
    with pytest.raises(RuntimeError, match="bug"):
        source.connect()


def test_health_tolerates_lag_probe_failures(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    client.xadd("cicerone:events", event_payload(event_id="e1"))
    list(source.poll(10))

    def boom(*_a, **_k):
        raise RedisError("nope")

    client.xpending = boom  # type: ignore[method-assign]
    client.xinfo_groups = boom  # type: ignore[method-assign]
    health = source.health()
    assert health.connected is True
    assert health.lag == 1


@pytest.mark.parametrize("method", ["xpending", "xinfo_groups"])
def test_health_propagates_unexpected_probe_failures(monkeypatch, method):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()

    def boom(*_a, **_k):
        raise RuntimeError("bug")

    setattr(client, method, boom)
    with pytest.raises(RuntimeError, match="bug"):
        source.health()


def test_read_failures_return_empty(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()

    def boom(*_a, **_k):
        raise RedisError("nope")

    client.xreadgroup = boom  # type: ignore[method-assign]
    client.xautoclaim = boom  # type: ignore[method-assign]
    assert list(source.poll(10)) == []


@pytest.mark.parametrize("method", ["xreadgroup", "xautoclaim"])
def test_read_propagates_unexpected_failures(monkeypatch, method):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()

    def boom(*_a, **_k):
        raise RuntimeError("bug")

    setattr(client, method, boom)
    with pytest.raises(RuntimeError, match="bug"):
        source.poll(10)


def test_failed_ack_still_allows_nack(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    client.xadd("cicerone:events", event_payload(event_id="e1"))
    events = list(source.poll(10))
    assert len(events) == 1

    def boom(*_a, **_k):
        raise RuntimeError("xack down")

    client.xack = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="xack down"):
        source.ack([events[0].event_id])
    source.nack(events)
    client.xack = FakeRedis.xack.__get__(client, FakeRedis)  # type: ignore[method-assign]
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["e1"]
    source.ack([again[0].event_id])
    assert client.xpending("cicerone:events", "cicerone")["pending"] == 0


def test_heartbeat_resets_pending_idle(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    client.xadd("cicerone:events", event_payload(event_id="e1"))
    events = list(source.poll(10))
    pending = client.groups[("cicerone:events", "cicerone")]["pending"]
    entry_id = next(iter(pending))
    pending[entry_id]["idle_ms"] = 60_000
    source.heartbeat(events)
    assert pending[entry_id]["idle_ms"] == 0
    source.ack([events[0].event_id])


def test_heartbeat_reraises_when_xclaim_fails(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    client.xadd("cicerone:events", event_payload(event_id="e1"))
    events = list(source.poll(10))

    def _boom(*_args, **_kwargs):
        raise RedisError("xclaim down")

    client.xclaim = _boom  # type: ignore[method-assign]
    with pytest.raises(RedisError, match="xclaim down"):
        source.heartbeat(events)


def test_heartbeat_propagates_unexpected_xclaim_failure(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    client.xadd("cicerone:events", event_payload(event_id="e1"))
    events = list(source.poll(10))

    def _boom(*_args, **_kwargs):
        raise RuntimeError("bug")

    client.xclaim = _boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="bug"):
        source.heartbeat(events)


def test_close_only_absorbs_redis_errors(monkeypatch, caplog):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    client.close = lambda: (_ for _ in ()).throw(RedisError("close down"))  # type: ignore[method-assign]
    source.close()
    assert "Redis Streams client close failed" in caplog.text

    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    client.close = lambda: (_ for _ in ()).throw(RuntimeError("bug"))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="bug"):
        source.close()


def test_reconnect_only_absorbs_redis_close_errors(monkeypatch):
    import cicerone.redis_client as redis_client

    first = FakeRedis()
    second = FakeRedis()
    _install_fake_redis(monkeypatch, first)
    clients = iter((first, second))
    monkeypatch.setattr(redis_client, "redis_from_url", lambda *_args, **_kwargs: next(clients))
    source = RedisStreamsEventSource(_options())
    source.connect()
    first.close = lambda: (_ for _ in ()).throw(RedisError("close down"))  # type: ignore[method-assign]
    source.connect()

    third = FakeRedis()
    monkeypatch.setattr(redis_client, "redis_from_url", lambda *_args, **_kwargs: third)
    second.close = lambda: (_ for _ in ()).throw(RuntimeError("bug"))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="bug"):
        source.connect()


def test_discard_xack_only_absorbs_redis_errors(monkeypatch, caplog):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    client.xack = lambda *_args, **_kwargs: (_ for _ in ()).throw(RedisError("xack down"))  # type: ignore[method-assign]
    client.xadd("cicerone:events", {"user_id": "u1"})
    assert list(source.poll(10)) == []
    assert "Failed to XACK discarded Redis Streams entries" in caplog.text

    client.xack = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bug"))  # type: ignore[method-assign]
    client.xadd("cicerone:events", {"user_id": "u2"})
    with pytest.raises(RuntimeError, match="bug"):
        source.poll(10)


def test_repeated_nack_does_not_duplicate(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    client.xadd("cicerone:events", event_payload(event_id="e1"))
    events = list(source.poll(10))
    source.nack(events)
    source.nack(events)
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["e1"]


def test_whitespace_required_options_rejected():
    with pytest.raises(ConfigError, match="redis_url"):
        validate_redis_stream_options({"redis_url": "  ", "stream": "s", "consumer_group": "g"})


@pytest.mark.parametrize(
    ("options", "expected"),
    [({}, None), ({"block_ms": None}, None), ({"block_ms": 0}, 0), ({"block_ms": 250}, 250)],
)
def test_read_sends_block_only_when_configured(monkeypatch, options, expected):
    # Redis treats BLOCK 0 as "wait forever"; unset must not send it.
    client = _install_fake_redis(monkeypatch, FakeRedis())
    seen: list[int | None] = []
    real_read = client.xreadgroup

    def _recording_read(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(inspect.signature(real_read).bind(*args, **kwargs).arguments.get("block"))
        return real_read(*args, **kwargs)

    client.xreadgroup = _recording_read  # type: ignore[method-assign]
    source = RedisStreamsEventSource(_options(**options))
    source.connect()
    assert list(source.poll(1)) == []
    assert seen == [expected]


def test_explicit_block_zero_warns(caplog):
    with caplog.at_level("WARNING", logger="cicerone.events.redis_streams"):
        RedisStreamsEventSource(_options(block_ms=0))
    assert "block_ms = 0" in caplog.text


def test_worker_retries_failed_xack_until_acked(monkeypatch, tmp_path, feature_config: FeatureConfig):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    _settings, _sink, updater = dataset_updater(tmp_path, feature_config)
    source = RedisStreamsEventSource(_options())
    worker = EventWorker(source, MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0), updater)
    source.connect()
    client.xadd("cicerone:events", event_payload(event_id="e1"))
    real_xack = client.xack

    def _fail_xack(*_args: Any) -> int:
        raise RedisError("xack down")

    client.xack = _fail_xack  # type: ignore[method-assign]
    with pytest.raises(RedisError, match="xack down"):
        worker.tick()
    client.xack = real_xack  # type: ignore[method-assign]
    assert worker.tick() == 0
    assert client.groups[("cicerone:events", "cicerone")]["pending"] == {}
