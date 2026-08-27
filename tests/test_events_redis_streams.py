from __future__ import annotations

import sys
from typing import Any

import pytest
from support.events import event_payload

from cicerone.config import ConfigError
from cicerone.events.redis_streams import RedisStreamsEventSource, validate_redis_stream_options
from cicerone.events.registry import build_event_source, registered_event_source_kinds


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
            if meta["consumer"] == consumername:
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


def test_missing_event_id_uses_stream_entry_id(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    payload = event_payload()
    payload.pop("event_id")
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
    import builtins

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("no redis")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
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
    client.ping = lambda: (_ for _ in ()).throw(RuntimeError("down"))  # type: ignore[method-assign]
    _install_fake_redis(monkeypatch, client)
    source = RedisStreamsEventSource(_options())
    with pytest.raises(ConfigError, match="unreachable"):
        source.connect()


def test_health_tolerates_lag_probe_failures(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()
    client.xadd("cicerone:events", event_payload(event_id="e1"))
    list(source.poll(10))

    def boom(*_a, **_k):
        raise RuntimeError("nope")

    client.xpending = boom  # type: ignore[method-assign]
    client.xinfo_groups = boom  # type: ignore[method-assign]
    health = source.health()
    assert health.connected is True
    assert health.lag == 1


def test_read_failures_return_empty(monkeypatch):
    client = _install_fake_redis(monkeypatch, FakeRedis())
    source = RedisStreamsEventSource(_options())
    source.connect()

    def boom(*_a, **_k):
        raise RuntimeError("nope")

    client.xreadgroup = boom  # type: ignore[method-assign]
    client.xautoclaim = boom  # type: ignore[method-assign]
    assert list(source.poll(10)) == []


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
