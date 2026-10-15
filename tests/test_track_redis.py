from __future__ import annotations

import json
import sys

import pandas as pd
import pytest
from support.toml_config import write_toml

from cicerone.config import (
    ConfigError,
    IOSettings,
    TrackSettings,
    load_settings,
    load_track_settings,
    make_settings,
)
from cicerone.track.normalize import normalize_track
from cicerone.track.store import TrackStore


class _FakeRedis:
    """Key-aware list/set store. ``fail_next[cmd]`` raises once; ``after[cmd]`` runs once after it."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.sets: dict[str, set[str]] = {}
        self.fail_next: dict[str, BaseException] = {}
        self.after: dict[str, object] = {}

    @property
    def rows(self) -> list[str]:
        return self.lists.setdefault("cicerone:track", [])

    @property
    def ids(self) -> set[str]:
        return self.sets.setdefault("cicerone:track:ids", set())

    def _call(self, name: str) -> None:
        failure = self.fail_next.pop(name, None)
        if failure is not None:
            raise failure

    def _done(self, name: str) -> None:
        hook = self.after.pop(name, None)
        if hook is not None:
            hook()

    def sadd(self, key: str, member: str) -> int:
        self._call("sadd")
        members = self.sets.setdefault(key, set())
        if member in members:
            return 0
        members.add(member)
        return 1

    def rpush(self, key: str, *values: str) -> int:
        self._call("rpush")
        rows = self.lists.setdefault(key, [])
        rows.extend(values)
        return len(rows)

    def lrange(self, key: str, _start: int, _end: int) -> list[str]:
        self._call("lrange")
        rows = list(self.lists.get(key, []))
        self._done("lrange")
        return rows

    def lrem(self, key: str, _count: int, value: str) -> int:
        self._call("lrem")
        rows = self.lists.get(key, [])
        kept = [row for row in rows if row != value]
        self.lists[key] = kept
        return len(rows) - len(kept)

    def srem(self, key: str, *members: str) -> int:
        self._call("srem")
        current = self.sets.setdefault(key, set())
        removed = len(current & set(members))
        current.difference_update(members)
        return removed


def _row(**overrides: object) -> dict[str, object]:
    payload = {
        "kind": "impression",
        "user_id": "alice",
        "item_id": "ipa-001",
        "rank": 1,
        "occurred_at": "2026-08-28T12:00:00Z",
        "event_id": "imp-1",
    }
    payload.update(overrides)
    return normalize_track(payload).as_row()


def _store(client: _FakeRedis) -> TrackStore:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/out"})
    return TrackStore(output, redis_url="redis://localhost:6379/0", redis_client=client)


def test_redis_track_appends_reads_and_forgets_one_user():
    client = _FakeRedis()
    store = _store(client)
    assert store.append_rows([_row(), _row(user_id="bob", item_id="ipa-002", event_id="imp-2")]) == 2
    assert store.append_rows([_row()]) == 0
    assert [row["user_id"] for row in store.read_rows()] == ["alice", "bob"]
    assert store.delete_redis_rows("alice") == 1
    client.rows.append("{")
    assert [row["user_id"] for row in store.read_rows()] == ["bob"]
    assert "imp-1" not in client.ids


def test_load_settings_allows_ha_track_when_redis_url_is_set(tmp_path):
    settings = load_settings(
        write_toml(
            tmp_path,
            """
            [job]
            mode = "serve"
            [job.trigger]
            lock_backend = "redis"
            redis_url = "redis://localhost:6379/0"
            [serve]
            auth_token = "tok"
            [events]
            enabled = true
            kind = "webhook"
            ha = true
            [input]
            kind = "dataset"
            [input.options]
            storage_backend = "local"
            path = "/tmp/in"
            [output]
            kind = "dataset"
            [output.options]
            storage_backend = "s3"
            bucket = "recs"
            access_key_id = "id"
            secret_access_key = "secret"
            [track]
            enabled = true
            redis_url = "redis://localhost:6379/1"
            """,
        )
    )
    assert settings.track.enabled is True
    assert settings.events.ha is True
    assert settings.track.redis_url == "redis://localhost:6379/1"
    nested = make_settings(track=TrackSettings(enabled=True, redis_url="redis://localhost:6379/0"))
    assert nested.track.redis_url == "redis://localhost:6379/0"


def test_load_track_rejects_a_blank_redis_url():
    with pytest.raises(ConfigError, match="track.redis_url"):
        load_track_settings({"enabled": True, "redis_url": "  "})


def test_load_track_redis_key_prefix_defaults_and_rejects_blank():
    assert load_track_settings({"enabled": True}).redis_key_prefix == "cicerone"
    assert load_track_settings({"redis_key_prefix": " shop-eu "}).redis_key_prefix == "shop-eu"
    with pytest.raises(ConfigError, match="track.redis_key_prefix"):
        load_track_settings({"redis_key_prefix": " "})


def test_redis_key_prefix_keeps_deployments_apart():
    client = _FakeRedis()
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/out"})
    eu = TrackStore(output, redis_url="redis://x", redis_client=client, redis_key_prefix="eu")
    us = TrackStore(output, redis_url="redis://x", redis_client=client, redis_key_prefix="us")
    assert eu.append_rows([_row()]) == 1
    assert us.append_rows([_row()]) == 1
    assert us.delete_redis_rows("alice") == 1
    assert [row["user_id"] for row in eu.read_rows()] == ["alice"]
    assert us.read_rows() == []


def test_redis_track_requires_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "redis", None)
    store = TrackStore(
        IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/out"}),
        redis_url="redis://localhost:6379/0",
    )
    with pytest.raises(ConfigError, match="redis package"):
        store.read_rows()


def test_redis_track_reports_an_unreachable_url(monkeypatch):
    class _RedisError(Exception):
        pass

    class _Client:
        def ping(self) -> None:
            raise _RedisError("down")

    fake = type(sys)("redis")
    fake.RedisError = _RedisError
    fake.Redis = type("Redis", (), {"from_url": staticmethod(lambda *_a, **_k: _Client())})
    monkeypatch.setitem(sys.modules, "redis", fake)
    store = TrackStore(
        IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/out"}),
        redis_url="redis://localhost:6379/0",
    )
    with pytest.raises(ConfigError, match="unreachable"):
        store.read_rows()


def test_forget_user_erases_the_redis_track_log_after_the_output(tmp_path, monkeypatch):
    from cicerone.forget import forget_user
    from cicerone.io.dataset_store import DatasetOutputSink

    client = _FakeRedis()
    monkeypatch.setattr("cicerone.track.store_redis._connect", lambda _url: client)
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    DatasetOutputSink(output.options).write_recommendations(
        pd.DataFrame(
            [
                {"user_id": "alice", "item_id": "ipa-001", "rank": 1, "score": 0.5, "source": "popular"},
                {"user_id": "bob", "item_id": "ipa-002", "rank": 1, "score": 0.5, "source": "popular"},
            ]
        )
    )
    store = TrackStore(output, redis_url="redis://localhost:6379/0", redis_client=client)
    store.append_rows([_row(), _row(user_id="bob", item_id="ipa-002", event_id="imp-2")])
    client.rows.append(json.dumps({**_row(event_id="imp-3"), "user_id": " alice "}))

    result = forget_user(output, "alice", redis_url="redis://localhost:6379/0")
    assert result.recommendations == 1
    assert result.track == 2
    assert [json.loads(row)["user_id"] for row in client.rows] == ["bob"]
    assert list(pd.read_parquet(tmp_path / "recommendations.parquet")["user_id"]) == ["bob"]


def test_redis_delete_keeps_an_append_that_races_the_read():
    client = _FakeRedis()
    store = _store(client)
    store.append_rows([_row(), _row(user_id="bob", item_id="ipa-002", event_id="imp-2")])
    carol = _row(user_id="carol", item_id="ipa-003", event_id="imp-3")
    client.after["lrange"] = lambda: client.rpush("cicerone:track", json.dumps(carol, separators=(",", ":")))
    assert store.delete_redis_rows("alice") == 1
    assert [json.loads(row)["user_id"] for row in client.rows] == ["bob", "carol"]
    assert client.ids == {"imp-2"}


def test_redis_delete_failure_never_drops_other_users():
    from cicerone.events.errors import RedisError

    client = _FakeRedis()
    store = _store(client)
    store.append_rows([_row(), _row(user_id="bob", item_id="ipa-002", event_id="imp-2")])
    client.fail_next["lrem"] = RedisError("OOM command not allowed")
    with pytest.raises(RedisError):
        store.delete_redis_rows("alice")
    assert [json.loads(row)["user_id"] for row in client.rows] == ["alice", "bob"]
    # The id was released first, so a rerun only has rows left to remove.
    assert client.ids == {"imp-2"}
    assert store.delete_redis_rows("alice") == 1
    assert [json.loads(row)["user_id"] for row in client.rows] == ["bob"]


def test_redis_delete_counts_every_copy_of_a_line():
    client = _FakeRedis()
    store = _store(client)
    line = json.dumps(_row(event_id=""), separators=(",", ":"))
    client.rpush("cicerone:track", line, line)
    store.append_rows([_row(user_id="bob", item_id="ipa-002", event_id="imp-2")])
    assert store.delete_redis_rows("alice") == 2
    assert [json.loads(row)["user_id"] for row in client.rows] == ["bob"]


def test_redis_append_does_not_strand_an_id_when_the_push_fails():
    from cicerone.events.errors import RedisError

    client = _FakeRedis()
    store = _store(client)
    client.fail_next["rpush"] = RedisError("connection reset")
    with pytest.raises(RedisError):
        store.append_rows([_row()])
    assert client.ids == set()
    assert store.append_rows([_row()]) == 1
    assert [json.loads(row)["event_id"] for row in client.rows] == ["imp-1"]


def test_redis_append_notes_an_id_it_could_not_release():
    from cicerone.events.errors import RedisError

    client = _FakeRedis()
    store = _store(client)
    client.fail_next["rpush"] = RedisError("connection reset")
    client.fail_next["srem"] = RedisError("connection reset")
    with pytest.raises(RedisError) as raised:
        store.append_rows([_row()])
    assert raised.value.__notes__ == ["event id 'imp-1' stays claimed; a retry will skip it"]


def test_forget_user_notes_a_redis_failure_after_the_output_erase(tmp_path, monkeypatch):
    from cicerone.events.errors import RedisError as RedisConnectionError
    from cicerone.forget import forget_user
    from cicerone.io.dataset_store import DatasetOutputSink

    client = _FakeRedis()
    monkeypatch.setattr("cicerone.track.store_redis._connect", lambda _url: client)
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    DatasetOutputSink(output.options).write_recommendations(
        pd.DataFrame(
            [{"user_id": "alice", "item_id": "ipa-001", "rank": 1, "score": 0.5, "source": "popular"}]
        )
    )
    _store(client).append_rows([_row()])

    client.fail_next["lrem"] = RedisConnectionError("connection reset")
    with pytest.raises(RedisConnectionError) as raised:
        forget_user(output, "alice", redis_url="redis://localhost:6379/0")
    assert raised.value.__notes__ == [
        "output rows were erased; run forget-user again to finish the Redis track log"
    ]
    assert pd.read_parquet(tmp_path / "recommendations.parquet").empty
    assert forget_user(output, "alice", redis_url="redis://localhost:6379/0").track == 1


def test_job_treats_a_redis_track_read_failure_as_optional(caplog):
    from cicerone.events.errors import RedisError
    from cicerone.job_eval import try_load

    def boom() -> list[dict[str, object]]:
        raise RedisError("Timeout reading from socket")

    with caplog.at_level("ERROR", logger="cicerone.job_eval"):
        assert try_load("read track rows", boom, None) is None
    assert any("Failed to read track rows (RedisError" in record.getMessage() for record in caplog.records)
