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
    def __init__(self) -> None:
        self.rows: list[str] = []
        self.ids: set[str] = set()
        self.on_execute: list[object] = []
        self.versions: dict[str, int] = {}

    def _touch(self, key: str) -> None:
        self.versions[key] = self.versions.get(key, 0) + 1

    def sadd(self, key: str, member: str) -> int:
        if member in self.ids:
            return 0
        self.ids.add(member)
        self._touch(key)
        return 1

    def rpush(self, key: str, *values: str) -> int:
        self.rows.extend(values)
        self._touch(key)
        return len(self.rows)

    def lrange(self, _key: str, _start: int, _end: int) -> list[str]:
        return list(self.rows)

    def delete(self, key: str) -> int:
        self.rows.clear()
        self._touch(key)
        return 1

    def srem(self, key: str, *members: str) -> int:
        for member in members:
            self.ids.discard(member)
        self._touch(key)
        return len(members)

    def sismember(self, _key: str, member: str) -> bool:
        return member in self.ids

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self)


class _FakePipeline:
    """WATCH/MULTI/EXEC over _FakeRedis; ``on_execute`` runs just before the next EXEC."""

    def __init__(self, client: _FakeRedis) -> None:
        self._client = client
        self._queued: list[tuple[str, tuple[object, ...]]] = []
        self._buffering = False
        self._watched: dict[str, int] = {}

    def __enter__(self) -> _FakePipeline:
        return self

    def __exit__(self, *_exc: object) -> None:
        self._queued.clear()

    def watch(self, *keys: str) -> None:
        self._buffering = False
        self._watched = {key: self._client.versions.get(key, 0) for key in keys}

    def multi(self) -> None:
        self._buffering = True

    def __getattr__(self, name: str):
        target = getattr(self._client, name)

        def call(*args: object):
            if not self._buffering:
                return target(*args)
            self._queued.append((name, args))
            return self

        return call

    def execute(self) -> list[object]:
        hook = self._client.on_execute.pop(0) if self._client.on_execute else None
        queued, self._queued = self._queued, []
        self._buffering = False
        if hook is not None:
            hook()
        watched, self._watched = self._watched, {}
        if any(self._client.versions.get(key, 0) != version for key, version in watched.items()):
            from redis.exceptions import WatchError

            raise WatchError("Watched variable changed.")
        return [getattr(self._client, name)(*args) for name, args in queued]


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
    assert settings.track.redis_url == "redis://localhost:6379/1"
    nested = make_settings(track=TrackSettings(enabled=True, redis_url="redis://localhost:6379/0"))
    assert nested.track.redis_url == "redis://localhost:6379/0"


def test_load_track_rejects_a_blank_redis_url():
    with pytest.raises(ConfigError, match="track.redis_url"):
        load_track_settings({"enabled": True, "redis_url": "  "})


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


def test_redis_delete_retries_when_an_append_races_the_rewrite():
    client = _FakeRedis()
    store = _store(client)
    store.append_rows([_row(), _row(user_id="bob", item_id="ipa-002", event_id="imp-2")])

    def racing_append() -> None:
        client.rpush("cicerone:track", json.dumps(_row(user_id="carol", item_id="ipa-003", event_id="imp-3")))

    client.on_execute.append(racing_append)
    assert store.delete_redis_rows("alice") == 1
    assert client.on_execute == []
    assert [json.loads(row)["user_id"] for row in client.rows] == ["bob", "carol"]


def test_redis_delete_gives_up_after_bounded_retries():
    from redis.exceptions import WatchError

    client = _FakeRedis()
    store = _store(client)
    store.append_rows([_row()])

    def racing_append() -> None:
        client.rpush("cicerone:track", json.dumps(_row(user_id="carol", item_id="ipa-003", event_id="")))

    client.on_execute.extend([racing_append] * 32)
    with pytest.raises(WatchError, match="gave up after 32 attempts"):
        store.delete_redis_rows("alice")
    assert any(json.loads(row)["user_id"] == "alice" for row in client.rows)


def test_redis_append_does_not_strand_an_id_when_the_push_fails():
    from redis.exceptions import ConnectionError as RedisConnectionError

    client = _FakeRedis()
    store = _store(client)

    def drop_connection() -> None:
        raise RedisConnectionError("connection reset")

    client.on_execute.append(drop_connection)
    with pytest.raises(RedisConnectionError):
        store.append_rows([_row()])
    assert client.ids == set()
    assert store.append_rows([_row()]) == 1
    assert [json.loads(row)["event_id"] for row in client.rows] == ["imp-1"]


def test_forget_user_notes_a_redis_failure_after_the_output_erase(tmp_path, monkeypatch):
    from redis.exceptions import ConnectionError as RedisConnectionError

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

    def drop_connection() -> None:
        raise RedisConnectionError("connection reset")

    client.on_execute.append(drop_connection)
    with pytest.raises(RedisConnectionError) as raised:
        forget_user(output, "alice", redis_url="redis://localhost:6379/0")
    assert raised.value.__notes__ == [
        "output rows were erased; run forget-user again to finish the Redis track log"
    ]
    assert pd.read_parquet(tmp_path / "recommendations.parquet").empty
    assert forget_user(output, "alice", redis_url="redis://localhost:6379/0").track == 1
