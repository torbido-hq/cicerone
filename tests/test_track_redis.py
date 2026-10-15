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
from cicerone.track.store_redis import default_redis_key_prefix


class _FakeRedis:
    """Key-aware hash store.

    ``fail_next[cmd]`` raises before the command runs; ``lose_reply[cmd]`` applies it, then raises;
    ``after[cmd]`` runs once after the command.
    """

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.fail_next: dict[str, BaseException] = {}
        self.lose_reply: dict[str, BaseException] = {}
        self.after: dict[str, object] = {}

    @property
    def rows(self) -> list[str]:
        return [value for fields in self.hashes.values() for value in fields.values()]

    def _call(self, name: str) -> None:
        failure = self.fail_next.pop(name, None)
        if failure is not None:
            raise failure

    def _done(self, name: str) -> None:
        lost = self.lose_reply.pop(name, None)
        if lost is not None:
            raise lost
        hook = self.after.pop(name, None)
        if hook is not None:
            hook()

    def hsetnx(self, key: str, field: str, value: str) -> int:
        self._call("hsetnx")
        fields = self.hashes.setdefault(key, {})
        stored = 0
        if field not in fields:
            fields[field] = value
            stored = 1
        self._done("hsetnx")
        return stored

    def pipeline(self, *, transaction: bool = True) -> _FakePipeline:
        assert transaction
        return _FakePipeline(self)

    def hvals(self, key: str) -> list[str]:
        self._call("hvals")
        return list(self.hashes.get(key, {}).values())

    def hgetall(self, key: str) -> dict[str, str]:
        self._call("hgetall")
        snapshot = dict(self.hashes.get(key, {}))
        self._done("hgetall")
        return snapshot

    def hdel(self, key: str, *fields: str) -> int:
        self._call("hdel")
        current = self.hashes.get(key, {})
        removed = sum(1 for field in fields if current.pop(field, None) is not None)
        self._done("hdel")
        return removed


class _FakePipeline:
    """MULTI/EXEC: queued commands apply together at ``execute``."""

    def __init__(self, client: _FakeRedis) -> None:
        self._client = client
        self._queued: list[tuple[str, tuple[object, ...]]] = []

    def __enter__(self) -> _FakePipeline:
        return self

    def __exit__(self, *_exc: object) -> None:
        self._queued.clear()

    def hsetnx(self, *args: object) -> _FakePipeline:
        self._queued.append(("hsetnx", args))
        return self

    def execute(self) -> list[object]:
        self._client._call("execute")
        results = [getattr(self._client, name)(*args) for name, args in self._queued]
        self._queued.clear()
        self._client._done("execute")
        return results


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


_OUTPUT = IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/out"})


def _store(client: _FakeRedis, output: IOSettings = _OUTPUT) -> TrackStore:
    return TrackStore(output, redis_url="redis://localhost:6379/0", redis_client=client)


def _log(client: _FakeRedis) -> dict[str, str]:
    (fields,) = client.hashes.values()
    return fields


def test_redis_track_appends_reads_and_forgets_one_user():
    client = _FakeRedis()
    store = _store(client)
    assert store.append_rows([_row(), _row(user_id="bob", item_id="ipa-002", event_id="imp-2")]) == 2
    assert store.append_rows([_row()]) == 0
    assert [row["user_id"] for row in store.read_rows()] == ["alice", "bob"]
    assert store.delete_redis_rows("alice") == 1
    _log(client)["junk"] = "{"
    assert [row["user_id"] for row in store.read_rows()] == ["bob"]
    assert "imp-1" not in _log(client)


def test_redis_append_is_idempotent_when_the_reply_is_lost():
    from cicerone.events.errors import RedisError

    client = _FakeRedis()
    store = _store(client)
    client.lose_reply["execute"] = RedisError("Timeout reading from socket")
    with pytest.raises(RedisError):
        store.append_rows([_row()])
    # The row was stored before the reply was lost; the retry neither duplicates nor drops it.
    assert store.append_rows([_row()]) == 0
    assert [json.loads(row)["event_id"] for row in client.rows] == ["imp-1"]


def test_redis_append_failure_before_the_write_lets_a_retry_store_the_row():
    from cicerone.events.errors import RedisError

    client = _FakeRedis()
    store = _store(client)
    client.fail_next["execute"] = RedisError("connection reset")
    with pytest.raises(RedisError):
        store.append_rows([_row()])
    assert client.rows == []
    assert store.append_rows([_row()]) == 1


def test_redis_append_stores_a_batch_whole_or_not_at_all():
    from cicerone.events.errors import RedisError

    client = _FakeRedis()
    store = _store(client)
    batch = [_row(), _row(user_id="bob", item_id="ipa-002", event_id="imp-2")]
    client.fail_next["execute"] = RedisError("connection reset")
    with pytest.raises(RedisError):
        store.append_rows(batch)
    assert client.rows == []
    assert store.append_rows(batch) == 2
    assert store.append_rows([*batch, _row(user_id="carol", item_id="ipa-003", event_id="imp-3")]) == 1


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


def test_load_track_redis_key_prefix_is_optional_and_rejects_blank():
    assert load_track_settings({"enabled": True}).redis_key_prefix is None
    assert load_track_settings({"redis_key_prefix": " shop-eu "}).redis_key_prefix == "shop-eu"
    with pytest.raises(ConfigError, match="track.redis_key_prefix"):
        load_track_settings({"redis_key_prefix": " "})


def test_default_redis_key_prefix_follows_the_output_location():
    other = IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/elsewhere"})
    db = IOSettings(kind="db", options={"database_url": "postgresql://app:old@db/recs"})
    rotated = IOSettings(kind="db", options={"database_url": "postgresql://app:new@db/recs"})
    assert default_redis_key_prefix(_OUTPUT) == default_redis_key_prefix(_OUTPUT)
    assert default_redis_key_prefix(_OUTPUT) != default_redis_key_prefix(other)
    assert default_redis_key_prefix(db) == default_redis_key_prefix(rotated)
    other_table = IOSettings(
        kind="db", options={"database_url": "postgresql://app:old@db/recs", "track_table": "t2"}
    )
    assert default_redis_key_prefix(db) != default_redis_key_prefix(other_table)
    s3 = {"storage_backend": "s3", "bucket": "recs", "access_key_id": "a", "secret_access_key": "x"}
    minio = {**s3, "endpoint_url": "http://minio:9000"}
    rotated_s3 = {**s3, "access_key_id": "b", "secret_access_key": "y"}
    prefix = default_redis_key_prefix(IOSettings(kind="dataset", options=s3))
    assert prefix != default_redis_key_prefix(IOSettings(kind="dataset", options=minio))
    assert prefix == default_redis_key_prefix(IOSettings(kind="dataset", options=rotated_s3))
    odd = IOSettings(kind="db", options={"database_url": "not a url://user:pw@host"})
    assert default_redis_key_prefix(odd).startswith("cicerone:")
    assert default_redis_key_prefix(_OUTPUT).startswith("cicerone:")


def test_deployments_sharing_redis_keep_their_logs_apart():
    client = _FakeRedis()
    other = IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/elsewhere"})
    eu, us = _store(client), _store(client, other)
    named = TrackStore(_OUTPUT, redis_url="redis://x", redis_client=client, redis_key_prefix="named")
    for store in (eu, us, named):
        assert store.append_rows([_row()]) == 1
    assert us.delete_redis_rows("alice") == 1
    assert [row["user_id"] for row in eu.read_rows()] == ["alice"]
    assert [row["user_id"] for row in named.read_rows()] == ["alice"]
    assert us.read_rows() == []
    assert "named:track" in client.hashes


def test_redis_track_requires_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "redis", None)
    store = TrackStore(_OUTPUT, redis_url="redis://localhost:6379/0")
    with pytest.raises(ConfigError, match="redis package"):
        store.read_rows()


def test_redis_track_keeps_the_redis_error_type_when_unreachable(monkeypatch):
    class _RedisError(Exception):
        pass

    class _Client:
        def hvals(self, _key: str) -> list[str]:
            raise _RedisError("Connection refused")

    fake = type(sys)("redis")
    fake.RedisError = _RedisError
    fake.Redis = type("Redis", (), {"from_url": staticmethod(lambda *_a, **_k: _Client())})
    monkeypatch.setitem(sys.modules, "redis", fake)
    store = TrackStore(_OUTPUT, redis_url="redis://localhost:6379/0")
    with pytest.raises(_RedisError, match="Connection refused"):
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
    store = _store(client, output)
    store.append_rows([_row(), _row(user_id="bob", item_id="ipa-002", event_id="imp-2")])
    _log(client)["imp-3"] = json.dumps({**_row(event_id="imp-3"), "user_id": " alice "})

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
    client.after["hgetall"] = lambda: store.append_rows([carol])
    assert store.delete_redis_rows("alice") == 1
    assert sorted(_log(client)) == ["imp-2", "imp-3"]


def test_redis_delete_failure_never_drops_other_users():
    from cicerone.events.errors import RedisError

    client = _FakeRedis()
    store = _store(client)
    store.append_rows([_row(), _row(user_id="bob", item_id="ipa-002", event_id="imp-2")])
    client.fail_next["hdel"] = RedisError("OOM command not allowed")
    with pytest.raises(RedisError):
        store.delete_redis_rows("alice")
    assert sorted(_log(client)) == ["imp-1", "imp-2"]
    assert store.delete_redis_rows("alice") == 1
    assert sorted(_log(client)) == ["imp-2"]


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
    _store(client, output).append_rows([_row()])

    client.fail_next["hdel"] = RedisConnectionError("connection reset")
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
