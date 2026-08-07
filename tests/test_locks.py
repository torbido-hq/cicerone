"""Unit tests for optional scheduler lock backends."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest
from support.postgres_defaults import resolve_test_database_url

from cicerone.config import ConfigError, IOSettings, make_settings
from cicerone.locks import (
    InProcessLock,
    PostgresAdvisoryLock,
    RedisLock,
    build_lock_backend,
    resolve_postgres_lock_url,
)
from cicerone.trigger import RunGuard

TEST_DATABASE_URL = resolve_test_database_url()
_SKIP_NO_TEST_DB = not TEST_DATABASE_URL
_SKIP_NO_TEST_DB_REASON = (
    "TEST_DATABASE_URL / POSTGRES_TEST_HOST not set — DB-backed tests run against compose Postgres"
)


def test_in_process_lock_is_noop():
    lock = InProcessLock()
    assert lock.acquire() is True
    assert lock.acquire() is True
    lock.release()


def test_build_lock_backend_in_process_default():
    settings = make_settings()
    assert isinstance(build_lock_backend(settings), InProcessLock)


def test_resolve_postgres_lock_url_prefers_explicit():
    settings = make_settings(
        trigger_lock_backend="postgres",
        trigger_postgres_url="postgresql+psycopg://explicit/db",
        output=IOSettings(kind="db", options={"database_url": "postgresql+psycopg://output/db"}),
    )
    assert resolve_postgres_lock_url(settings) == "postgresql+psycopg://explicit/db"


def test_resolve_postgres_lock_url_falls_back_to_output_db():
    settings = make_settings(
        trigger_lock_backend="postgres",
        output=IOSettings(kind="db", options={"database_url": "postgresql+psycopg://output/db"}),
    )
    assert resolve_postgres_lock_url(settings) == "postgresql+psycopg://output/db"


def test_resolve_postgres_lock_url_none_for_dataset_without_explicit():
    settings = make_settings(trigger_lock_backend="postgres")
    assert resolve_postgres_lock_url(settings) is None


def test_build_postgres_lock_without_url_raises():
    settings = make_settings(trigger_lock_backend="postgres")
    with pytest.raises(ConfigError, match="needs a database URL"):
        build_lock_backend(settings)


def test_redis_lock_acquire_release(monkeypatch):
    client = MagicMock()
    client.set.return_value = True
    fake_redis = MagicMock()
    fake_redis.Redis.from_url.return_value = client
    monkeypatch.setitem(__import__("sys").modules, "redis", fake_redis)

    lock = RedisLock("redis://localhost:6379/0")
    assert lock.acquire() is True
    assert lock.acquire() is False
    lock.release()
    client.eval.assert_called_once()
    assert lock.acquire() is True


def test_redis_lock_contention(monkeypatch):
    client = MagicMock()
    client.set.return_value = False
    fake_redis = MagicMock()
    fake_redis.Redis.from_url.return_value = client
    monkeypatch.setitem(__import__("sys").modules, "redis", fake_redis)

    lock = RedisLock("redis://localhost:6379/0")
    assert lock.acquire() is False


def test_redis_lock_missing_package(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("no redis")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ConfigError, match="requirements-redis"):
        RedisLock("redis://localhost:6379/0")


def test_run_guard_respects_distributed_lock_failure():
    class HeldLock:
        def acquire(self) -> bool:
            return False

        def release(self) -> None:
            return None

    calls: list[str] = []
    guard = RunGuard(
        debounce_seconds=0,
        run_fn=lambda triggered_by: calls.append(triggered_by),
        lock_backend=HeldLock(),
    )
    assert guard.trigger("webhook") is False
    assert calls == []


def test_run_guard_releases_backend_after_run():
    events: list[str] = []

    class TrackingLock:
        def acquire(self) -> bool:
            events.append("acquire")
            return True

        def release(self) -> None:
            events.append("release")

    done = threading.Event()

    def fake_run(triggered_by: str) -> None:
        events.append(f"run:{triggered_by}")
        done.set()

    guard = RunGuard(debounce_seconds=0, run_fn=fake_run, lock_backend=TrackingLock())
    assert guard.trigger("cron") is True
    assert done.wait(timeout=5)
    for _ in range(50):
        if events[-1:] == ["release"]:
            break
        time.sleep(0.05)
    assert events == ["acquire", "run:cron", "release"]


def test_build_redis_lock_without_url_raises():
    from dataclasses import replace

    settings = make_settings(trigger_lock_backend="redis", trigger_redis_url="redis://x")
    settings = replace(settings, trigger=replace(settings.trigger, redis_url=None))
    with pytest.raises(ConfigError, match="job.trigger.redis_url"):
        build_lock_backend(settings)


def test_build_redis_lock_backend(monkeypatch):
    client = MagicMock()
    fake_redis = MagicMock()
    fake_redis.Redis.from_url.return_value = client
    monkeypatch.setitem(__import__("sys").modules, "redis", fake_redis)

    settings = make_settings(trigger_lock_backend="redis", trigger_redis_url="redis://localhost:6379/0")
    backend = build_lock_backend(settings)
    assert isinstance(backend, RedisLock)


def test_build_postgres_lock_backend(monkeypatch):
    settings = make_settings(
        trigger_lock_backend="postgres",
        trigger_postgres_url="postgresql+psycopg://u:p@h/db",
    )
    fake_engine = MagicMock()
    monkeypatch.setattr("sqlalchemy.create_engine", lambda *a, **k: fake_engine)
    backend = build_lock_backend(settings)
    assert isinstance(backend, PostgresAdvisoryLock)


def test_build_lock_backend_unknown_raises():
    from dataclasses import replace

    settings = make_settings()
    settings = replace(settings, trigger=replace(settings.trigger, lock_backend="etcd"))
    with pytest.raises(ConfigError, match="must be one of"):
        build_lock_backend(settings)


def test_redis_release_when_not_held(monkeypatch):
    client = MagicMock()
    fake_redis = MagicMock()
    fake_redis.Redis.from_url.return_value = client
    monkeypatch.setitem(__import__("sys").modules, "redis", fake_redis)

    lock = RedisLock("redis://localhost:6379/0")
    lock.release()
    client.eval.assert_not_called()


def test_redis_release_logs_on_failure(monkeypatch):
    client = MagicMock()
    client.set.return_value = True
    client.eval.side_effect = RuntimeError("boom")
    fake_redis = MagicMock()
    fake_redis.Redis.from_url.return_value = client
    monkeypatch.setitem(__import__("sys").modules, "redis", fake_redis)

    lock = RedisLock("redis://localhost:6379/0")
    assert lock.acquire() is True
    lock.release()
    assert lock._held is False


def test_postgres_advisory_lock_mocked(monkeypatch):
    conn = MagicMock()
    conn.execute.return_value.scalar.return_value = True
    engine = MagicMock()
    engine.connect.return_value = conn
    monkeypatch.setattr("sqlalchemy.create_engine", lambda *a, **k: engine)

    lock = PostgresAdvisoryLock("postgresql+psycopg://u:p@h/db")
    assert lock.acquire() is True
    assert lock.acquire() is False
    lock.release()
    conn.close.assert_called()
    assert lock._conn is None
    lock.release()  # no-op when not held


def test_postgres_advisory_lock_contention_mocked(monkeypatch):
    conn = MagicMock()
    conn.execute.return_value.scalar.return_value = False
    engine = MagicMock()
    engine.connect.return_value = conn
    monkeypatch.setattr("sqlalchemy.create_engine", lambda *a, **k: engine)

    lock = PostgresAdvisoryLock("postgresql+psycopg://u:p@h/db")
    assert lock.acquire() is False
    conn.close.assert_called()


def test_postgres_advisory_lock_acquire_error_closes(monkeypatch):
    conn = MagicMock()
    conn.execute.side_effect = RuntimeError("db down")
    engine = MagicMock()
    engine.connect.return_value = conn
    monkeypatch.setattr("sqlalchemy.create_engine", lambda *a, **k: engine)

    lock = PostgresAdvisoryLock("postgresql+psycopg://u:p@h/db")
    with pytest.raises(RuntimeError, match="db down"):
        lock.acquire()
    conn.close.assert_called()


def test_postgres_advisory_lock_release_error(monkeypatch):
    conn = MagicMock()
    conn.execute.return_value.scalar.return_value = True
    engine = MagicMock()
    engine.connect.return_value = conn
    monkeypatch.setattr("sqlalchemy.create_engine", lambda *a, **k: engine)

    lock = PostgresAdvisoryLock("postgresql+psycopg://u:p@h/db")
    assert lock.acquire() is True
    conn.execute.side_effect = RuntimeError("unlock failed")
    lock.release()
    assert lock._conn is None


def test_cron_run_with_lock_skips_when_held(monkeypatch):
    from cicerone import scheduler

    class Held:
        def acquire(self) -> bool:
            return False

        def release(self) -> None:
            raise AssertionError("should not release")

    runs = []
    monkeypatch.setattr(scheduler.job, "run", lambda **kwargs: runs.append(kwargs))
    scheduler._cron_run_with_lock(Held())
    assert runs == []


def test_cron_run_with_lock_runs_and_releases(monkeypatch):
    from cicerone import scheduler

    events: list[str] = []

    class Ok:
        def acquire(self) -> bool:
            events.append("acquire")
            return True

        def release(self) -> None:
            events.append("release")

    monkeypatch.setattr(scheduler.job, "run", lambda **kwargs: events.append(f"run:{kwargs['triggered_by']}"))
    scheduler._cron_run_with_lock(Ok())
    assert events == ["acquire", "run:cron", "release"]
