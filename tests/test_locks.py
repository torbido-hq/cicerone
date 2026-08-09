"""Unit tests for optional scheduler lock backends."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from cicerone.config import (
    POSTGRES_LOCK_URL_REQUIRED,
    ConfigError,
    IOSettings,
    make_settings,
    resolve_postgres_lock_url,
)
from cicerone.locks import (
    PostgresAdvisoryLock,
    RedisLock,
    advisory_keys_from_lock_key,
    build_lock_backend,
)
from cicerone.trigger import RunGuard


def _mock_redis_module(monkeypatch, *, client: MagicMock | None = None) -> MagicMock:
    client = client or MagicMock()
    release_script = MagicMock(name="release_script")
    refresh_script = MagicMock(name="refresh_script")
    refresh_script.return_value = 1
    client.register_script.side_effect = [release_script, refresh_script]
    client.release_script = release_script
    client.refresh_script = refresh_script
    fake_redis = MagicMock()
    fake_redis.Redis.from_url.return_value = client
    monkeypatch.setitem(__import__("sys").modules, "redis", fake_redis)
    return client


def test_build_lock_backend_rejects_in_process():
    settings = make_settings()
    with pytest.raises(AssertionError, match="only for distributed backends"):
        build_lock_backend(settings)


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
    with pytest.raises(ConfigError, match="needs a database URL") as exc_info:
        build_lock_backend(settings)
    assert str(exc_info.value) == POSTGRES_LOCK_URL_REQUIRED


def test_advisory_keys_stable_for_default_and_differ_by_lock_key():
    from cicerone.locks import PG_ADVISORY_KEY1, PG_ADVISORY_KEY2

    assert advisory_keys_from_lock_key("cicerone:scheduler:run_guard") == (
        PG_ADVISORY_KEY1,
        PG_ADVISORY_KEY2,
    )
    assert advisory_keys_from_lock_key("other-job") != (
        PG_ADVISORY_KEY1,
        PG_ADVISORY_KEY2,
    )


def test_build_redis_lock_uses_configured_key_and_ttl(monkeypatch):
    client = _mock_redis_module(monkeypatch)
    client.set.return_value = True

    settings = make_settings(
        trigger_lock_backend="redis",
        trigger_redis_url="redis://localhost:6379/0",
        trigger_lock_key="my-job:run",
        trigger_lock_ttl_seconds=120,
    )
    lock = build_lock_backend(settings)
    assert isinstance(lock, RedisLock)
    assert lock.acquire() is True
    client.set.assert_called_with("my-job:run", lock._token, nx=True, px=120_000)
    assert client.register_script.call_count == 2
    lock.release()


def test_build_postgres_lock_uses_configured_lock_key(monkeypatch):
    settings = make_settings(
        trigger_lock_backend="postgres",
        trigger_postgres_url="postgresql+psycopg://u:p@h/db",
        trigger_lock_key="job-a",
    )
    fake_engine = MagicMock()
    monkeypatch.setattr("sqlalchemy.create_engine", lambda *a, **k: fake_engine)
    backend = build_lock_backend(settings)
    assert isinstance(backend, PostgresAdvisoryLock)
    assert (backend._key1, backend._key2) == advisory_keys_from_lock_key("job-a")


def test_redis_lock_acquire_release(monkeypatch):
    client = _mock_redis_module(monkeypatch)
    client.set.return_value = True

    lock = RedisLock("redis://localhost:6379/0")
    assert lock.acquire() is True
    assert lock.acquire() is False
    lock.release()
    client.release_script.assert_called_once_with(keys=[lock._key], args=[lock._token])
    assert lock.acquire() is True
    lock.release()


def test_redis_lock_contention(monkeypatch):
    client = _mock_redis_module(monkeypatch)
    client.set.return_value = False

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


def test_run_guard_without_backend_uses_local_exclusion_only():
    done = threading.Event()
    calls: list[str] = []

    def fake_run(triggered_by: str) -> None:
        calls.append(triggered_by)
        done.set()

    guard = RunGuard(debounce_seconds=60, run_fn=fake_run)
    assert guard._backend is None
    assert guard.trigger("webhook") is True
    assert done.wait(timeout=5)
    assert guard.trigger("webhook") is False
    assert calls == ["webhook"]


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


def test_run_guard_debounce_with_distributed_lock_backend():
    class AlwaysAcquiringLock:
        def __init__(self) -> None:
            self.acquires = 0

        def acquire(self) -> bool:
            self.acquires += 1
            return True

        def release(self) -> None:
            return None

    calls: list[str] = []
    lock_backend = AlwaysAcquiringLock()
    done = threading.Event()

    def fake_run(triggered_by: str) -> None:
        calls.append(triggered_by)
        done.set()

    guard = RunGuard(
        debounce_seconds=60,
        run_fn=fake_run,
        lock_backend=lock_backend,
    )
    assert guard.trigger("first") is True
    assert done.wait(timeout=5)
    assert guard.trigger("second") is False
    assert calls == ["first"]
    assert lock_backend.acquires == 1


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
    _mock_redis_module(monkeypatch)
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


def test_build_lock_backend_unknown_is_programming_error():
    from dataclasses import replace

    settings = make_settings()
    settings = replace(settings, trigger=replace(settings.trigger, lock_backend="etcd"))
    with pytest.raises(AssertionError, match="only for distributed backends"):
        build_lock_backend(settings)


def test_redis_release_when_not_held(monkeypatch):
    client = _mock_redis_module(monkeypatch)

    lock = RedisLock("redis://localhost:6379/0")
    lock.release()
    client.release_script.assert_not_called()


def test_redis_release_logs_on_failure(monkeypatch):
    client = _mock_redis_module(monkeypatch)
    client.set.return_value = True
    client.release_script.side_effect = RuntimeError("boom")

    lock = RedisLock("redis://localhost:6379/0")
    assert lock.acquire() is True
    lock.release()
    assert lock._held is False


def test_redis_lock_refreshes_ttl_while_held(monkeypatch):
    client = _mock_redis_module(monkeypatch)
    client.set.return_value = True

    lock = RedisLock(
        "redis://localhost:6379/0",
        ttl_ms=200,
        refresh_interval_ms=20,
    )
    assert lock.acquire() is True
    deadline = time.time() + 2.0
    while time.time() < deadline and client.refresh_script.call_count < 1:
        time.sleep(0.02)
    lock.release()
    assert client.refresh_script.call_count >= 1
    client.refresh_script.assert_called_with(keys=[lock._key], args=[lock._token, 200])


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
    lock.release()


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
