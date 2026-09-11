"""Optional distributed lock backends for multi-replica schedulers.

Default single-instance exclusion is RunGuard's threading.Lock (no backend).
``postgres`` / ``redis`` are opt-in; clients are imported only when selected.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

from cicerone.config.constants import (
    DEFAULT_DATASET_APPEND_LOCK_TTL_SECONDS,
    DEFAULT_LOCK_ACQUIRE_TIMEOUT_SECONDS,
)
from cicerone.config.lock_url import resolve_postgres_lock_url
from cicerone.config.settings import Settings
from cicerone.locks.keys import (
    PG_ADVISORY_KEY1,
    PG_ADVISORY_KEY2,
    REDIS_LOCK_KEY,
    REDIS_LOCK_TTL_MS,
    advisory_keys_from_lock_key,
    dataset_append_lock_key,
    events_apply_lock_key,
)
from cicerone.locks.postgres import PostgresAdvisoryLock
from cicerone.locks.redis import RedisLock

__all__ = [
    "LockBackend",
    "LockLostError",
    "PG_ADVISORY_KEY1",
    "PG_ADVISORY_KEY2",
    "PostgresAdvisoryLock",
    "REDIS_LOCK_KEY",
    "REDIS_LOCK_TTL_MS",
    "RedisLock",
    "acquire_blocking",
    "advisory_keys_from_lock_key",
    "build_dataset_writer_lock",
    "build_lock_backend",
    "dataset_append_lock_key",
    "events_apply_lock_key",
    "has_distributed_lock",
    "held_writer_lock",
]


class LockLostError(RuntimeError):
    """Lease expired or was stolen before a fenced write."""


class LockBackend(Protocol):
    def acquire(self) -> bool: ...

    def release(self) -> None: ...

    def owned(self) -> bool:
        """True when this instance still holds the lease (fencing)."""
        ...

    def is_locked(self) -> bool:
        """True when any process holds this key (probe; does not acquire)."""
        ...


def has_distributed_lock(settings: Settings) -> bool:
    return settings.trigger.lock_backend != "in_process"


def acquire_blocking(
    lock: LockBackend,
    *,
    timeout_seconds: float = DEFAULT_LOCK_ACQUIRE_TIMEOUT_SECONDS,
    interval_seconds: float = 0.05,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if lock.acquire():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval_seconds)


@contextmanager
def held_writer_lock(lock: LockBackend | None) -> Iterator[None]:
    if lock is None:
        yield
        return
    if not acquire_blocking(lock):
        raise RuntimeError("dataset writer lock busy")
    try:
        yield
    finally:
        lock.release()


def build_dataset_writer_lock(settings: Settings) -> LockBackend | None:
    if not has_distributed_lock(settings) or settings.output.kind != "dataset":
        return None
    return build_lock_backend(
        settings,
        lock_key=dataset_append_lock_key(settings.trigger.lock_key),
        ttl_seconds=min(
            settings.trigger.lock_ttl_seconds,
            DEFAULT_DATASET_APPEND_LOCK_TTL_SECONDS,
        ),
    )


def build_lock_backend(
    settings: Settings,
    *,
    lock_key: str | None = None,
    ttl_seconds: float | None = None,
) -> LockBackend:
    """Build a distributed lock backend (``postgres`` / ``redis`` only).

    ``lock_backend`` and required URLs must already be validated at config load.
    Callers must not invoke this for ``in_process``.
    ``lock_key`` overrides ``settings.trigger.lock_key`` (events apply lease).
    ``ttl_seconds`` overrides Redis TTL (Postgres advisory locks have none).
    """
    backend = settings.trigger.lock_backend
    key = settings.trigger.lock_key if lock_key is None else lock_key
    if backend == "postgres":
        url = resolve_postgres_lock_url(settings)
        assert url is not None, "postgres lock URL should be validated at config load"
        return PostgresAdvisoryLock(url, lock_key=key)
    if backend == "redis":
        redis_url = settings.trigger.redis_url
        assert redis_url is not None, "redis_url should be validated at config load"
        ttl = settings.trigger.lock_ttl_seconds if ttl_seconds is None else ttl_seconds
        return RedisLock(
            redis_url,
            key=key,
            ttl_ms=int(ttl * 1000),
        )
    raise AssertionError(f"build_lock_backend is only for distributed backends, got {backend!r}")
