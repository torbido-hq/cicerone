"""Optional distributed lock backends for multi-replica schedulers.

Default single-instance exclusion is RunGuard's threading.Lock (no backend).
``postgres`` / ``redis`` are opt-in; clients are imported only when selected.
"""

from __future__ import annotations

from typing import Protocol

from cicerone.config.lock_url import resolve_postgres_lock_url
from cicerone.config.settings import Settings
from cicerone.locks.keys import (
    PG_ADVISORY_KEY1,
    PG_ADVISORY_KEY2,
    REDIS_LOCK_KEY,
    REDIS_LOCK_TTL_MS,
    advisory_keys_from_lock_key,
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
    "advisory_keys_from_lock_key",
    "build_lock_backend",
    "events_apply_lock_key",
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
