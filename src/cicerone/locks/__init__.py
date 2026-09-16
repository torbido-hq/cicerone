"""Optional distributed lock backends for multi-replica schedulers.

Default single-instance exclusion is RunGuard's threading.Lock (no backend).
``postgres`` / ``redis`` are opt-in; clients are imported only when selected.
"""

from __future__ import annotations

from cicerone.config.constants import DEFAULT_DATASET_APPEND_LOCK_TTL_SECONDS
from cicerone.config.lock_url import resolve_postgres_lock_url
from cicerone.config.settings import Settings
from cicerone.locks.hold import (
    LockBackend,
    LockLostError,
    WriterLockBusyError,
    acquire_blocking,
    ensure_writer_owned,
    held_writer_lock,
    writer_lock_held_here,
)
from cicerone.locks.hold import _acquire_once as _acquire_once
from cicerone.locks.hold import _acquire_until as _acquire_until
from cicerone.locks.hold import _bind_writer_generation as _bind_writer_generation
from cicerone.locks.hold import _bound_writer_generation as _bound_writer_generation
from cicerone.locks.hold import _lock_owned as _lock_owned
from cicerone.locks.hold import _unbind_writer_generation as _unbind_writer_generation
from cicerone.locks.hold import _writer_hold as _writer_hold
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
    "WriterLockBusyError",
    "PG_ADVISORY_KEY1",
    "PG_ADVISORY_KEY2",
    "PostgresAdvisoryLock",
    "REDIS_LOCK_KEY",
    "REDIS_LOCK_TTL_MS",
    "RedisLock",
    "acquire_blocking",
    "advisory_keys_from_lock_key",
    "build_dataset_writer_lock",
    "build_output_writer_lock",
    "build_lock_backend",
    "dataset_append_lock_key",
    "events_apply_lock_key",
    "has_distributed_lock",
    "ensure_writer_owned",
    "held_writer_lock",
    "writer_lock_held_here",
]


def has_distributed_lock(settings: Settings) -> bool:
    return settings.trigger.lock_backend != "in_process"


def build_output_writer_lock(settings: Settings) -> LockBackend | None:
    if not has_distributed_lock(settings) or settings.output.kind not in {"dataset", "db"}:
        return None
    return build_lock_backend(
        settings,
        lock_key=dataset_append_lock_key(settings.trigger.lock_key),
        ttl_seconds=min(
            settings.trigger.lock_ttl_seconds,
            DEFAULT_DATASET_APPEND_LOCK_TTL_SECONDS,
        ),
    )


def build_dataset_writer_lock(settings: Settings) -> LockBackend | None:
    if settings.output.kind != "dataset":
        return None
    return build_output_writer_lock(settings)


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
