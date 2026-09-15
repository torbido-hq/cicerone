"""Optional distributed lock backends for multi-replica schedulers.

Default single-instance exclusion is RunGuard's threading.Lock (no backend).
``postgres`` / ``redis`` are opt-in; clients are imported only when selected.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
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
    "build_lock_backend",
    "dataset_append_lock_key",
    "events_apply_lock_key",
    "has_distributed_lock",
    "ensure_writer_owned",
    "held_writer_lock",
    "writer_lock_held_here",
]


class LockLostError(RuntimeError):
    """Lease expired or was stolen before a fenced write."""

    def __init__(self, message: str, *, kind: str = "lock") -> None:
        super().__init__(message)
        self.kind = kind


class WriterLockBusyError(RuntimeError):
    """Another writer held the dataset-append lease until acquire timed out."""


class LockBackend(Protocol):
    def acquire(self) -> bool: ...

    def release(self) -> None: ...

    def owned(self, generation: int | None = None) -> bool:
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


_writer_hold = threading.local()


def _bind_writer_generation(lock: LockBackend, generation: int | None) -> None:
    stack = getattr(_writer_hold, "stack", None)
    if stack is None:
        _writer_hold.stack = []
        stack = _writer_hold.stack
    stack.append((id(lock), generation))


def _unbind_writer_generation() -> None:
    stack = getattr(_writer_hold, "stack", None)
    if stack:
        stack.pop()


def writer_lock_held_here(lock: LockBackend | None) -> bool:
    if lock is None:
        return False
    stack = getattr(_writer_hold, "stack", None)
    if not stack:
        return False
    lock_id = id(lock)
    return any(stored_id == lock_id for stored_id, _generation in stack)


def _bound_writer_generation(lock: LockBackend) -> int | None:
    stack = getattr(_writer_hold, "stack", None)
    if not stack:
        return None
    lock_id = id(lock)
    for stored_id, generation in reversed(stack):
        if stored_id == lock_id:
            return generation
    return None


def _lock_owned(lock: LockBackend, generation: int | None) -> bool:
    expected = generation if generation is not None else _bound_writer_generation(lock)
    owned = lock.owned
    if expected is None:
        return bool(owned())
    try:
        return bool(owned(generation=expected))
    except TypeError:
        current = getattr(lock, "hold_generation", None)
        return bool(owned()) and current == expected


def ensure_writer_owned(
    lock: LockBackend | None,
    *,
    generation: int | None = None,
    fence_check: Callable[[], bool] | None = None,
    fence_lost: str = "lock lost before write",
    fence_kind: str = "lock",
) -> None:
    if lock is not None and not _lock_owned(lock, generation):
        raise LockLostError("dataset writer lock lost before write", kind="writer")
    if fence_check is not None and not fence_check():
        raise LockLostError(fence_lost, kind=fence_kind)


@contextmanager
def held_writer_lock(
    lock: LockBackend | None,
    *,
    fence_check: Callable[[], bool] | None = None,
    fence_lost: str = "lock lost before write",
    fence_kind: str = "lock",
) -> Iterator[None]:
    if lock is None:
        if fence_check is not None and not fence_check():
            raise LockLostError(fence_lost, kind=fence_kind)
        yield
        return
    if not acquire_blocking(lock):
        raise WriterLockBusyError("dataset writer lock busy")
    generation = getattr(lock, "hold_generation", None)
    _bind_writer_generation(lock, generation)
    try:
        ensure_writer_owned(
            lock,
            generation=generation,
            fence_check=fence_check,
            fence_lost=fence_lost,
            fence_kind=fence_kind,
        )
        yield
    finally:
        _unbind_writer_generation()
        release_generation = getattr(lock, "release_generation", None)
        if callable(release_generation) and generation is not None:
            release_generation(generation)
        else:
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
