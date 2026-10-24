"""Acquire, fence, and thread-local hold tracking for writer locks."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from cicerone.capability import CapabilityMeta
from cicerone.config.constants import DEFAULT_LOCK_ACQUIRE_TIMEOUT_SECONDS


class LockLostError(RuntimeError):
    """Lease expired or was stolen before a fenced write."""

    def __init__(self, message: str, *, kind: str = "lock") -> None:
        super().__init__(message)
        self.kind = kind


class WriterLockBusyError(RuntimeError):
    """A blocking acquire timed out while another holder had the lease."""


class LockBackend(Protocol):
    def acquire(self) -> bool: ...

    def release(self) -> None: ...

    def owned(self, generation: int | None = None) -> bool:
        """True when this instance still holds the lease (fencing)."""
        ...

    def is_locked(self) -> bool:
        """True when any process holds this key (probe; does not acquire)."""
        ...


# Optional lock capabilities; callers check them with isinstance.
@runtime_checkable
class TryAcquireLock(Protocol, metaclass=CapabilityMeta):
    def try_acquire(self) -> int | None:
        """Hold generation on success, ``None`` when busy."""
        ...


@runtime_checkable
class GenerationLock(Protocol, metaclass=CapabilityMeta):
    @property
    def hold_generation(self) -> int: ...


@runtime_checkable
class GenerationReleaseLock(Protocol, metaclass=CapabilityMeta):
    def release_generation(self, generation: int) -> None: ...


def _hold_generation(lock: LockBackend) -> int | None:
    return lock.hold_generation if isinstance(lock, GenerationLock) else None


def acquire_once(lock: LockBackend) -> tuple[bool, int | None]:
    if isinstance(lock, TryAcquireLock):
        generation = lock.try_acquire()
        if generation is None:
            return False, None
        return True, generation
    if lock.acquire():
        return True, _hold_generation(lock)
    return False, None


def acquire_until(
    lock: LockBackend,
    *,
    timeout_seconds: float = DEFAULT_LOCK_ACQUIRE_TIMEOUT_SECONDS,
    interval_seconds: float = 0.05,
) -> tuple[bool, int | None]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        ok, generation = acquire_once(lock)
        if ok:
            return True, generation
        if time.monotonic() >= deadline:
            return False, None
        time.sleep(interval_seconds)


def acquire_blocking(
    lock: LockBackend,
    *,
    timeout_seconds: float = DEFAULT_LOCK_ACQUIRE_TIMEOUT_SECONDS,
    interval_seconds: float = 0.05,
) -> bool:
    ok, _generation = acquire_until(lock, timeout_seconds=timeout_seconds, interval_seconds=interval_seconds)
    return ok


writer_hold = threading.local()


def bind_writer_generation(lock: LockBackend, generation: int | None) -> None:
    stack = getattr(writer_hold, "stack", None)
    if stack is None:
        writer_hold.stack = []
        stack = writer_hold.stack
    stack.append((id(lock), generation))


def unbind_writer_generation() -> None:
    stack = getattr(writer_hold, "stack", None)
    if stack:
        stack.pop()


def writer_lock_held_here(lock: LockBackend | None) -> bool:
    if lock is None:
        return False
    stack = getattr(writer_hold, "stack", None)
    if not stack:
        return False
    lock_id = id(lock)
    return any(stored_id == lock_id for stored_id, _generation in stack)


def bound_writer_generation(lock: LockBackend) -> int | None:
    stack = getattr(writer_hold, "stack", None)
    if not stack:
        return None
    lock_id = id(lock)
    for stored_id, generation in reversed(stack):
        if stored_id == lock_id:
            return generation
    return None


def lock_owned(lock: LockBackend, generation: int | None) -> bool:
    expected = generation if generation is not None else bound_writer_generation(lock)
    owned = lock.owned
    if expected is None:
        return bool(owned())
    try:
        return bool(owned(generation=expected))
    except TypeError:
        return bool(owned()) and _hold_generation(lock) == expected


@dataclass(frozen=True)
class WriteFence:
    """Writer lock plus fence check guarding one store's writes."""

    lock: LockBackend | None = None
    check: Callable[[], bool] | None = None
    lost: str = "lock lost before write"
    kind: str = "lock"

    def held_here(self) -> bool:
        return writer_lock_held_here(self.lock)

    def ensure(self, generation: int | None = None) -> None:
        if self.lock is not None and not lock_owned(self.lock, generation):
            raise LockLostError("dataset writer lock lost before write", kind="writer")
        if self.check is not None and not self.check():
            raise LockLostError(self.lost, kind=self.kind)

    @contextmanager
    def hold(self) -> Iterator[None]:
        if self.lock is None:
            self.ensure()
            yield
            return
        ok, generation = acquire_until(self.lock)
        if not ok:
            raise WriterLockBusyError("dataset writer lock busy")
        bind_writer_generation(self.lock, generation)
        try:
            self.ensure(generation)
            yield
        finally:
            unbind_writer_generation()
            if isinstance(self.lock, GenerationReleaseLock) and generation is not None:
                self.lock.release_generation(generation)
            else:
                self.lock.release()

    @contextmanager
    def lease(self) -> Iterator[None]:
        """Re-entrant hold: re-check the fence if this thread already holds the lock."""
        if self.held_here():
            self.ensure()
            yield
            return
        with self.hold():
            yield
