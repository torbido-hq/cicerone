"""Acquire, fence, and thread-local hold tracking for writer locks."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Protocol

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

    def last_input_marker(self) -> str | None:
        """Input change the last poll-triggered run started from (shared by every instance)."""
        ...

    def set_input_marker(self, marker: str) -> None:
        """Record ``marker``; call only while holding the lock."""
        ...


def acquire_once(lock: LockBackend) -> tuple[bool, int | None]:
    try_acquire = getattr(lock, "try_acquire", None)
    if callable(try_acquire):
        generation = try_acquire()
        if generation is None:
            return False, None
        return True, generation
    if lock.acquire():
        return True, getattr(lock, "hold_generation", None)
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
    if lock is not None and not lock_owned(lock, generation):
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
    ok, generation = acquire_until(lock)
    if not ok:
        raise WriterLockBusyError("dataset writer lock busy")
    bind_writer_generation(lock, generation)
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
        unbind_writer_generation()
        release_generation = getattr(lock, "release_generation", None)
        if callable(release_generation) and generation is not None:
            release_generation(generation)
        else:
            lock.release()


@dataclass(frozen=True)
class WriteFence:
    """Writer lock plus fence check guarding one store's writes."""

    lock: LockBackend | None = None
    check: Callable[[], bool] | None = None
    lost: str = "lock lost before write"
    kind: str = "lock"

    @classmethod
    def resolve(
        cls,
        fence: WriteFence | None,
        *,
        writer_lock: LockBackend | None = None,
        fence_check: Callable[[], bool] | None = None,
        fence_lost: str = "lock lost before write",
        fence_kind: str = "lock",
    ) -> WriteFence:
        if fence is None:
            return cls(writer_lock, fence_check, fence_lost, fence_kind)
        # Identity, not equality: a lock or check may define its own __eq__.
        if (
            writer_lock is not None
            or fence_check is not None
            or (fence_lost, fence_kind) != (cls.lost, cls.kind)
        ):
            raise ValueError("pass either fence= or writer_lock/fence_* keywords, not both")
        return fence

    def held_here(self) -> bool:
        return writer_lock_held_here(self.lock)

    def ensure(self) -> None:
        ensure_writer_owned(self.lock, fence_check=self.check, fence_lost=self.lost, fence_kind=self.kind)

    def hold(self) -> AbstractContextManager[None]:
        return held_writer_lock(self.lock, fence_check=self.check, fence_lost=self.lost, fence_kind=self.kind)

    @contextmanager
    def lease(self) -> Iterator[None]:
        """Re-entrant hold: re-check the fence if this thread already holds the lock."""
        if self.held_here():
            self.ensure()
            yield
            return
        with self.hold():
            yield
