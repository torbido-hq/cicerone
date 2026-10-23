"""Shared fakes for storage tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class FakeS3Body:
    def __init__(self, payload: bytes, *, fail: bool = False) -> None:
        self._payload = payload
        self._fail = fail
        self.closed = False
        self.reads: list[int | None] = []

    def read(self, size: int | None = None) -> bytes:
        self.reads.append(size)
        if self._fail:
            raise RuntimeError("read failed")
        if size is None:
            return self._payload
        return self._payload[:size]

    def close(self) -> None:
        self.closed = True


class FakeS3Client:
    def __init__(self, body: FakeS3Body) -> None:
        self.body = body

    def get_object(self, **_kwargs: Any) -> dict[str, Any]:
        return {"Body": self.body}


class FakeLock:
    """Writer lock whose ``acquire`` / ``owned`` answers are fixed values or callables."""

    def __init__(
        self, *, acquire: bool | Callable[[], bool] = True, owned: bool | Callable[[], bool] = True
    ) -> None:
        self._acquire = acquire
        self._owned = owned
        self.acquires = 0

    def acquire(self) -> bool:
        self.acquires += 1
        return self._acquire() if callable(self._acquire) else self._acquire

    def release(self) -> None:
        return None

    def owned(self) -> bool:
        return self._owned() if callable(self._owned) else self._owned

    def is_locked(self) -> bool:
        return True


class FenceCheck:
    """Fence check that passes for the first ``fail_at - 1`` calls, then fails."""

    def __init__(self, fail_at: int) -> None:
        self.fail_at = fail_at
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls < self.fail_at


def raising(exc: BaseException) -> Callable[..., Any]:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise exc

    return _raise
