"""Thread-safe cached value: reloaded by ``refresh`` or once ``ttl_seconds`` old.

With ``initial``, ``get`` serves it until the first successful load instead of loading.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")
_UNSET: Any = object()


class RefreshedValue(Generic[T]):
    def __init__(
        self,
        load: Callable[[], T],
        *,
        ttl_seconds: float = math.inf,
        initial: T = _UNSET,
    ) -> None:
        self._load = load
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._state: tuple[float, T] | None = None if initial is _UNSET else (math.inf, initial)

    def refresh(self) -> None:
        with self._lock:
            self._reload()

    def get(self) -> T:
        state = self._state
        if state is not None and time.monotonic() < state[0]:
            return state[1]
        # One load at a time; waiters reuse its result instead of loading again.
        with self._lock:
            state = self._state
            if state is not None and time.monotonic() < state[0]:
                return state[1]
            return self._reload()

    def _reload(self) -> T:
        now = time.monotonic()
        value = self._load()
        self._state = (now + self._ttl_seconds, value)
        return value
