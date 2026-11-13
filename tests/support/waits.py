"""Bounded waits for tests that coordinate with background threads."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

WAIT_SECONDS = 5.0
# A component timeout the test must never reach: a call that waits for it fails a WAIT_SECONDS wait.
UNREACHED_SECONDS = 12 * WAIT_SECONDS


def wait_until(predicate: Callable[[], object], timeout: float = WAIT_SECONDS) -> bool:
    """Poll ``predicate`` until it is truthy; False once ``timeout`` seconds pass."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def call_in_thread(fn: Callable[[], T]) -> tuple[threading.Event, list[T]]:
    """Run ``fn`` on a daemon thread; the Event is set once its result is in the list."""
    done = threading.Event()
    result: list[T] = []

    def _run() -> None:
        result.append(fn())
        done.set()

    threading.Thread(target=_run, daemon=True).start()
    return done, result
