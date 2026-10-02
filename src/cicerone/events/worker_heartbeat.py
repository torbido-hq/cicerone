"""In-flight source heartbeat for EventWorker apply."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any

from cicerone.events.base import EventSource, NormalizedEvent
from cicerone.events.errors import EVENT_SOURCE_ERRORS

logger = logging.getLogger(__name__)


class HeartbeatError(RuntimeError):
    """Raised when an in-flight heartbeat fails (apply must not ack)."""


def _call_heartbeat(
    beat: Callable[..., Any],
    events: Sequence[NormalizedEvent],
    *,
    fail_closed: bool = False,
) -> None:
    try:
        beat(events)
    except EVENT_SOURCE_ERRORS as exc:
        logger.exception("Event source heartbeat failed")
        if fail_closed:
            raise HeartbeatError("Event source heartbeat failed") from exc


@contextmanager
def inflight_heartbeat(
    source: EventSource,
    events: Sequence[NormalizedEvent],
    interval_seconds: float,
):
    """Beat at start of apply and again every ``interval_seconds`` until exit."""
    beat = getattr(source, "heartbeat", None)
    if not callable(beat):
        yield
        return
    _call_heartbeat(beat, events, fail_closed=True)
    if interval_seconds <= 0:
        yield
        return
    stop = threading.Event()
    failed = threading.Event()
    caught: list[BaseException] = []

    def _loop() -> None:
        while not stop.wait(interval_seconds):
            try:
                beat(events)
            except EVENT_SOURCE_ERRORS as exc:
                logger.exception("Event source heartbeat failed")
                caught.append(exc)
                failed.set()
                return
            except Exception as exc:
                # The caller is blocked in apply; record and stop this thread.
                logger.exception("Event source heartbeat failed")
                caught.append(exc)
                failed.set()
                return

    thread = threading.Thread(target=_loop, name="cicerone-events-heartbeat", daemon=True)
    thread.start()
    completed = False
    try:
        yield
        completed = True
    finally:
        stop.set()
        thread.join(timeout=max(1.0, interval_seconds))
    if completed and (failed.is_set() or thread.is_alive()):
        exc = caught[0] if caught else None
        if exc is not None and not isinstance(exc, EVENT_SOURCE_ERRORS):
            raise exc
        if exc is None:
            raise HeartbeatError("Event source heartbeat failed")
        raise HeartbeatError("Event source heartbeat failed") from exc
