"""In-flight source heartbeat for EventWorker apply."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any

from cicerone.events.base import EventSource, NormalizedEvent

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
    except Exception as exc:
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

    def _loop() -> None:
        while not stop.wait(interval_seconds):
            try:
                beat(events)
            except Exception:
                logger.exception("Event source heartbeat failed")
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
        raise HeartbeatError("Event source heartbeat failed")
