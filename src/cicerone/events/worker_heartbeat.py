"""In-flight source heartbeat for EventWorker apply."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from contextlib import contextmanager

from cicerone.events.base import EventSource, HeartbeatSource, NormalizedEvent
from cicerone.events.errors import EVENT_SOURCE_ERRORS

logger = logging.getLogger(__name__)


class HeartbeatError(RuntimeError):
    """Raised when an in-flight heartbeat fails (apply must not ack)."""


@contextmanager
def inflight_heartbeat(
    source: EventSource,
    events: Sequence[NormalizedEvent],
    interval_seconds: float,
):
    """Beat at start of apply and again every ``interval_seconds`` until exit."""
    if not isinstance(source, HeartbeatSource):
        yield
        return
    beat = source.heartbeat
    try:
        beat(events)
    except EVENT_SOURCE_ERRORS as exc:
        logger.exception("Event source heartbeat failed")
        raise HeartbeatError("Event source heartbeat failed") from exc
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
            except Exception as exc:
                # Heartbeat thread's top level: the caller, blocked in apply, re-raises what is recorded.
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
        recorded = caught[0] if caught else None
        if recorded is not None and not isinstance(recorded, EVENT_SOURCE_ERRORS):
            raise recorded
        if recorded is None:
            raise HeartbeatError("Event source heartbeat failed")
        raise HeartbeatError("Event source heartbeat failed") from recorded
