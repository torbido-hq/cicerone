"""Shared helpers for events package tests."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from cicerone.config import IOSettings, make_settings
from cicerone.events.updater import IncrementalUpdater
from cicerone.events.webhook import WebhookEventSource
from cicerone.io.factory import build_output_sink


def wait_until(predicate: Callable[[], object], timeout: float = 2.0) -> bool:
    """Poll ``predicate`` until it is truthy; False once ``timeout`` seconds pass."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


class BeatingSource(WebhookEventSource):
    """Counts heartbeats; sets ``second_beat`` on the second and raises ``error`` from then on."""

    def __init__(self, error: BaseException | None = None) -> None:
        super().__init__({})
        self.beats = 0
        self.error = error
        self.second_beat = threading.Event()

    def heartbeat(self, events) -> None:  # type: ignore[no-untyped-def]
        del events
        self.beats += 1
        if self.beats > 1:
            self.second_beat.set()
            if self.error is not None:
                raise self.error


def apply_after_second_beat(worker, source: BeatingSource) -> None:  # type: ignore[no-untyped-def]
    """Hold ``worker``'s apply until the in-flight heartbeat thread has beaten twice."""
    original = worker._updater.apply

    def _held(events, *, persist_online: bool = True):  # type: ignore[no-untyped-def]
        source.second_beat.wait(timeout=5)
        return original(events, persist_online=persist_online)

    worker._updater.apply = _held


def event_payload(**overrides) -> dict:
    base = {
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "purchase",
        "quantity": 1,
        "occurred_at": "2026-08-13T12:00:00Z",
        "event_id": "e1",
    }
    base.update(overrides)
    return base


def dataset_updater(tmp_path, feature_config, **kwargs):  # type: ignore[no-untyped-def]
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    sink = build_output_sink(settings.output)
    updater = IncrementalUpdater(
        sink=sink, output_settings=settings.output, feature_config=feature_config, top_k=3, **kwargs
    )
    return settings, sink, updater
