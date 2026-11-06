"""Shared helpers for events package tests."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from cicerone.config import IOSettings, make_settings
from cicerone.events.base import NormalizedEvent
from cicerone.events.updater import IncrementalUpdater
from cicerone.events.webhook import WebhookEventSource
from cicerone.events.worker import EventWorker
from cicerone.io.factory import build_output_sink
from support.waits import WAIT_SECONDS


class BeatingSource(WebhookEventSource):
    """Counts heartbeats; sets ``second_beat`` on the second and raises ``error`` from then on."""

    def __init__(self, error: BaseException | None = None) -> None:
        super().__init__({})
        self.beats = 0
        self.error = error
        self.second_beat = threading.Event()

    def heartbeat(self, events: Sequence[NormalizedEvent]) -> None:
        del events
        self.beats += 1
        if self.beats > 1:
            self.second_beat.set()
            if self.error is not None:
                raise self.error


def apply_after_second_beat(worker: EventWorker, source: BeatingSource) -> None:
    """Hold ``worker``'s apply until the in-flight heartbeat thread has beaten twice."""
    original = worker._updater.apply

    def _held(events: Sequence[NormalizedEvent], *, persist_online: bool = True) -> Any:
        assert source.second_beat.wait(WAIT_SECONDS), "in-flight heartbeat never beat twice"
        return original(events, persist_online=persist_online)

    worker._updater.apply = _held  # type: ignore[method-assign]


# json.loads rejects these without a JSONDecodeError: int digit limit (ValueError), nesting (RecursionError).
UNPARSABLE_JSON_BODIES = (b'{"user_id": ' + b"1" * 4301 + b"}", b"[" * 200_000)


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


def dataset_updater(tmp_path, feature_config, top_k=3, **kwargs):  # type: ignore[no-untyped-def]
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=top_k,
    )
    sink = build_output_sink(settings.output)
    updater = IncrementalUpdater(
        sink=sink, output_settings=settings.output, feature_config=feature_config, top_k=top_k, **kwargs
    )
    return settings, sink, updater
