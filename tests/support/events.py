"""Shared helpers for events package tests."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from cicerone.config import make_settings
from cicerone.events.base import NormalizedEvent
from cicerone.events.online_result import OnlineRefreshResult
from cicerone.events.updater import IncrementalUpdater
from cicerone.events.webhook import WebhookEventSource
from cicerone.events.worker import EventWorker
from cicerone.io.factory import build_output_sink
from support.settings import local_output
from support.waits import WAIT_SECONDS


class FakeOnline:
    """Online refresher whose every refresh returns ``rows`` for one user."""

    def __init__(self, rows: Any) -> None:
        self._rows = rows

    def refresh(self, events: Sequence[NormalizedEvent]) -> OnlineRefreshResult:
        del events
        return OnlineRefreshResult(rows=self._rows, users_refreshed=1, fit_partial_epochs=1)

    def invalidate(self) -> None:
        return None


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
        output=local_output(out),
        top_k=3,
    )
    sink = build_output_sink(settings.output)
    updater = IncrementalUpdater(
        sink=sink, output_settings=settings.output, feature_config=feature_config, top_k=3, **kwargs
    )
    return settings, sink, updater
