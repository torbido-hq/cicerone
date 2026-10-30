"""Shared helpers for events package tests."""

from __future__ import annotations

from cicerone.config import IOSettings, make_settings
from cicerone.events.updater import IncrementalUpdater
from cicerone.io.factory import build_output_sink


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
