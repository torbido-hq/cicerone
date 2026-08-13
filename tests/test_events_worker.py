from __future__ import annotations

import pandas as pd
from support.events import event_payload

from cicerone.config import EventsSettings, IOSettings, make_settings
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.updater import IncrementalUpdater
from cicerone.events.webhook import WebhookEventSource
from cicerone.events.worker import EventWorker
from cicerone.feature_config import FeatureConfig
from cicerone.io.factory import build_output_sink


def test_event_worker_tick(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
        events=EventsSettings(enabled=True, kind="webhook"),
    )
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="w1", item_id="i7"))
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
    )
    buffer = MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0)
    worker = EventWorker(source, buffer, updater, poll_interval_seconds=0.01)
    assert worker.tick() == 1
    assert source.health().lag == 0


def test_event_worker_busy_nacks(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="busy1", item_id="ib"))
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        busy_check=lambda: True,
    )
    buffer = MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0)
    worker = EventWorker(source, buffer, updater, poll_interval_seconds=0.01)
    assert worker.tick() == 0
    assert source.health().lag == 1


def test_event_worker_apply_failure_nacks(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="fail1"))

    class _Boom(IncrementalUpdater):
        def apply(self, events):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0),
        _Boom(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    try:
        worker.tick()
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert source.health().lag == 1


def test_event_worker_tick_noop_when_empty(tmp_path, feature_config: FeatureConfig):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    worker = EventWorker(
        WebhookEventSource({}),
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    assert worker.tick() == 0
