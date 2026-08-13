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


def test_event_worker_partial_apply_nacks(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="p1", item_id="i1"))
    source.ingest(event_payload(event_id="p2", item_id="i2"))

    class _Partial(IncrementalUpdater):
        def apply(self, events):  # type: ignore[no-untyped-def]
            return max(len(events) - 1, 0)

    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=2, batch_window_seconds=60.0),
        _Partial(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    try:
        worker.tick()
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "partial apply" in str(exc)
    assert raised
    assert source.health().lag == 2


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


def test_event_worker_stop_returns_false_when_join_times_out(tmp_path, feature_config, caplog):
    import logging

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
        poll_interval_seconds=0.01,
    )
    worker.start()
    assert worker._thread is not None
    worker._thread.join = lambda timeout=None: None  # type: ignore[method-assign]
    worker._thread.is_alive = lambda: True  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING):
        assert worker.stop(join_timeout_seconds=0.01) is False
    assert any("still alive" in record.getMessage() for record in caplog.records)
    worker._stop.set()


def test_event_worker_stop_returns_true_when_idle(tmp_path, feature_config: FeatureConfig):
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
        poll_interval_seconds=0.01,
    )
    worker.start()
    assert worker.stop(join_timeout_seconds=2.0) is True


def test_event_worker_stop_swallows_source_close_errors(tmp_path, feature_config, caplog):
    import logging

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )

    class _BoomClose(WebhookEventSource):
        def close(self) -> None:
            raise RuntimeError("close failed")

    worker = EventWorker(
        _BoomClose({}),
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
        poll_interval_seconds=0.01,
    )
    worker.start()
    with caplog.at_level(logging.ERROR):
        assert worker.stop(join_timeout_seconds=2.0) is True
    assert any("close()" in record.getMessage() for record in caplog.records)
