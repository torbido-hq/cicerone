from __future__ import annotations

import pandas as pd
import pytest

from cicerone.config import EventsSettings, IOSettings, make_settings
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.normalize import EventNormalizeError, normalize_event
from cicerone.events.registry import build_event_source, register_event_source, registered_event_source_kinds
from cicerone.events.store import load_recommendations_frame
from cicerone.events.updater import IncrementalUpdater
from cicerone.events.webhook import WebhookEventSource
from cicerone.events.worker import EventWorker
from cicerone.feature_config import FeatureConfig
from cicerone.io.recommendation_reader import RECOMMENDATION_COLUMNS


def _event(**overrides) -> dict:
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


def test_register_and_build_webhook():
    assert "webhook" in registered_event_source_kinds()
    source = build_event_source("webhook", {})
    assert isinstance(source, WebhookEventSource)


def test_build_unknown_kind():
    with pytest.raises(ValueError, match="Unknown events kind"):
        build_event_source("nope", {})


def test_register_duplicate_kind():
    with pytest.raises(ValueError, match="already registered"):
        register_event_source("webhook", lambda options: WebhookEventSource(options))


def test_normalize_event_and_errors():
    event = normalize_event(_event())
    assert event.user_id == "u1"
    assert event.quantity == 1
    with pytest.raises(EventNormalizeError, match="missing"):
        normalize_event({"user_id": "u1"})
    with pytest.raises(EventNormalizeError, match="quantity"):
        normalize_event(_event(quantity=0))


def test_webhook_ingest_poll_ack_health():
    source = WebhookEventSource({})
    source.connect()
    source.ingest(_event(event_id="a"))
    source.ingest([_event(event_id="b", item_id="i2")])
    health = source.health()
    assert health.connected is True
    assert health.lag == 2
    polled = list(source.poll(1))
    assert len(polled) == 1
    assert source.health().lag == 2  # one pending + one in-flight
    source.ack([polled[0].event_id])
    assert source.health().lag == 1


def test_micro_batch_buffer_count_and_dedupe():
    buffer = MicroBatchBuffer(batch_size=2, batch_window_seconds=60.0)
    e1 = normalize_event(_event(event_id="1"))
    e2 = normalize_event(_event(event_id="dup"))  # same fingerprint as e1
    e3 = normalize_event(_event(event_id="3", item_id="i2"))
    assert buffer.extend([e1, e2]) == 1
    assert buffer.ready() is False
    assert buffer.extend([e3]) == 1
    flushed = buffer.flush_if_ready()
    assert len(flushed) == 2


def test_micro_batch_buffer_window():
    buffer = MicroBatchBuffer(batch_size=100, batch_window_seconds=10.0)
    buffer.extend([normalize_event(_event())])
    assert buffer.ready(now=0.0) is False
    buffer._window_started_at = 0.0
    assert buffer.ready(now=10.0) is True


def test_incremental_updater_write_through(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    existing = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "old",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
            },
            {
                "user_id": "u2",
                "item_id": "x",
                "rank": 1,
                "score": 0.5,
                "source": "personalized",
            },
        ]
    )
    existing.to_parquet(out / "recommendations.parquet", index=False)

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    from cicerone.io.factory import build_output_sink

    sink = build_output_sink(settings.output)
    updater = IncrementalUpdater(
        sink=sink,
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
    )
    events = [
        normalize_event(_event(user_id="u1", item_id="i9", event_id="n1")),
        normalize_event(_event(user_id="u1", item_id="i8", event_type="view", event_id="n2")),
    ]
    assert updater.apply(events) == 2
    frame = load_recommendations_frame(settings.output)
    u1 = frame[frame["user_id"] == "u1"]
    assert "old" in set(u1["item_id"].astype(str))
    assert "i9" in set(u1["item_id"].astype(str))
    u2 = frame[frame["user_id"] == "u2"]
    assert list(u2["item_id"]) == ["x"]
    cold = frame[frame["user_id"] == "__cold_start__"]
    assert not cold.empty
    assert updater.events_applied == 2
    assert updater.last_success_at is not None


def test_incremental_updater_skips_when_busy(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(columns=list(RECOMMENDATION_COLUMNS)).to_parquet(
        out / "recommendations.parquet", index=False
    )
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
    )
    from cicerone.io.factory import build_output_sink

    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        busy_check=lambda: True,
    )
    assert updater.apply([normalize_event(_event())]) == 0


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
    from cicerone.io.factory import build_output_sink

    source = WebhookEventSource({})
    source.ingest(_event(event_id="w1", item_id="i7"))
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


def test_make_settings_events_defaults():
    settings = make_settings()
    assert settings.events.enabled is False
    assert settings.events.kind == "webhook"
    assert settings.events.incremental.batch_size == 100
    assert settings.events_enabled is False
    assert settings.events_kind == "webhook"


def test_normalize_more_edge_cases():
    from cicerone.events.normalize import events_to_dataframe, normalize_events

    with pytest.raises(EventNormalizeError, match="JSON object"):
        normalize_event("not-a-dict")
    with pytest.raises(EventNormalizeError, match="quantity"):
        normalize_event(_event(quantity="x"))
    with pytest.raises(EventNormalizeError, match="non-empty"):
        normalize_event(_event(user_id="  "))
    with pytest.raises(EventNormalizeError, match="occurred_at"):
        normalize_event(_event(occurred_at="not-a-date"))
    bare = normalize_event(
        {
            "user_id": "u1",
            "item_id": "i1",
            "event_type": "view",
            "occurred_at": "2026-08-13T12:00:00",
            "idempotency_key": "idem-1",
        }
    )
    assert bare.event_id == "idem-1"
    assert bare.quantity == 1
    assert events_to_dataframe([]).empty
    assert len(normalize_events([_event(event_id="z")])) == 1


def test_buffer_validation_and_len():
    with pytest.raises(ValueError, match="batch_size"):
        MicroBatchBuffer(batch_size=0, batch_window_seconds=1.0)
    with pytest.raises(ValueError, match="batch_window"):
        MicroBatchBuffer(batch_size=1, batch_window_seconds=0)
    buffer = MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0, dedupe=False)
    assert len(buffer) == 0
    assert buffer.flush_if_ready() == []
    buffer.extend([normalize_event(_event()), normalize_event(_event(event_id="same-fp"))])
    assert len(buffer) == 2
    assert buffer.ready() is False


def test_webhook_poll_zero_and_reject_scalar():
    source = WebhookEventSource({})
    assert source.poll(0) == []
    with pytest.raises(EventNormalizeError):
        source.ingest(123)


def test_load_recommendations_missing_file(tmp_path):
    from cicerone.events.store import empty_recommendations_frame

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    frame = load_recommendations_frame(settings.output)
    assert list(frame.columns) == list(empty_recommendations_frame().columns)
    assert frame.empty


def test_load_recommendations_unsupported_kind():
    with pytest.raises(ValueError, match="Unsupported output kind"):
        load_recommendations_frame(IOSettings(kind="other", options={}))


def test_incremental_updater_empty_and_unknown_event_type(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    from cicerone.io.factory import build_output_sink

    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        on_success=lambda: None,
    )
    assert updater.apply([]) == 0
    applied = updater.apply([normalize_event(_event(event_type="unknown_type", event_id="u", item_id="ix"))])
    assert applied == 1
    frame = load_recommendations_frame(settings.output)
    assert not frame.empty


def test_incremental_updater_no_feature_config(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=2,
    )
    from cicerone.io.factory import build_output_sink

    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=None,
        top_k=2,
    )
    assert updater.apply([normalize_event(_event(event_id="nfc"))]) == 1


def test_event_worker_busy_requeues_and_start_stop(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    from cicerone.io.factory import build_output_sink

    source = WebhookEventSource({})
    source.ingest(_event(event_id="busy1", item_id="ib"))
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
    assert len(buffer) == 1

    worker.start()
    worker.start()
    worker.stop()


def test_event_worker_tick_noop_when_empty(tmp_path, feature_config: FeatureConfig):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    from cicerone.io.factory import build_output_sink

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


def test_start_events_runtime_disabled_and_webhook(tmp_path, feature_config: FeatureConfig):
    from cicerone.serve.bootstrap_events import start_events_runtime

    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)

    class _Reader:
        def refresh(self) -> None:
            return None

    disabled = start_events_runtime(
        make_settings(
            output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)})
        ),
        feature_config=feature_config,
        reader=_Reader(),  # type: ignore[arg-type]
    )
    assert disabled.webhook_source is None
    assert disabled.worker is None
    disabled.stop()

    enabled = start_events_runtime(
        make_settings(
            output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
            events=EventsSettings(enabled=True, kind="webhook"),
        ),
        feature_config=feature_config,
        reader=_Reader(),  # type: ignore[arg-type]
    )
    assert isinstance(enabled.webhook_source, WebhookEventSource)
    assert enabled.worker is not None
    enabled.stop()


def test_coerce_events_settings_errors():
    from cicerone.config.events import coerce_events_settings

    with pytest.raises(TypeError):
        coerce_events_settings("bad")
    with pytest.raises(TypeError):
        coerce_events_settings({"incremental": "bad"})
