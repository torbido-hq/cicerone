from __future__ import annotations

from contextlib import contextmanager

import pandas as pd
import pytest
from support.events import event_payload
from support.prometheus_metrics import registry_metric_value

from cicerone.config import IOSettings, make_settings
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.normalize import normalize_event
from cicerone.events.updater import IncrementalUpdater
from cicerone.events.webhook import WebhookEventSource
from cicerone.events.worker import EventWorker, HeartbeatError
from cicerone.feature_config import FeatureConfig
from cicerone.io.factory import build_manifest_reader, build_output_sink
from cicerone.publish.base import PublishError


def _settings(tmp_path):  # type: ignore[no-untyped-def]
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    return make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )


def _fails_once(real, exc: Exception):  # type: ignore[no-untyped-def]
    calls = {"n": 0}

    def _call(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise exc
        return real(*args, **kwargs)

    return _call, calls


class _Publisher:
    def __init__(self, fail_with: Exception | None = None) -> None:
        self.fail_with = fail_with
        self.published: list[tuple[pd.DataFrame, list[str]]] = []

    def connect(self) -> None:
        if self.fail_with is not None:
            exc, self.fail_with = self.fail_with, None
            raise exc

    def publish(self, df: pd.DataFrame, *, user_ids=None) -> None:  # type: ignore[no-untyped-def]
        self.published.append((df.copy(), list(user_ids or [])))

    def close(self) -> None:
        return None


def _updater(settings, feature_config: FeatureConfig, **kwargs):  # type: ignore[no-untyped-def]
    sink = kwargs.pop("sink", None) or build_output_sink(settings.output)
    return sink, IncrementalUpdater(
        sink=sink, output_settings=settings.output, feature_config=feature_config, top_k=3, **kwargs
    )


def _events(event_id: str = "n1") -> list:  # type: ignore[type-arg]
    return [normalize_event(event_payload(event_id=event_id, user_id="u1", item_id="i9"))]


def test_a_failed_manifest_is_written_by_the_repair(tmp_path, feature_config: FeatureConfig) -> None:
    settings = _settings(tmp_path)
    sink = build_output_sink(settings.output)
    sink.write_manifest, _calls = _fails_once(sink.write_manifest, OSError("disk full"))  # type: ignore[method-assign]
    _sink, updater = _updater(settings, feature_config, sink=sink)

    with pytest.raises(OSError, match="disk full"):
        updater.apply(_events())
    assert updater.post_write_pending
    assert build_manifest_reader(settings.output).read_latest() is None

    updater.repair_post_write()
    assert not updater.post_write_pending
    latest = build_manifest_reader(settings.output).read_latest()
    assert latest is not None
    assert latest["triggered_by"] == "incremental"


def test_the_repair_does_not_overwrite_a_newer_manifest(tmp_path, feature_config: FeatureConfig) -> None:
    settings = _settings(tmp_path)
    sink = build_output_sink(settings.output)
    real_write = sink.write_manifest
    sink.write_manifest, _calls = _fails_once(real_write, OSError("disk full"))  # type: ignore[method-assign]
    _sink, updater = _updater(settings, feature_config, sink=sink)
    with pytest.raises(OSError, match="disk full"):
        updater.apply(_events())

    real_write({"triggered_by": "job", "status": "success", "generated_at": "2999-01-01T00:00:00+00:00"})
    updater.repair_post_write()
    assert not updater.post_write_pending
    assert build_manifest_reader(settings.output).read_latest()["triggered_by"] == "job"


def test_a_failed_reader_refresh_is_retried(tmp_path, feature_config: FeatureConfig) -> None:
    refreshed: list[int] = []

    def _refresh() -> None:
        refreshed.append(1)
        if len(refreshed) == 1:
            raise RuntimeError("reader refresh failed")

    _sink, updater = _updater(_settings(tmp_path), feature_config, on_success=_refresh)
    with pytest.raises(RuntimeError, match="reader refresh failed"):
        updater.apply(_events())
    assert updater.post_write_pending

    updater.repair_post_write()
    assert refreshed == [1, 1]
    assert not updater.post_write_pending


@pytest.mark.parametrize("failure", [RuntimeError("sidecar bug"), PublishError("broker down")])
def test_a_failed_sidecar_publish_is_retried_with_the_stored_rows(
    tmp_path, feature_config: FeatureConfig, failure: Exception
) -> None:
    publisher = _Publisher(fail_with=failure)
    _sink, updater = _updater(_settings(tmp_path), feature_config, publisher=publisher)
    if isinstance(failure, PublishError):
        assert updater.apply(_events()) == 1
    else:
        with pytest.raises(RuntimeError, match="sidecar bug"):
            updater.apply(_events())
    assert publisher.published == []
    assert updater.post_write_pending

    updater.repair_post_write()
    assert not updater.post_write_pending
    [(frame, user_ids)] = publisher.published
    assert "u1" in user_ids
    assert "i9" in set(frame[frame["user_id"] == "u1"]["item_id"].astype(str))


def test_a_failed_repair_backs_off(tmp_path, feature_config: FeatureConfig) -> None:
    calls: list[int] = []

    def _refresh() -> None:
        calls.append(1)
        raise RuntimeError("reader refresh failed")

    _sink, updater = _updater(_settings(tmp_path), feature_config, on_success=_refresh)
    with pytest.raises(RuntimeError, match="reader refresh failed"):
        updater.apply(_events())
    with pytest.raises(RuntimeError):
        updater.repair_post_write()
    updater.repair_post_write()
    assert calls == [1, 1]
    assert updater.post_write_pending


def test_the_worker_repairs_on_a_later_tick_without_new_events(
    tmp_path, feature_config: FeatureConfig
) -> None:
    settings = _settings(tmp_path)
    sink = build_output_sink(settings.output)
    sink.write_manifest, _calls = _fails_once(sink.write_manifest, OSError("disk full"))  # type: ignore[method-assign]
    _sink, updater = _updater(settings, feature_config, sink=sink)
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="repair-1", user_id="u1", item_id="i9"))
    worker = EventWorker(source, MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0), updater)

    assert worker.tick() == 1
    assert source.health().lag == 0
    assert build_manifest_reader(settings.output).read_latest() is None

    assert worker.tick() == 0
    assert not updater.post_write_pending
    assert build_manifest_reader(settings.output).read_latest()["triggered_by"] == "incremental"


def test_the_publish_repair_skips_when_a_full_write_lands_meanwhile(
    tmp_path, feature_config: FeatureConfig
) -> None:
    publisher = _Publisher(fail_with=RuntimeError("sidecar bug"))
    sink, updater = _updater(_settings(tmp_path), feature_config, publisher=publisher)
    with pytest.raises(RuntimeError, match="sidecar bug"):
        updater.apply(_events())
    load_users = updater._load_users

    def _load_then_retrain(user_ids):  # type: ignore[no-untyped-def]
        rows = load_users(user_ids)
        sink.write_manifest({"triggered_by": "job", "generated_at": "2999-01-01T00:00:00+00:00"})
        return rows

    updater._load_users = _load_then_retrain  # type: ignore[method-assign]
    updater.repair_post_write()
    assert publisher.published == []
    assert not updater.post_write_pending


def test_the_worker_counts_a_handled_publish_failure_as_a_flush_error(
    tmp_path, feature_config: FeatureConfig
) -> None:
    publisher = _Publisher(fail_with=PublishError("broker down"))
    _sink, updater = _updater(_settings(tmp_path), feature_config, publisher=publisher)
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="publish-fails", user_id="u1", item_id="i9"))
    worker = EventWorker(source, MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0), updater)
    before_error = registry_metric_value("cicerone_events_flush_total", {"status": "error"})
    before_tick = registry_metric_value("cicerone_events_tick_errors_total")

    assert worker.tick() == 1
    assert source.health().lag == 0
    assert updater.post_write_pending
    assert registry_metric_value("cicerone_events_flush_total", {"status": "error"}) == before_error + 1
    assert registry_metric_value("cicerone_events_tick_errors_total") == before_tick + 1


def test_a_reader_refresh_that_keeps_old_data_is_retried(tmp_path, feature_config: FeatureConfig) -> None:
    results = [False, True]
    _sink, updater = _updater(_settings(tmp_path), feature_config, on_success=lambda: results.pop(0))

    assert updater.apply(_events()) == 1
    assert updater.post_write_pending

    updater.repair_post_write()
    assert results == []
    assert not updater.post_write_pending


def test_a_heartbeat_failure_after_a_written_batch_returns_the_next_batch(
    tmp_path, feature_config: FeatureConfig, monkeypatch
) -> None:
    _sink, updater = _updater(_settings(tmp_path), feature_config)
    source = WebhookEventSource({})
    worker = EventWorker(source, MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0), updater)
    source.ingest(event_payload(event_id="written", user_id="u1", item_id="i9"))
    assert worker.tick() == 1
    assert updater.last_apply_written

    @contextmanager
    def _heartbeat_fails(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise HeartbeatError("lease lost")
        yield

    monkeypatch.setattr("cicerone.events.worker.inflight_heartbeat", _heartbeat_fails)
    source.ingest(event_payload(event_id="not-written", user_id="u1", item_id="i8"))
    assert worker.tick() == 0
    assert source.health().lag == 1


def test_a_repair_that_raises_returns_the_ready_batch(
    tmp_path, feature_config: FeatureConfig, monkeypatch
) -> None:
    _sink, updater = _updater(_settings(tmp_path), feature_config)
    source = WebhookEventSource({})
    worker = EventWorker(source, MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0), updater)
    updater._repair.refresh = True

    def _repair_bug() -> None:
        raise RuntimeError("repair bug")

    monkeypatch.setattr(updater, "repair_post_write", _repair_bug)
    source.ingest(event_payload(event_id="ready", user_id="u1", item_id="i9"))
    with pytest.raises(RuntimeError, match="repair bug"):
        worker.tick()
    assert [event.event_id for event in source.poll(10)] == ["ready"]
