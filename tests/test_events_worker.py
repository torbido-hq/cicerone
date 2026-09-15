from __future__ import annotations

import pandas as pd
import pytest
from support.events import event_payload
from support.prometheus_metrics import registry_metric_value

from cicerone.config import EventsSettings, IOSettings, make_settings
from cicerone.events.base import EventSourceHealth
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.normalize import normalize_event
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


def test_event_worker_records_flush_metrics(tmp_path, feature_config: FeatureConfig):
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
    source.ingest(event_payload(event_id="metrics-1", item_id="i9"))
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
    )
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0),
        updater,
    )

    before = registry_metric_value("cicerone_events_flush_total", {"status": "success"})
    assert worker.tick() == 1
    assert registry_metric_value("cicerone_events_flush_total", {"status": "success"}) == before + 1
    assert registry_metric_value("cicerone_events_flush_events_total") >= 1


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
    before_busy = registry_metric_value("cicerone_events_flush_total", {"status": "busy"})
    assert worker.tick() == 0
    assert source.health().lag == 1
    assert registry_metric_value("cicerone_events_flush_total", {"status": "busy"}) == before_busy + 1


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
        def apply(self, events, *, persist_online: bool = True):  # type: ignore[no-untyped-def]
            del persist_online
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
    before_error = registry_metric_value("cicerone_events_flush_total", {"status": "error"})
    before_tick = registry_metric_value("cicerone_events_tick_errors_total")
    assert worker.tick() == 0
    assert source.health().lag == 1
    assert registry_metric_value("cicerone_events_flush_total", {"status": "error"}) == before_error + 1
    assert registry_metric_value("cicerone_events_tick_errors_total") == before_tick


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
        def apply(self, events, *, persist_online: bool = True):  # type: ignore[no-untyped-def]
            del persist_online
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
    before = registry_metric_value("cicerone_events_flush_total", {"status": "error"})
    assert worker.tick() == 0
    assert source.health().lag == 2
    assert registry_metric_value("cicerone_events_flush_total", {"status": "error"}) == before + 1


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
    closed = {"n": 0}

    class _CloseCount(WebhookEventSource):
        def close(self) -> None:
            closed["n"] += 1
            super().close()

    worker = EventWorker(
        _CloseCount({}),
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
    assert worker._source_guard.acquire(blocking=False)
    try:
        with caplog.at_level(logging.WARNING):
            assert worker.stop(join_timeout_seconds=0.01) is False
        assert any("still alive" in record.getMessage() for record in caplog.records)
        assert closed["n"] == 0
    finally:
        worker._source_guard.release()
    worker._stop.set()


def test_event_worker_stop_closes_on_join_timeout_when_idle(tmp_path, feature_config, caplog):
    import logging

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    closed = {"n": 0}

    class _CloseCount(WebhookEventSource):
        def close(self) -> None:
            closed["n"] += 1
            super().close()

    worker = EventWorker(
        _CloseCount({}),
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
    assert closed["n"] == 1
    worker._stop.set()


def test_event_worker_stop_skips_close_during_apply_ack(tmp_path, feature_config, caplog):
    import logging

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    closed = {"n": 0}

    class _CloseCount(WebhookEventSource):
        def close(self) -> None:
            closed["n"] += 1
            super().close()

    worker = EventWorker(
        _CloseCount({}),
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
    assert worker._tick_guard.acquire(blocking=False)
    assert worker._source_guard.acquire(blocking=False)
    try:
        with caplog.at_level(logging.WARNING):
            assert worker.stop(join_timeout_seconds=0.01) is False
        assert any("still alive" in record.getMessage() for record in caplog.records)
        assert closed["n"] == 0
    finally:
        worker._tick_guard.release()
        worker._source_guard.release()
    worker._stop.set()


def test_event_worker_stop_skips_close_during_poll(tmp_path, feature_config: FeatureConfig):
    import threading
    import time

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    poll_started = threading.Event()
    poll_release = threading.Event()
    order: list[str] = []

    class _GatePoll(WebhookEventSource):
        def poll(self, max_events: int = 100):  # type: ignore[override]
            order.append("poll-start")
            poll_started.set()
            poll_release.wait(timeout=2)
            return super().poll(max_events)

        def close(self) -> None:
            order.append("close")
            super().close()

    worker = EventWorker(
        _GatePoll({}),
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
    assert poll_started.wait(timeout=2)
    assert worker.stop(join_timeout_seconds=0.05) is False
    assert "close" not in order
    poll_release.set()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and "close" not in order:
        time.sleep(0.01)
    assert "poll-start" in order
    assert "close" in order
    assert order.index("poll-start") < order.index("close")


def test_event_worker_stop_skips_close_during_never_started_tick(tmp_path, feature_config: FeatureConfig):
    import threading

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    poll_started = threading.Event()
    poll_release = threading.Event()
    order: list[str] = []

    class _GatePoll(WebhookEventSource):
        def poll(self, max_events: int = 100):  # type: ignore[override]
            poll_started.set()
            poll_release.wait(timeout=2)
            return super().poll(max_events)

        def close(self) -> None:
            order.append("close")
            super().close()

    worker = EventWorker(
        _GatePoll({}),
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )

    def _tick() -> None:
        worker.tick()

    ticker = threading.Thread(target=_tick)
    ticker.start()
    try:
        assert poll_started.wait(timeout=2)
        assert worker.stop(join_timeout_seconds=0.05) is False
        assert "close" not in order
    finally:
        poll_release.set()
        ticker.join(timeout=2)


def test_event_worker_stop_does_not_close_before_in_flight_ack(tmp_path, feature_config: FeatureConfig):
    import threading
    import time

    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    ack_started = threading.Event()
    ack_release = threading.Event()
    order: list[str] = []

    class _GateAck(WebhookEventSource):
        def ack(self, event_ids):  # type: ignore[no-untyped-def,override]
            order.append("ack-start")
            ack_started.set()
            ack_release.wait(timeout=2)
            super().ack(event_ids)
            order.append("ack-done")

        def close(self) -> None:
            order.append("close")
            super().close()

    source = _GateAck({})
    source.ingest(event_payload(event_id="ack-race", item_id="i7"))
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
        poll_interval_seconds=0.01,
    )
    worker.start()
    assert ack_started.wait(timeout=2)
    assert worker.stop(join_timeout_seconds=0.05) is False
    assert "close" not in order
    ack_release.set()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and "close" not in order:
        time.sleep(0.01)
    assert "ack-done" in order
    assert "close" in order
    assert order.index("ack-done") < order.index("close")


def test_event_worker_reconnect_nacks_buffer_after_connect(tmp_path, feature_config: FeatureConfig):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    order: list[str] = []
    nacked: list[str] = []

    class _TrackNack(WebhookEventSource):
        def connect(self) -> None:
            order.append("connect")
            super().connect()

        def nack(self, events):  # type: ignore[no-untyped-def,override]
            order.append("nack")
            nacked.extend(event.event_id for event in events)
            return super().nack(events)

    source = _TrackNack({})
    source.connect()
    source.ingest(event_payload(event_id="buf-1", item_id="i7"))
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    kept = worker._buffer.extend(source.poll(1))
    assert kept.kept_count == 1
    assert worker._reconnect_source() is True
    assert nacked == ["buf-1"]
    assert len(worker._buffer) == 0
    assert [event.event_id for event in source.poll(1)] == ["buf-1"]
    assert order == ["connect", "connect", "nack"]


def test_event_worker_reconnect_restores_buffer_when_nack_fails(tmp_path, feature_config: FeatureConfig):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )

    class _NackBoom(WebhookEventSource):
        def nack(self, events):  # type: ignore[no-untyped-def,override]
            raise RuntimeError("nack unavailable")

    source = _NackBoom({})
    source.connect()
    source.ingest(event_payload(event_id="buf-2", item_id="i8"))
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    kept = worker._buffer.extend(source.poll(1))
    assert kept.kept_count == 1
    assert worker._reconnect_source() is True
    assert [event.event_id for event in worker._buffer.flush()] == ["buf-2"]


def test_event_worker_reconnect_keeps_buffer_when_nack_is_noop(tmp_path, feature_config: FeatureConfig):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )

    class _SilentNack(WebhookEventSource):
        def connect(self) -> None:
            super().connect()
            with self._lock:
                self._pending.clear()
                self._pending_ids.clear()
                self._in_flight.clear()

        def nack(self, events):  # type: ignore[no-untyped-def,override]
            return list(events)

    source = _SilentNack({})
    source.connect()
    source.ingest(event_payload(event_id="buf-4", item_id="i10"))
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    kept = worker._buffer.extend(source.poll(1))
    assert kept.kept_count == 1
    assert worker._reconnect_source() is True
    assert [event.event_id for event in worker._buffer.flush()] == ["buf-4"]


def test_event_worker_reconnect_restores_buffer_when_connect_fails(tmp_path, feature_config: FeatureConfig):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    connects = {"n": 0}

    class _ConnectBoom(WebhookEventSource):
        def connect(self) -> None:
            connects["n"] += 1
            if connects["n"] > 1:
                raise RuntimeError("reconnect refused")
            super().connect()

    source = _ConnectBoom({})
    source.connect()
    source.ingest(event_payload(event_id="buf-3", item_id="i9"))
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    kept = worker._buffer.extend(source.poll(1))
    assert kept.kept_count == 1
    assert worker._reconnect_source() is False
    assert [event.event_id for event in worker._buffer.flush()] == ["buf-3"]


def test_event_worker_flush_restores_rejected_nack(tmp_path, feature_config: FeatureConfig):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )

    class _RejectNack(WebhookEventSource):
        def heartbeat(self, events):  # type: ignore[no-untyped-def]
            del events
            raise RuntimeError("heartbeat failed")

        def nack(self, events):  # type: ignore[no-untyped-def,override]
            return list(events)

    source = _RejectNack({})
    source.connect()
    source.ingest(event_payload(event_id="rej-1", item_id="i11"))
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    kept = worker._buffer.extend(source.poll(1))
    ready = worker._buffer.flush()
    assert kept.kept_count == 1
    assert worker._flush_ready(ready) == 0
    assert [event.event_id for event in worker._buffer.flush()] == ["rej-1"]


def test_event_worker_reconnect_holds_tick_guard(tmp_path, feature_config: FeatureConfig):
    import threading

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    started = threading.Event()
    release = threading.Event()

    class _SlowReconnect(WebhookEventSource):
        def connect(self) -> None:
            self._opens = getattr(self, "_opens", 0) + 1
            if self._opens > 1:
                started.set()
                release.wait(timeout=2)
            super().connect()

    source = _SlowReconnect({})
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    source.connect()
    source.ingest(event_payload(event_id="tick-1", item_id="i12"))
    worker._buffer.extend(source.poll(1))

    def _reconnect() -> None:
        worker._reconnect_source()

    thread = threading.Thread(target=_reconnect)
    thread.start()
    try:
        assert started.wait(timeout=2)
        assert worker._tick_guard.acquire(blocking=False) is False
    finally:
        release.set()
        thread.join(timeout=2)


def test_event_worker_start_health_holds_tick_guard(tmp_path, feature_config: FeatureConfig):
    import threading

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    started = threading.Event()
    release = threading.Event()

    class _SlowHealth(WebhookEventSource):
        def health(self) -> EventSourceHealth:
            started.set()
            release.wait(timeout=2)
            return super().health()

    worker = EventWorker(
        _SlowHealth({}),
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
        poll_interval_seconds=0.01,
    )
    starter = threading.Thread(target=worker.start)
    starter.start()
    try:
        assert started.wait(timeout=2)
        assert worker._tick_guard.acquire(blocking=False) is False
        assert worker._source_guard.acquire(blocking=False) is False
    finally:
        release.set()
        starter.join(timeout=2)
        worker.stop(join_timeout_seconds=2.0)


def test_event_worker_remembers_numeric_event_id_after_ack_timeout(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    applies: list[str] = []
    event = normalize_event(event_payload(event_id="42", item_id="inum"))

    class _AckTimeout:
        def __init__(self) -> None:
            self._pending = [event]
            self._acks = 0

        def connect(self) -> None:
            return None

        def poll(self, max_events: int = 100):  # type: ignore[no-untyped-def]
            del max_events
            return list(self._pending)

        def ack(self, event_ids):  # type: ignore[no-untyped-def]
            del event_ids
            self._acks += 1
            if self._acks == 1:
                raise TimeoutError("ack timed out")
            self._pending = []

        def nack(self, events):  # type: ignore[no-untyped-def]
            return list(events)

        def health(self) -> EventSourceHealth:
            return EventSourceHealth(connected=True, lag=len(self._pending))

    class _CountApply(IncrementalUpdater):
        def apply(self, events, *, persist_online: bool = True):  # type: ignore[no-untyped-def,override]
            applies.extend(item.event_id for item in events)
            return super().apply(events, persist_online=persist_online)

    worker = EventWorker(
        _AckTimeout(),  # type: ignore[arg-type]
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0),
        _CountApply(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    assert worker.tick() == 1
    assert applies == ["42"]
    assert worker.tick() == 0
    assert applies == ["42"]


def test_event_worker_ack_timeout_persists_without_nack(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    nacked: list[str] = []

    class _AckTimeout(WebhookEventSource):
        def ack(self, event_ids):  # type: ignore[no-untyped-def,override]
            raise TimeoutError("ack timed out")

        def nack(self, events):  # type: ignore[no-untyped-def,override]
            nacked.extend(event.event_id for event in events)
            return super().nack(events)

    source = _AckTimeout({})
    source.ingest(event_payload(event_id="ack-to", item_id="i13"))
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    assert worker.tick() == 1
    assert nacked == []
    frame = pd.read_parquet(out / "recommendations.parquet")
    assert "i13" in set(frame["item_id"].astype(str))


def test_event_worker_retries_failed_post_apply_ack(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    applies: list[str] = []
    event = normalize_event(event_payload(event_id="ack-retry", item_id="iretry"))

    class _FailFirstAck:
        def __init__(self) -> None:
            self._pending = [event]
            self.acked: list[str] = []

        def connect(self) -> None:
            return None

        def poll(self, max_events: int = 100):  # type: ignore[no-untyped-def]
            del max_events
            return list(self._pending)

        def ack(self, event_ids):  # type: ignore[no-untyped-def]
            self.acked.extend(str(event_id) for event_id in event_ids)
            if len(self.acked) == 1:
                raise RuntimeError("commit failed")
            self._pending = []

        def nack(self, events):  # type: ignore[no-untyped-def]
            return list(events)

        def health(self) -> EventSourceHealth:
            return EventSourceHealth(connected=True, lag=len(self._pending))

    class _CountApply(IncrementalUpdater):
        def apply(self, events, *, persist_online: bool = True):  # type: ignore[no-untyped-def,override]
            applies.extend(item.event_id for item in events)
            return super().apply(events, persist_online=persist_online)

    source = _FailFirstAck()
    worker = EventWorker(
        source,  # type: ignore[arg-type]
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0),
        _CountApply(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    with pytest.raises(RuntimeError, match="commit failed"):
        worker.tick()
    assert applies == ["ack-retry"]
    assert worker.tick() == 0
    assert applies == ["ack-retry"]
    assert source.acked == ["ack-retry", "ack-retry"]
    assert source._pending == []


def test_event_worker_retries_failed_unbuffered_ack(tmp_path, feature_config: FeatureConfig):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    event = normalize_event(event_payload(event_id="already-1", item_id="idone"))
    acked: list[str] = []

    class _FailUnbuffered:
        def connect(self) -> None:
            return None

        def poll(self, max_events: int = 100):  # type: ignore[no-untyped-def]
            del max_events
            return [event]

        def ack(self, event_ids):  # type: ignore[no-untyped-def]
            acked.extend(str(event_id) for event_id in event_ids)
            if len(acked) == 1:
                raise RuntimeError("unbuffered ack failed")

        def nack(self, events):  # type: ignore[no-untyped-def]
            return list(events)

        def health(self) -> EventSourceHealth:
            return EventSourceHealth(connected=True, lag=1)

    worker = EventWorker(
        _FailUnbuffered(),  # type: ignore[arg-type]
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    worker._remember_applied([event])
    with pytest.raises(RuntimeError, match="unbuffered ack failed"):
        worker._poll_into_buffer()
    worker._flush_retry_acks()
    assert acked == ["already-1", "already-1"]


def test_event_worker_retry_acks_deferred_duplicates(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    original = normalize_event(event_payload(event_id="retry-fp-1", item_id="ifp"))
    duplicate = normalize_event(event_payload(event_id="retry-fp-2", item_id="ifp"))
    acked: list[str] = []

    class _FailApplyAck:
        def __init__(self) -> None:
            self._pending = [original, duplicate]

        def connect(self) -> None:
            return None

        def poll(self, max_events: int = 100):  # type: ignore[no-untyped-def]
            del max_events
            batch = list(self._pending)
            self._pending = []
            return batch

        def ack(self, event_ids):  # type: ignore[no-untyped-def]
            acked.extend(str(event_id) for event_id in event_ids)
            if acked == ["retry-fp-1"]:
                raise RuntimeError("apply ack failed")

        def nack(self, events):  # type: ignore[no-untyped-def]
            return list(events)

        def health(self) -> EventSourceHealth:
            return EventSourceHealth(connected=True, lag=0)

    worker = EventWorker(
        _FailApplyAck(),  # type: ignore[arg-type]
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0, max_events=10),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    with pytest.raises(RuntimeError, match="apply ack failed"):
        worker.tick()
    assert "retry-fp-2" not in acked
    worker.tick()
    assert "retry-fp-1" in acked
    assert "retry-fp-2" in acked


def test_event_worker_skips_poll_when_startup_health_disconnected(tmp_path, feature_config: FeatureConfig):
    import time

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    connects = {"n": 0}
    polls_at_connect: list[int] = []

    class _UnhealthyUntilReconnect(WebhookEventSource):
        def connect(self) -> None:
            connects["n"] += 1
            super().connect()

        def poll(self, max_events: int = 100):  # type: ignore[override]
            polls_at_connect.append(connects["n"])
            return super().poll(max_events)

        def health(self) -> EventSourceHealth:
            return EventSourceHealth(connected=connects["n"] >= 2, lag=0)

    worker = EventWorker(
        _UnhealthyUntilReconnect({}),
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
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not polls_at_connect:
        time.sleep(0.01)
    worker.stop(join_timeout_seconds=2.0)
    assert polls_at_connect
    assert polls_at_connect[0] >= 2


def test_event_worker_reconnects_when_source_reports_disconnected(tmp_path, feature_config: FeatureConfig):
    import time

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    connects = {"n": 0}

    class _Flaky(WebhookEventSource):
        def connect(self) -> None:
            connects["n"] += 1
            super().connect()

        def health(self) -> EventSourceHealth:
            return EventSourceHealth(connected=connects["n"] >= 2, lag=0)

    worker = EventWorker(
        _Flaky({}),
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
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and connects["n"] < 2:
        time.sleep(0.01)
    worker.stop(join_timeout_seconds=2.0)
    assert connects["n"] >= 2


def test_event_worker_stop_closes_reconnect_in_progress(tmp_path, feature_config: FeatureConfig):
    import threading
    import time

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    reconnect_started = threading.Event()
    connects = {"n": 0}
    closes = {"n": 0}

    class _SlowReconnect(WebhookEventSource):
        def connect(self) -> None:
            connects["n"] += 1
            if connects["n"] >= 2:
                reconnect_started.set()
                time.sleep(1.0)
            super().connect()

        def close(self) -> None:
            closes["n"] += 1
            super().close()

        def health(self) -> EventSourceHealth:
            return EventSourceHealth(connected=False, lag=0)

    worker = EventWorker(
        _SlowReconnect({}),
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
    assert reconnect_started.wait(timeout=2)
    began = time.monotonic()
    assert worker.stop(join_timeout_seconds=0.05) is False
    assert time.monotonic() - began < 0.4
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and closes["n"] < 1:
        time.sleep(0.01)
    assert closes["n"] >= 1


def test_event_worker_stop_does_not_block_on_dead_thread_during_restart(
    tmp_path, feature_config: FeatureConfig
):
    import threading
    import time

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    started = threading.Event()
    release = threading.Event()

    class _SlowSecondConnect(WebhookEventSource):
        def connect(self) -> None:
            if started.is_set():
                release.wait(timeout=2)
                return
            started.set()
            super().connect()

    worker = EventWorker(
        _SlowSecondConnect({}),
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
    assert worker._thread is not None and worker._thread.is_alive() is False
    starter = threading.Thread(target=worker.start)
    starter.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and worker._source_guard.acquire(blocking=False):
        worker._source_guard.release()
        time.sleep(0.01)
    began = time.monotonic()
    assert worker.stop(join_timeout_seconds=0.05) is False
    assert time.monotonic() - began < 0.4
    release.set()
    starter.join(timeout=2)


def test_event_worker_start_never_acquires_tick_while_holding_source(tmp_path, feature_config: FeatureConfig):
    import threading

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    owned = threading.local()
    inversions: list[str] = []

    class _OrderLock:
        def __init__(self, name: str) -> None:
            self.name = name
            self._inner = threading.Lock()

        def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
            if self.name == "tick" and getattr(owned, "source", False):
                inversions.append("tick-while-source")
            got = self._inner.acquire(blocking, timeout)
            if got and self.name == "source":
                owned.source = True
            return got

        def release(self) -> None:
            if self.name == "source":
                owned.source = False
            self._inner.release()

        def __enter__(self) -> _OrderLock:
            self.acquire()
            return self

        def __exit__(self, *_exc: object) -> None:
            self.release()

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
    worker._source_guard = _OrderLock("source")  # type: ignore[assignment]
    worker._tick_guard = _OrderLock("tick")  # type: ignore[assignment]
    worker.start()
    assert inversions == []
    assert worker.stop(join_timeout_seconds=2.0) is True
    assert inversions == []


def test_event_worker_start_does_not_revive_after_completed_stop(tmp_path, feature_config: FeatureConfig):
    import threading

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    inner = threading.Lock()
    started = threading.Event()
    release = threading.Event()
    starter_thread: list[threading.Thread] = []

    class _GateLock:
        def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
            if starter_thread and threading.current_thread() is starter_thread[0]:
                started.set()
                release.wait(timeout=2)
            return inner.acquire(blocking, timeout)

        def release(self) -> None:
            inner.release()

        def __enter__(self) -> _GateLock:
            self.acquire()
            return self

        def __exit__(self, *_exc: object) -> None:
            self.release()

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
    worker._source_guard = _GateLock()  # type: ignore[assignment]

    def _start() -> None:
        starter_thread.append(threading.current_thread())
        worker.start()

    starter = threading.Thread(target=_start)
    starter.start()
    assert started.wait(timeout=2)
    assert worker.stop(join_timeout_seconds=0.05) is True
    release.set()
    starter.join(timeout=2)
    assert worker._thread is None or worker._thread.is_alive() is False
    worker.start()
    assert worker._thread is not None and worker._thread.is_alive()
    assert worker.stop(join_timeout_seconds=2.0) is True


def test_event_worker_start_does_not_reacquire_guard_after_launch(tmp_path, feature_config: FeatureConfig):
    import threading

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    inner = threading.Lock()
    starter_thread: list[threading.Thread] = []
    released = threading.Event()
    block_second = threading.Event()

    class _GateLock:
        def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
            if (
                starter_thread
                and threading.current_thread() is starter_thread[0]
                and worker._thread is not None
                and worker._thread.is_alive()
            ):
                block_second.wait(timeout=2)
            return inner.acquire(blocking, timeout)

        def release(self) -> None:
            inner.release()
            if starter_thread and threading.current_thread() is starter_thread[0]:
                released.set()

        def __enter__(self) -> _GateLock:
            self.acquire()
            return self

        def __exit__(self, *_exc: object) -> None:
            self.release()

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
    worker._source_guard = _GateLock()  # type: ignore[assignment]

    def _start() -> None:
        starter_thread.append(threading.current_thread())
        worker.start()

    starter = threading.Thread(target=_start)
    starter.start()
    assert released.wait(timeout=2)
    starter.join(timeout=0.4)
    assert starter.is_alive() is False
    assert worker._thread is not None and worker._thread.is_alive()
    assert worker.stop(join_timeout_seconds=2.0) is True
    worker.start()
    assert worker._thread is not None and worker._thread.is_alive()
    assert worker.stop(join_timeout_seconds=2.0) is True


def test_event_worker_start_aborts_if_stopped_during_connect(tmp_path, feature_config: FeatureConfig):
    import threading
    import time

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    started = threading.Event()
    release = threading.Event()
    closes = {"n": 0}

    class _SlowStart(WebhookEventSource):
        def connect(self) -> None:
            started.set()
            release.wait(timeout=2)
            super().connect()

        def close(self) -> None:
            closes["n"] += 1
            super().close()

    worker = EventWorker(
        _SlowStart({}),
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
        poll_interval_seconds=0.01,
    )
    starter = threading.Thread(target=worker.start)
    starter.start()
    assert started.wait(timeout=2)
    began = time.monotonic()
    assert worker.stop(join_timeout_seconds=0.05) is False
    assert time.monotonic() - began < 0.4
    release.set()
    starter.join(timeout=2)
    assert worker._thread is None or worker._thread.is_alive() is False
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and closes["n"] < 1:
        time.sleep(0.01)
    assert closes["n"] >= 1


def test_event_worker_start_abort_drains_and_closes_once(tmp_path, feature_config: FeatureConfig):
    import threading
    import time

    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    started = threading.Event()
    release = threading.Event()
    closes = {"n": 0}

    class _SlowStart(WebhookEventSource):
        def connect(self) -> None:
            started.set()
            release.wait(timeout=2)
            super().connect()

        def close(self) -> None:
            closes["n"] += 1
            super().close()

    source = _SlowStart({})
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
        poll_interval_seconds=0.01,
    )
    worker._buffer.extend([normalize_event(event_payload(event_id="drain-start", item_id="idrain"))])
    starter = threading.Thread(target=worker.start)
    starter.start()
    assert started.wait(timeout=2)
    assert worker.stop(join_timeout_seconds=0.05) is False
    release.set()
    starter.join(timeout=2)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and closes["n"] < 1:
        time.sleep(0.01)
    assert closes["n"] == 1
    assert worker._finalized is True
    frame = pd.read_parquet(out / "recommendations.parquet")
    assert "idrain" in set(frame["item_id"].astype(str))
    assert worker.stop(join_timeout_seconds=0.05) is True
    assert closes["n"] == 1


def test_event_worker_start_abort_drains_when_connect_raises(tmp_path, feature_config: FeatureConfig):
    import threading
    import time

    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    started = threading.Event()
    release = threading.Event()
    closes = {"n": 0}
    errors: list[BaseException] = []

    class _BoomStart(WebhookEventSource):
        def connect(self) -> None:
            started.set()
            release.wait(timeout=2)
            raise RuntimeError("connect failed")

        def close(self) -> None:
            closes["n"] += 1
            super().close()

    source = _BoomStart({})
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
        poll_interval_seconds=0.01,
    )
    worker._buffer.extend([normalize_event(event_payload(event_id="drain-boom", item_id="idrain"))])

    def _start() -> None:
        try:
            worker.start()
        except Exception as exc:
            errors.append(exc)

    starter = threading.Thread(target=_start)
    starter.start()
    assert started.wait(timeout=2)
    assert worker.stop(join_timeout_seconds=0.05) is False
    release.set()
    starter.join(timeout=2)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and closes["n"] < 1:
        time.sleep(0.01)
    assert errors
    assert closes["n"] == 1
    assert worker._finalized is True
    frame = pd.read_parquet(out / "recommendations.parquet")
    assert "idrain" in set(frame["item_id"].astype(str))


def test_event_worker_reconnect_closes_after_failed_connect(tmp_path, feature_config: FeatureConfig):
    import threading
    import time

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    reconnect_started = threading.Event()
    closes = {"n": 0}
    connects = {"n": 0}

    class _FailReconnect(WebhookEventSource):
        def connect(self) -> None:
            connects["n"] += 1
            if connects["n"] >= 2:
                reconnect_started.set()
                time.sleep(0.2)
                raise RuntimeError("broker down")
            super().connect()

        def close(self) -> None:
            closes["n"] += 1
            super().close()

        def health(self) -> EventSourceHealth:
            return EventSourceHealth(connected=False, lag=0)

    worker = EventWorker(
        _FailReconnect({}),
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
    assert reconnect_started.wait(timeout=2)
    worker.stop(join_timeout_seconds=0.05)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and closes["n"] < 1:
        time.sleep(0.01)
    assert closes["n"] >= 1


def test_event_worker_skips_tick_after_failed_reconnect(tmp_path, feature_config: FeatureConfig):
    import threading
    import time

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    reconnect_started = threading.Event()
    polls = {"n": 0}

    class _FailReconnect(WebhookEventSource):
        def connect(self) -> None:
            if polls["n"] > 0:
                reconnect_started.set()
                raise RuntimeError("broker down")
            super().connect()

        def poll(self, max_events: int = 100):  # type: ignore[override]
            polls["n"] += 1
            return super().poll(max_events)

        def health(self) -> EventSourceHealth:
            return EventSourceHealth(connected=False, lag=0)

    worker = EventWorker(
        _FailReconnect({}),
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
        poll_interval_seconds=0.02,
    )
    worker.start()
    assert reconnect_started.wait(timeout=2)
    after = polls["n"]
    time.sleep(0.08)
    assert polls["n"] == after
    worker.stop(join_timeout_seconds=1.0)


def test_event_worker_stop_closes_source_once(tmp_path, feature_config: FeatureConfig):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    closed = {"n": 0}

    class _CloseCount(WebhookEventSource):
        def close(self) -> None:
            closed["n"] += 1
            super().close()

    worker = EventWorker(
        _CloseCount({}),
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
    assert closed["n"] == 1


def test_event_worker_stop_waits_for_health_before_close(tmp_path, feature_config: FeatureConfig):
    import threading
    import time

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    health_started = threading.Event()
    health_release = threading.Event()
    order: list[str] = []

    class _SlowHealth(WebhookEventSource):
        def health(self) -> EventSourceHealth:
            order.append("health-start")
            health_started.set()
            health_release.wait(timeout=2)
            order.append("health-end")
            return super().health()

        def close(self) -> None:
            order.append("close")

    worker = EventWorker(
        _SlowHealth({}),
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
        poll_interval_seconds=0.01,
    )
    starter = threading.Thread(target=worker.start)
    starter.start()
    assert health_started.wait(timeout=2)
    assert worker.stop(join_timeout_seconds=0.05) is False
    assert "close" not in order
    health_release.set()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and "close" not in order:
        time.sleep(0.01)
    assert "health-start" in order
    assert "close" in order
    assert order.index("health-start") < order.index("close")
    starter.join(timeout=2)


def test_event_worker_start_returns_before_first_poll(tmp_path, feature_config: FeatureConfig):
    import threading

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    healths = {"n": 0}
    poll_started = threading.Event()
    poll_release = threading.Event()

    class _GatePoll(WebhookEventSource):
        def health(self) -> EventSourceHealth:
            healths["n"] += 1
            return super().health()

        def poll(self, max_events: int = 100):  # type: ignore[override]
            poll_started.set()
            poll_release.wait(timeout=5)
            return super().poll(max_events)

    worker = EventWorker(
        _GatePoll({}),
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
        poll_interval_seconds=0.01,
    )
    starter = threading.Thread(target=worker.start)
    starter.start()
    starter.join(timeout=1)
    assert not starter.is_alive()
    assert healths["n"] >= 1
    assert poll_started.wait(timeout=2)
    poll_release.set()
    assert worker.stop(join_timeout_seconds=2.0) is True


def test_event_worker_tick_skips_when_stopped(tmp_path, feature_config: FeatureConfig):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    polled = {"n": 0}

    class _CountPoll(WebhookEventSource):
        def poll(self, max_events: int = 100):  # type: ignore[override]
            polled["n"] += 1
            return super().poll(max_events)

    worker = EventWorker(
        _CountPoll({}),
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    worker._stop.set()
    assert worker.tick() == 0
    assert polled["n"] == 0


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


def test_event_worker_does_not_ack_fingerprint_redelivery_while_buffered(
    tmp_path, feature_config: FeatureConfig
):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    original = normalize_event(event_payload(event_id="pending-fp-1", item_id="ipend"))
    redelivery = normalize_event(event_payload(event_id="pending-fp-2", item_id="ipend"))
    acked: list[str] = []

    class _Replay:
        def connect(self) -> None:
            return None

        def poll(self, max_events: int = 100):  # type: ignore[no-untyped-def]
            del max_events
            return [redelivery]

        def ack(self, event_ids):  # type: ignore[no-untyped-def]
            acked.extend(str(event_id) for event_id in event_ids)

        def nack(self, events):  # type: ignore[no-untyped-def]
            return list(events)

        def health(self) -> EventSourceHealth:
            return EventSourceHealth(connected=True, lag=1)

    worker = EventWorker(
        _Replay(),  # type: ignore[arg-type]
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    worker._buffer.extend([original])
    worker._poll_into_buffer()
    assert acked == []
    assert [item.event_id for item in worker._buffer.flush()] == ["pending-fp-1"]


def test_event_worker_acks_fingerprint_redelivery_after_apply(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    applies: list[str] = []
    payload = event_payload(item_id="ifp")
    payload.pop("event_id")
    original = normalize_event(payload)
    redelivery = normalize_event(payload)
    assert original.generated_event_id is True
    assert redelivery.generated_event_id is True
    assert original.event_id != redelivery.event_id

    class _Redeliver:
        ephemeral_event_ids = True

        def __init__(self) -> None:
            self._pending = [original]
            self.acked: list[str] = []

        def connect(self) -> None:
            return None

        def poll(self, max_events: int = 100):  # type: ignore[no-untyped-def]
            del max_events
            return list(self._pending)

        def ack(self, event_ids):  # type: ignore[no-untyped-def]
            self.acked.extend(str(event_id) for event_id in event_ids)
            self._pending = []

        def nack(self, events):  # type: ignore[no-untyped-def]
            return list(events)

        def health(self) -> EventSourceHealth:
            return EventSourceHealth(connected=True, lag=len(self._pending))

    class _CountApply(IncrementalUpdater):
        def apply(self, events, *, persist_online: bool = True):  # type: ignore[no-untyped-def,override]
            applies.extend(item.event_id for item in events)
            return super().apply(events, persist_online=persist_online)

    source = _Redeliver()
    worker = EventWorker(
        source,  # type: ignore[arg-type]
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0),
        _CountApply(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    assert worker.tick() == 1
    source._pending = [redelivery]
    assert worker.tick() == 0
    assert applies == [original.event_id]
    assert redelivery.event_id in source.acked


def test_event_worker_applies_same_fingerprint_with_new_id(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    applies: list[str] = []
    first = normalize_event(event_payload(event_id="shape-1", item_id="ishape"))
    second = normalize_event(event_payload(event_id="shape-2", item_id="ishape"))

    class _TwoShots:
        def __init__(self) -> None:
            self._pending = [first]
            self.acked: list[str] = []

        def connect(self) -> None:
            return None

        def poll(self, max_events: int = 100):  # type: ignore[no-untyped-def]
            del max_events
            return list(self._pending)

        def ack(self, event_ids):  # type: ignore[no-untyped-def]
            self.acked.extend(str(event_id) for event_id in event_ids)
            self._pending = []

        def nack(self, events):  # type: ignore[no-untyped-def]
            return list(events)

        def health(self) -> EventSourceHealth:
            return EventSourceHealth(connected=True, lag=len(self._pending))

    class _CountApply(IncrementalUpdater):
        def apply(self, events, *, persist_online: bool = True):  # type: ignore[no-untyped-def,override]
            applies.extend(item.event_id for item in events)
            return super().apply(events, persist_online=persist_online)

    source = _TwoShots()
    worker = EventWorker(
        source,  # type: ignore[arg-type]
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0),
        _CountApply(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    assert worker.tick() == 1
    source._pending = [second]
    assert worker.tick() == 1
    assert applies == ["shape-1", "shape-2"]


def test_event_worker_applies_same_fingerprint_with_explicit_ids_on_ephemeral_source(
    tmp_path, feature_config: FeatureConfig
):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    applies: list[str] = []
    first = normalize_event(event_payload(event_id="explicit-1", item_id="ishape"))
    second = normalize_event(event_payload(event_id="explicit-2", item_id="ishape"))

    class _TwoShots:
        ephemeral_event_ids = True

        def __init__(self) -> None:
            self._pending = [first]
            self.acked: list[str] = []

        def connect(self) -> None:
            return None

        def poll(self, max_events: int = 100):  # type: ignore[no-untyped-def]
            del max_events
            return list(self._pending)

        def ack(self, event_ids):  # type: ignore[no-untyped-def]
            self.acked.extend(str(event_id) for event_id in event_ids)
            self._pending = []

        def nack(self, events):  # type: ignore[no-untyped-def]
            return list(events)

        def health(self) -> EventSourceHealth:
            return EventSourceHealth(connected=True, lag=len(self._pending))

    class _CountApply(IncrementalUpdater):
        def apply(self, events, *, persist_online: bool = True):  # type: ignore[no-untyped-def,override]
            applies.extend(item.event_id for item in events)
            return super().apply(events, persist_online=persist_online)

    source = _TwoShots()
    worker = EventWorker(
        source,  # type: ignore[arg-type]
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0),
        _CountApply(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    assert worker.tick() == 1
    source._pending = [second]
    assert worker.tick() == 1
    assert applies == ["explicit-1", "explicit-2"]


def test_event_worker_acks_buffer_duplicates(tmp_path, feature_config: FeatureConfig):
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
    source.ingest(event_payload(event_id="dup-a", item_id="i1"))
    source.ingest(event_payload(event_id="dup-b", item_id="i1"))  # same fingerprint
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    assert worker.tick() == 0  # window fills buffer; not ready until size/window
    assert source.health().lag == 2  # fingerprint duplicate stays unacked while original is buffered


def test_event_worker_acks_deferred_fingerprint_duplicate_after_apply(
    tmp_path, feature_config: FeatureConfig
):
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
    source.ingest(event_payload(event_id="dup-c", item_id="i1"))
    source.ingest(event_payload(event_id="dup-d", item_id="i1"))
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0, max_events=10),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    assert worker.tick() == 1
    assert source.health().lag == 0


def test_event_worker_restores_rejected_overflow(tmp_path, feature_config: FeatureConfig):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )

    class _RejectOverflow(WebhookEventSource):
        def nack(self, events):  # type: ignore[no-untyped-def,override]
            return list(events)

    source = _RejectOverflow({})
    source.connect()
    for i in range(2):
        source.ingest(event_payload(event_id=f"ov-rej-{i}", item_id=f"i{i}"))
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0, max_events=1),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    worker._poll_into_buffer()
    assert [event.event_id for event in worker._buffer.flush()] == ["ov-rej-0"]
    assert [event.event_id for event in worker._held] == ["ov-rej-1"]
    worker._poll_into_buffer()
    assert [event.event_id for event in worker._buffer.flush()] == ["ov-rej-1"]
    assert worker._held == []


def test_event_worker_does_not_ack_buffered_redelivery(tmp_path, feature_config: FeatureConfig):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    event = normalize_event(event_payload(event_id="pending-1", item_id="ipend"))
    acked: list[str] = []

    class _Replay:
        def connect(self) -> None:
            return None

        def poll(self, max_events: int = 100):  # type: ignore[no-untyped-def]
            del max_events
            return [event]

        def ack(self, event_ids):  # type: ignore[no-untyped-def]
            acked.extend(str(event_id) for event_id in event_ids)

        def nack(self, events):  # type: ignore[no-untyped-def]
            return list(events)

        def health(self) -> EventSourceHealth:
            return EventSourceHealth(connected=True, lag=1)

    worker = EventWorker(
        _Replay(),  # type: ignore[arg-type]
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    worker._buffer.extend([event])
    worker._poll_into_buffer()
    assert acked == []
    assert [item.event_id for item in worker._buffer.flush()] == ["pending-1"]


def test_event_worker_acks_redelivery_after_applied_restore(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    applies: list[str] = []
    event = normalize_event(event_payload(event_id="rebind-1", item_id="irebind"))

    class _Redeliver:
        def __init__(self) -> None:
            self._pending = [event]
            self.acked: list[str] = []

        def connect(self) -> None:
            return None

        def poll(self, max_events: int = 100):  # type: ignore[no-untyped-def]
            del max_events
            return list(self._pending)

        def ack(self, event_ids):  # type: ignore[no-untyped-def]
            self.acked.extend(str(event_id) for event_id in event_ids)
            self._pending = []

        def nack(self, events):  # type: ignore[no-untyped-def]
            return list(events)

        def health(self) -> EventSourceHealth:
            return EventSourceHealth(connected=True, lag=len(self._pending))

    class _CountApply(IncrementalUpdater):
        def apply(self, events, *, persist_online: bool = True):  # type: ignore[no-untyped-def,override]
            applies.extend(item.event_id for item in events)
            return super().apply(events, persist_online=persist_online)

    source = _Redeliver()
    worker = EventWorker(
        source,  # type: ignore[arg-type]
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0),
        _CountApply(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    worker._buffer.extend([event])
    assert worker._reconnect_source() is True
    assert worker.tick() == 1
    source._pending = [event]
    assert worker.tick() == 0
    assert applies == ["rebind-1"]
    assert "rebind-1" in source.acked


def test_event_worker_stop_does_not_drain_newer_start(tmp_path, feature_config: FeatureConfig):
    import threading
    import time

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    inner = threading.Lock()
    stop_thread: list[threading.Thread] = []
    started_second = threading.Event()

    class _GateLock:
        def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
            if stop_thread and threading.current_thread() is stop_thread[0]:
                started_second.wait(timeout=2)
            return inner.acquire(blocking, timeout)

        def release(self) -> None:
            inner.release()

        def __enter__(self) -> _GateLock:
            self.acquire()
            return self

        def __exit__(self, *_exc: object) -> None:
            self.release()

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
    worker._source_guard = _GateLock()  # type: ignore[assignment]
    worker.start()
    first = worker._thread
    assert first is not None and first.is_alive()

    def _stop() -> None:
        stop_thread.append(threading.current_thread())
        worker.stop(join_timeout_seconds=2.0)

    stopper = threading.Thread(target=_stop)
    stopper.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and first.is_alive():
        time.sleep(0.01)
    assert first.is_alive() is False
    worker.start()
    second = worker._thread
    assert second is not None and second is not first and second.is_alive()
    started_second.set()
    stopper.join(timeout=2)
    assert worker._thread is second and second.is_alive()
    assert worker.stop(join_timeout_seconds=2.0) is True


def test_event_worker_stop_dead_thread_does_not_drain_newer_start(tmp_path, feature_config: FeatureConfig):
    import threading

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    inner = threading.Lock()
    stop_thread: list[threading.Thread] = []
    captured = threading.Event()
    started_second = threading.Event()

    class _GateLock:
        def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
            if stop_thread and threading.current_thread() is stop_thread[0]:
                captured.set()
                started_second.wait(timeout=2)
            return inner.acquire(blocking, timeout)

        def release(self) -> None:
            inner.release()

        def __enter__(self) -> _GateLock:
            self.acquire()
            return self

        def __exit__(self, *_exc: object) -> None:
            self.release()

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
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    worker._thread = dead
    worker._tick_guard = _GateLock()  # type: ignore[assignment]

    def _stop() -> None:
        stop_thread.append(threading.current_thread())
        worker.stop(join_timeout_seconds=2.0)

    stopper = threading.Thread(target=_stop)
    stopper.start()
    assert captured.wait(timeout=2)
    worker.start()
    second = worker._thread
    assert second is not None and second is not dead and second.is_alive()
    started_second.set()
    stopper.join(timeout=2)
    assert worker._thread is second and second.is_alive()
    assert worker.stop(join_timeout_seconds=2.0) is True


def test_event_worker_start_abort_after_launch_does_not_reacquire_locks(
    tmp_path, feature_config: FeatureConfig
):
    import threading
    import time

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    closes = {"n": 0}

    class _CountClose(WebhookEventSource):
        def close(self) -> None:
            closes["n"] += 1

    worker = EventWorker(
        _CountClose({}),
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
        poll_interval_seconds=0.01,
    )
    original_start = threading.Thread.start

    def _start(self: threading.Thread) -> None:
        if self.name == "cicerone-events":
            worker._stop.set()
            worker._stop_epoch += 1
        original_start(self)

    threading.Thread.start = _start  # type: ignore[method-assign]
    try:
        worker.start()
    finally:
        threading.Thread.start = original_start  # type: ignore[method-assign]
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and closes["n"] < 1:
        time.sleep(0.01)
    assert closes["n"] >= 1
    assert worker._thread is None or worker._thread.is_alive() is False
    assert worker._tick_guard.acquire(blocking=False)
    worker._tick_guard.release()
    assert worker._source_guard.acquire(blocking=False)
    worker._source_guard.release()


def test_event_worker_nacks_overflow(tmp_path, feature_config: FeatureConfig):
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
    for i in range(3):
        source.ingest(event_payload(event_id=f"ov-{i}", item_id=f"i{i}"))
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0, max_events=1),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    assert worker.tick() == 1  # flushes the one kept event
    # Overflow was nacked back; still pending on the webhook source.
    assert source.health().lag >= 1


def test_event_worker_stop_drains_buffer(tmp_path, feature_config: FeatureConfig):
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
    source.ingest(event_payload(event_id="drain-1", item_id="idrain"))
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
        poll_interval_seconds=60.0,
    )
    worker.start()
    assert worker.tick() == 0  # buffered, not flushed
    assert worker.stop(join_timeout_seconds=2.0) is True
    frame = pd.read_parquet(out / "recommendations.parquet")
    assert "idrain" in set(frame["item_id"].astype(str))


def test_event_worker_stop_returns_retry_acks_when_drain_ack_fails(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )

    class _AckBoom(WebhookEventSource):
        def ack(self, event_ids):  # type: ignore[no-untyped-def]
            del event_ids
            raise RuntimeError("ack failed")

    source = _AckBoom({})
    worker = EventWorker(
        source,
        MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0),
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
        ),
    )
    worker._buffer.extend([normalize_event(event_payload(event_id="retry-drain", item_id="idrain"))])
    assert worker.stop(join_timeout_seconds=2.0) is True
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["retry-drain"]
