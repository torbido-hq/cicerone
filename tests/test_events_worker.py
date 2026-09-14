from __future__ import annotations

import pandas as pd
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
            super().close()

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
    worker.start()
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
    assert source.health().lag == 1  # duplicate acked; one remains buffered/in-flight


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
