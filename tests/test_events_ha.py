from __future__ import annotations

import threading
import time
from typing import Any

import pandas as pd
import pytest
from support.events import event_payload
from support.prometheus_metrics import registry_metric_value
from support.toml_config import write_toml

from cicerone.config import ConfigError, EventsSettings, IOSettings, load_settings, make_settings
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.ha import ingest_is_fanout, poll_without_apply_lock
from cicerone.events.normalize import normalize_event
from cicerone.events.online_result import OnlineRefreshResult, empty_online_rows
from cicerone.events.updater import IncrementalUpdater
from cicerone.events.webhook import WebhookEventSource
from cicerone.events.worker import EventWorker
from cicerone.feature_config import FeatureConfig
from cicerone.io.factory import build_output_sink
from cicerone.locks import LockLostError, events_apply_lock_key
from cicerone.serve.bootstrap_events import start_events_runtime


class SharedLock:
    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._owner: int | None = None
        self.acquires = 0
        self.skips = 0

    def acquire(self) -> bool:
        if self._mutex.acquire(blocking=False):
            self._owner = threading.get_ident()
            self.acquires += 1
            return True
        self.skips += 1
        return False

    def release(self) -> None:
        self._owner = None
        self._mutex.release()

    def owned(self) -> bool:
        return self._mutex.locked() and self._owner == threading.get_ident()

    def is_locked(self) -> bool:
        return self._mutex.locked()


class HeldLock:
    def acquire(self) -> bool:
        return False

    def release(self) -> None:
        return None

    def owned(self) -> bool:
        return False

    def is_locked(self) -> bool:
        return True


def _seed_out(tmp_path, users: list[str] | None = None):
    out = tmp_path / "out"
    out.mkdir()
    rows = [
        {"user_id": user_id, "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}
        for user_id in (users or ["u1", "u2"])
    ]
    pd.DataFrame(rows).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    return out, settings


def _worker(
    settings,
    source,
    feature_config: FeatureConfig,
    *,
    apply_lock=None,
    poll_without_lock: bool = True,
    busy_check=None,
    write_busy_check=None,
    fence_check=None,
    online=None,
    heartbeat_interval_seconds: float = 15.0,
) -> EventWorker:
    if fence_check is None and apply_lock is not None:
        fence_check = apply_lock.owned
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        busy_check=busy_check,
        write_busy_check=write_busy_check,
        fence_check=fence_check,
        online=online,
    )
    return EventWorker(
        source,
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0),
        updater,
        apply_lock=apply_lock,
        poll_without_lock=poll_without_lock,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )


def test_events_apply_lock_key_distinct_from_retrain():
    assert events_apply_lock_key("cicerone:scheduler:run_guard") == (
        "cicerone:scheduler:run_guard:events:apply"
    )


def test_ingest_fanout_kinds():
    assert ingest_is_fanout("redis_streams") is True
    assert ingest_is_fanout("kafka") is True
    assert ingest_is_fanout("rabbitmq") is True
    assert ingest_is_fanout("webhook") is False
    assert ingest_is_fanout("db") is False
    assert ingest_is_fanout("s3", {"mode": "list"}) is False
    assert ingest_is_fanout("s3", {"mode": "sqs"}) is True
    assert ingest_is_fanout("s3", {"queue_url": "https://sqs"}) is True
    assert poll_without_apply_lock("webhook") is True
    assert poll_without_apply_lock("db") is False
    assert poll_without_apply_lock("s3", {"mode": "list"}) is False
    assert poll_without_apply_lock("redis_streams") is True
    assert poll_without_apply_lock("kafka") is True
    assert poll_without_apply_lock("rabbitmq") is True


def test_second_replica_skips_apply_under_lock(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)
    lock = SharedLock()
    source_a = WebhookEventSource({})
    source_b = WebhookEventSource({})
    source_a.ingest(event_payload(event_id="a1", user_id="u1", item_id="ia"))
    source_b.ingest(event_payload(event_id="b1", user_id="u2", item_id="ib"))
    worker_a = _worker(settings, source_a, feature_config, apply_lock=lock)
    worker_b = _worker(settings, source_b, feature_config, apply_lock=lock)

    assert lock.acquire() is True
    before_skip = registry_metric_value("cicerone_events_lock_total", {"status": "skip"})
    before_busy = registry_metric_value("cicerone_events_apply_busy_total", {"reason": "lock"})
    assert worker_b.tick() == 0
    assert source_b.health().lag == 1
    assert registry_metric_value("cicerone_events_lock_total", {"status": "skip"}) == before_skip + 1
    assert registry_metric_value("cicerone_events_apply_busy_total", {"reason": "lock"}) == before_busy + 1
    lock.release()

    assert worker_a.tick() == 1
    assert worker_b.tick() == 1
    recs = pd.read_parquet(_out / "recommendations.parquet")
    assert {"u1", "u2"}.issubset(set(recs["user_id"].astype(str)))
    assert "ia" in set(recs["item_id"].astype(str))
    assert "ib" in set(recs["item_id"].astype(str))


def test_non_fanout_skips_poll_when_lock_held(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)

    class _CountingSource(WebhookEventSource):
        def __init__(self) -> None:
            super().__init__({})
            self.polls = 0

        def poll(self, max_events: int = 100):
            self.polls += 1
            return super().poll(max_events)

    source = _CountingSource()
    source.ingest(event_payload(event_id="db1", user_id="u1"))
    worker = _worker(
        settings,
        source,
        feature_config,
        apply_lock=HeldLock(),
        poll_without_lock=False,
    )
    assert worker.tick() == 0
    assert source.polls == 0
    assert source.health().lag == 1


def test_retrain_lock_on_other_process_blocks_apply(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)
    retrain_held = {"v": True}
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="r1", user_id="u1", item_id="ir"))
    worker = _worker(
        settings,
        source,
        feature_config,
        busy_check=lambda: retrain_held["v"],
    )
    before = registry_metric_value("cicerone_events_apply_busy_total", {"reason": "retrain"})
    assert worker.tick() == 0
    assert source.health().lag == 1
    assert registry_metric_value("cicerone_events_apply_busy_total", {"reason": "retrain"}) == before + 1
    retrain_held["v"] = False
    assert worker.tick() == 1


def test_fence_loss_skips_dataset_write(tmp_path, feature_config: FeatureConfig):
    out, settings = _seed_out(tmp_path, users=["u1"])
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="f1", user_id="u1", item_id="lost"))
    worker = _worker(
        settings,
        source,
        feature_config,
        fence_check=lambda: False,
    )
    assert worker.tick() == 0
    recs = pd.read_parquet(out / "recommendations.parquet")
    assert "lost" not in set(recs["item_id"].astype(str))


def test_updater_raises_lock_lost_on_fence(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        fence_check=lambda: False,
    )
    with pytest.raises(LockLostError, match="apply lock lost"):
        updater.apply([normalize_event(event_payload())])


def test_two_writers_serialized_keep_both_users(tmp_path, feature_config: FeatureConfig):
    out, settings = _seed_out(tmp_path)
    lock = SharedLock()
    errors: list[BaseException] = []

    def run_replica(user_id: str, item_id: str) -> None:
        try:
            source = WebhookEventSource({})
            source.ingest(event_payload(event_id=f"e-{user_id}", user_id=user_id, item_id=item_id))
            worker = _worker(settings, source, feature_config, apply_lock=lock)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if worker.tick() == 1:
                    return
                time.sleep(0.01)
            errors.append(AssertionError(f"{user_id} never applied"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=run_replica, args=("u1", "ia")),
        threading.Thread(target=run_replica, args=("u2", "ib")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert errors == []
    recs = pd.read_parquet(out / "recommendations.parquet")
    assert {"u1", "u2"}.issubset(set(recs["user_id"].astype(str)))
    assert {"ia", "ib"}.issubset(set(recs["item_id"].astype(str)))


def test_redis_streams_unique_consumers_leader_only(tmp_path, feature_config, monkeypatch):
    from test_events_redis_streams import FakeRedis, _install_fake_redis, _options

    from cicerone.events.redis_streams import RedisStreamsEventSource

    _out, settings = _seed_out(tmp_path)
    client = _install_fake_redis(monkeypatch, FakeRedis())
    lock = SharedLock()
    leader = RedisStreamsEventSource(_options(consumer_name="serve-1"))
    follower = RedisStreamsEventSource(_options(consumer_name="serve-2"))
    leader.connect()
    follower.connect()
    client.xadd("cicerone:events", event_payload(event_id="s1", user_id="u1", item_id="is1"))
    client.xadd("cicerone:events", event_payload(event_id="s2", user_id="u2", item_id="is2"))

    worker_leader = _worker(settings, leader, feature_config, apply_lock=lock)
    worker_follower = _worker(settings, follower, feature_config, apply_lock=lock)

    assert lock.acquire() is True
    assert worker_follower.tick() == 0
    lock.release()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        worker_leader.tick()
        worker_follower.tick()
        recs = pd.read_parquet(_out / "recommendations.parquet")
        if {"is1", "is2"}.issubset(set(recs["item_id"].astype(str))):
            break
        time.sleep(0.01)
    else:
        recs = pd.read_parquet(_out / "recommendations.parquet")
        raise AssertionError(f"missing stream items in {recs['item_id'].tolist()}")
    assert leader._consumer == "serve-1"
    assert follower._consumer == "serve-2"


def test_load_events_ha_requires_distributed_lock(tmp_path):
    path = write_toml(
        tmp_path,
        """
        [job]
        mode = "serve"
        [job.trigger]
        lock_backend = "in_process"
        [serve]
        auth_token = "tok"
        [events]
        enabled = true
        kind = "webhook"
        ha = true
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )
    with pytest.raises(ConfigError, match="events.ha = true requires"):
        load_settings(path)


def test_load_events_ha_with_redis_lock(tmp_path):
    path = write_toml(
        tmp_path,
        """
        [job]
        mode = "serve"
        [job.trigger]
        lock_backend = "redis"
        redis_url = "redis://localhost:6379/0"
        [serve]
        auth_token = "tok"
        [events]
        enabled = true
        kind = "webhook"
        ha = true
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )
    settings = load_settings(path)
    assert settings.events.ha is True
    assert settings.trigger.lock_backend == "redis"


def test_start_events_runtime_wires_apply_lock(tmp_path, feature_config: FeatureConfig, monkeypatch):
    out, _settings = _seed_out(tmp_path)
    apply_fake = SharedLock()
    retrain_fake = HeldLock()
    retrain_fake.is_locked = lambda: False  # type: ignore[method-assign]

    seen_ttl: dict[str, float | None] = {}

    def _build(
        _settings: Any,
        *,
        lock_key: str | None = None,
        ttl_seconds: float | None = None,
    ) -> SharedLock | HeldLock:
        if lock_key is not None and lock_key.endswith(":events:apply"):
            seen_ttl["apply"] = ttl_seconds
            return apply_fake
        seen_ttl["retrain"] = ttl_seconds
        return retrain_fake

    monkeypatch.setattr("cicerone.serve.bootstrap_events.build_lock_backend", _build)

    class _Reader:
        def refresh(self) -> None:
            return None

    runtime = start_events_runtime(
        make_settings(
            output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
            trigger_lock_backend="redis",
            trigger_redis_url="redis://localhost:6379/0",
            events=EventsSettings(enabled=True, kind="webhook", ha=True),
        ),
        feature_config=feature_config,
        reader=_Reader(),  # type: ignore[arg-type]
    )
    assert runtime.apply_lock is apply_fake
    assert runtime.worker is not None
    assert runtime.worker._apply_lock is apply_fake
    assert seen_ttl["apply"] == 60.0
    assert seen_ttl["retrain"] is None
    runtime.stop()


def test_start_events_runtime_skips_apply_lock_without_ha(
    tmp_path, feature_config: FeatureConfig, monkeypatch
):
    out, _settings = _seed_out(tmp_path)

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("build_lock_backend must not run when events.ha is false")

    monkeypatch.setattr("cicerone.serve.bootstrap_events.build_lock_backend", _boom)

    class _Reader:
        def refresh(self) -> None:
            return None

    runtime = start_events_runtime(
        make_settings(
            output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
            trigger_lock_backend="redis",
            trigger_redis_url="redis://localhost:6379/0",
            events=EventsSettings(enabled=True, kind="webhook", ha=False),
        ),
        feature_config=feature_config,
        reader=_Reader(),  # type: ignore[arg-type]
    )
    assert runtime.apply_lock is None
    assert runtime.worker is not None
    assert runtime.worker._apply_lock is None
    runtime.stop()


def test_leader_gauge_cleared_after_release(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="g1", user_id="u1", item_id="ig"))
    worker = _worker(settings, source, feature_config, apply_lock=SharedLock())
    assert worker.tick() == 1
    assert registry_metric_value("cicerone_events_leader") == 0


def test_idle_fanout_tick_does_not_acquire_apply_lock(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)
    lock = SharedLock()
    worker = _worker(
        settings,
        WebhookEventSource({}),
        feature_config,
        apply_lock=lock,
        poll_without_lock=True,
    )
    assert worker.tick() == 0
    assert lock.acquires == 0


def test_exclusive_poll_acquires_lock_when_idle(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)
    lock = SharedLock()

    class _CountingSource(WebhookEventSource):
        def __init__(self) -> None:
            super().__init__({})
            self.polls = 0

        def poll(self, max_events: int = 100):
            self.polls += 1
            return super().poll(max_events)

    source = _CountingSource()
    worker = _worker(
        settings,
        source,
        feature_config,
        apply_lock=lock,
        poll_without_lock=False,
    )
    assert worker.tick() == 0
    assert lock.acquires == 1
    assert source.polls == 1
    assert lock.is_locked() is False


def test_stop_drain_nacks_when_apply_lock_busy(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="d1", user_id="u1"))
    worker = _worker(settings, source, feature_config, apply_lock=HeldLock())
    polled = list(source.poll(10))
    worker._buffer.extend(polled)
    worker._drain_buffer_on_stop()
    assert source.health().lag >= 1


def test_stop_drain_nacks_when_retrain_busy(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="d2", user_id="u1"))
    worker = _worker(settings, source, feature_config, busy_check=lambda: True)
    polled = list(source.poll(10))
    worker._buffer.extend(polled)
    worker._drain_buffer_on_stop()
    assert source.health().lag >= 1


def test_stop_drain_nacks_on_fence_loss(tmp_path, feature_config: FeatureConfig):
    out, settings = _seed_out(tmp_path, users=["u1"])
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="d3", user_id="u1", item_id="lost"))
    worker = _worker(settings, source, feature_config, fence_check=lambda: False)
    polled = list(source.poll(10))
    worker._buffer.extend(polled)
    worker._drain_buffer_on_stop()
    recs = pd.read_parquet(out / "recommendations.parquet")
    assert "lost" not in set(recs["item_id"].astype(str))
    assert source.health().lag >= 1


def test_stop_drain_applies_when_lease_acquired(tmp_path, feature_config: FeatureConfig):
    out, settings = _seed_out(tmp_path)
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="d4", user_id="u1", item_id="idrain"))
    worker = _worker(settings, source, feature_config, apply_lock=SharedLock())
    polled = list(source.poll(10))
    worker._buffer.extend(polled)
    worker._drain_buffer_on_stop()
    assert source.health().lag == 0
    recs = pd.read_parquet(out / "recommendations.parquet")
    assert "idrain" in set(recs["item_id"].astype(str))


class _FakeOnline:
    def __init__(self, *, fail_commits: int = 0, lost_on_commit: bool = False) -> None:
        self.commits = 0
        self.aborts = 0
        self.refreshes = 0
        self._fail_commits = fail_commits
        self._lost_on_commit = lost_on_commit

    def refresh(self, events):  # type: ignore[no-untyped-def]
        del events
        self.refreshes += 1
        return OnlineRefreshResult(rows=empty_online_rows())

    def invalidate(self) -> None:
        return None

    def commit(self) -> None:
        if self._lost_on_commit:
            raise LockLostError("lease lost")
        if self._fail_commits > 0:
            self._fail_commits -= 1
            raise RuntimeError("persist boom")
        self.commits += 1

    def abort(self) -> None:
        self.aborts += 1


def test_ha_online_commits_after_ack(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="on1", user_id="u1", item_id="ion"))
    online = _FakeOnline()
    worker = _worker(settings, source, feature_config, apply_lock=SharedLock(), online=online)
    assert worker.tick() == 1
    assert online.refreshes == 1
    assert online.commits == 1
    assert online.aborts == 0
    assert source.health().lag == 0


def test_ha_online_retries_persist_then_commits(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="on2", user_id="u1", item_id="ion2"))
    online = _FakeOnline(fail_commits=2)
    worker = _worker(settings, source, feature_config, apply_lock=SharedLock(), online=online)
    assert worker.tick() == 1
    assert online.commits == 1
    assert online.aborts == 0


def test_ha_online_aborts_pending_when_persist_exhausted(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="on3", user_id="u1", item_id="ion3"))
    online = _FakeOnline(fail_commits=10)
    worker = _worker(settings, source, feature_config, apply_lock=SharedLock(), online=online)
    assert worker.tick() == 1
    assert source.health().lag == 0
    assert online.commits == 0
    assert online.aborts == 1


def test_ha_online_aborts_pending_on_fence_loss_during_persist(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="on4", user_id="u1", item_id="ion4"))
    online = _FakeOnline(lost_on_commit=True)
    worker = _worker(settings, source, feature_config, apply_lock=SharedLock(), online=online)
    assert worker.tick() == 1
    assert online.commits == 0
    assert online.aborts == 1


def test_ha_worker_heartbeats_inflight_during_flush(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)

    class _BeatingSource(WebhookEventSource):
        def __init__(self) -> None:
            super().__init__({})
            self.beats = 0

        def heartbeat(self, events):  # type: ignore[no-untyped-def]
            del events
            self.beats += 1

    source = _BeatingSource()
    source.ingest(event_payload(event_id="hb1", user_id="u1", item_id="ihb"))
    worker = _worker(
        settings,
        source,
        feature_config,
        apply_lock=SharedLock(),
        heartbeat_interval_seconds=0.05,
    )
    original = worker._updater.apply

    def _slow(events, *, persist_online: bool = True):  # type: ignore[no-untyped-def]
        time.sleep(0.16)
        return original(events, persist_online=persist_online)

    worker._updater.apply = _slow  # type: ignore[method-assign]
    assert worker.tick() == 1
    assert source.beats >= 2


def test_ha_worker_heartbeats_once_when_interval_disabled(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)

    class _BeatingSource(WebhookEventSource):
        def __init__(self) -> None:
            super().__init__({})
            self.beats = 0

        def heartbeat(self, events):  # type: ignore[no-untyped-def]
            del events
            self.beats += 1

    source = _BeatingSource()
    source.ingest(event_payload(event_id="hb0", user_id="u1", item_id="ihb0"))
    worker = _worker(
        settings,
        source,
        feature_config,
        apply_lock=SharedLock(),
        heartbeat_interval_seconds=0,
    )
    assert worker.tick() == 1
    assert source.beats == 1


def test_ha_worker_nacks_when_heartbeat_raises(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)

    class _BoomBeat(WebhookEventSource):
        def heartbeat(self, events):  # type: ignore[no-untyped-def]
            del events
            raise RuntimeError("beat failed")

    source = _BoomBeat({})
    source.ingest(event_payload(event_id="hbx", user_id="u1", item_id="ihbx"))
    worker = _worker(
        settings,
        source,
        feature_config,
        apply_lock=SharedLock(),
        heartbeat_interval_seconds=0,
    )
    assert worker.tick() == 0
    assert source.health().lag == 1


def test_ha_worker_nacks_when_later_heartbeat_raises(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)

    class _LaterBoom(WebhookEventSource):
        def __init__(self) -> None:
            super().__init__({})
            self.beats = 0

        def heartbeat(self, events):  # type: ignore[no-untyped-def]
            del events
            self.beats += 1
            if self.beats > 1:
                raise RuntimeError("lost visibility")

    source = _LaterBoom()
    source.ingest(event_payload(event_id="hbl", user_id="u1", item_id="ihbl"))
    worker = _worker(
        settings,
        source,
        feature_config,
        apply_lock=SharedLock(),
        heartbeat_interval_seconds=0.05,
    )
    original = worker._updater.apply

    def _slow(events, *, persist_online: bool = True):  # type: ignore[no-untyped-def]
        time.sleep(0.16)
        return original(events, persist_online=persist_online)

    worker._updater.apply = _slow  # type: ignore[method-assign]
    assert worker.tick() == 0
    assert source.beats >= 2
    assert source.health().lag == 1


def test_ha_online_skips_persist_when_write_busy_after_apply(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="onbusy", user_id="u1", item_id="ionb"))
    online = _FakeOnline()
    busy = {"v": False}

    def _write_busy() -> bool:
        return busy["v"]

    worker = _worker(
        settings,
        source,
        feature_config,
        apply_lock=SharedLock(),
        online=online,
        write_busy_check=_write_busy,
        busy_check=lambda: False,
    )
    original = worker._updater.apply

    def _flip(events, *, persist_online: bool = True):  # type: ignore[no-untyped-def]
        applied = original(events, persist_online=persist_online)
        busy["v"] = True
        return applied

    worker._updater.apply = _flip  # type: ignore[method-assign]
    assert worker.tick() == 1
    assert source.health().lag == 0
    assert online.commits == 0
    assert online.aborts == 1
