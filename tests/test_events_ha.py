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
    fence_check=None,
) -> EventWorker:
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        busy_check=busy_check,
        fence_check=fence_check,
    )
    return EventWorker(
        source,
        MicroBatchBuffer(batch_size=1, batch_window_seconds=60.0),
        updater,
        apply_lock=apply_lock,
        poll_without_lock=poll_without_lock,
    )


def test_events_apply_lock_key_distinct_from_retrain():
    assert events_apply_lock_key("cicerone:scheduler:run_guard") == (
        "cicerone:scheduler:run_guard:events:apply"
    )


def test_ingest_fanout_kinds():
    assert ingest_is_fanout("redis_streams") is True
    assert ingest_is_fanout("webhook") is False
    assert ingest_is_fanout("db") is False
    assert ingest_is_fanout("s3", {"mode": "list"}) is False
    assert ingest_is_fanout("s3", {"mode": "sqs"}) is True
    assert ingest_is_fanout("s3", {"queue_url": "https://sqs"}) is True
    assert poll_without_apply_lock("webhook") is True
    assert poll_without_apply_lock("db") is False
    assert poll_without_apply_lock("s3", {"mode": "list"}) is False
    assert poll_without_apply_lock("redis_streams") is True


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

    def _build(_settings: Any, lock_key: str | None = None) -> SharedLock | HeldLock:
        if lock_key is not None and lock_key.endswith(":events:apply"):
            return apply_fake
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
    runtime.stop()


def test_stop_drain_nacks_when_apply_lock_busy(tmp_path, feature_config: FeatureConfig):
    _out, settings = _seed_out(tmp_path)
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="d1", user_id="u1"))
    worker = _worker(settings, source, feature_config, apply_lock=HeldLock())
    polled = list(source.poll(10))
    worker._buffer.extend(polled)
    worker._drain_buffer_on_stop()
    assert source.health().lag >= 1
