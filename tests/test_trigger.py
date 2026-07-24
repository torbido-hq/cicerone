from __future__ import annotations

import threading
import time

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from cicerone.config import IOSettings, Settings
from cicerone.trigger import RunGuard, create_app, poll_input_forever


def _settings(**overrides) -> Settings:
    base = dict(
        input=IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/in"}),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/out"}),
        feature_config_path="/app/config/features.toml",
        top_k=10,
        half_life_days=90,
        cron_schedule="0 3 * * *",
        models=None,
        model_weights=None,
        rrf_k=None,
        automl_enabled=False,
        automl_n_splits=2,
        automl_test_days=14,
        automl_primary_metric="MAP",
        automl_candidates=None,
        mode="batch",
        serve_host="0.0.0.0",
        serve_port=8000,
        serve_auth_token=None,
        serve_default_k=10,
        serve_refresh_interval_seconds=60,
        trigger_enabled=True,
        trigger_host="0.0.0.0",
        trigger_port=8080,
        trigger_auth_token="secret",
        trigger_debounce_seconds=60,
        trigger_poll_input_bucket=False,
        trigger_poll_interval_seconds=300,
    )
    base.update(overrides)
    return Settings(**base)


class _FakeGuard:
    def __init__(self, result: bool = True):
        self.result = result
        self.calls: list[str] = []

    def trigger(self, triggered_by: str) -> bool:
        self.calls.append(triggered_by)
        return self.result


def test_run_guard_starts_a_run_when_idle():
    done = threading.Event()
    calls = []

    def fake_run(triggered_by: str) -> None:
        calls.append(triggered_by)
        done.set()

    guard = RunGuard(debounce_seconds=60, run_fn=fake_run)

    assert guard.trigger("webhook") is True
    assert done.wait(timeout=5)
    assert calls == ["webhook"]


def test_run_guard_ignores_trigger_while_running():
    release = threading.Event()
    started = threading.Event()

    def fake_run(triggered_by: str) -> None:
        started.set()
        release.wait(timeout=5)

    guard = RunGuard(debounce_seconds=0, run_fn=fake_run)

    assert guard.trigger("webhook") is True
    assert started.wait(timeout=5)
    assert guard.trigger("s3-poll") is False

    release.set()


def test_run_guard_ignores_trigger_within_debounce_window():
    done = threading.Event()

    def fake_run(triggered_by: str) -> None:
        done.set()

    guard = RunGuard(debounce_seconds=60, run_fn=fake_run)
    assert guard.trigger("webhook") is True
    assert done.wait(timeout=5)

    assert guard.trigger("webhook") is False


def test_run_guard_allows_trigger_after_debounce_window_elapses():
    done = threading.Event()

    def fake_run(triggered_by: str) -> None:
        done.set()

    guard = RunGuard(debounce_seconds=0.01, run_fn=fake_run)
    assert guard.trigger("webhook") is True
    assert done.wait(timeout=5)
    time.sleep(0.05)

    assert guard.trigger("webhook") is True


def test_run_guard_recovers_after_run_fn_raises():
    def failing_run(triggered_by: str) -> None:
        raise ValueError("boom")

    guard = RunGuard(debounce_seconds=0, run_fn=failing_run)
    assert guard.trigger("webhook") is True

    for _ in range(50):
        with guard._lock:
            if not guard._running:
                break
        time.sleep(0.05)

    assert guard._running is False


def test_trigger_endpoint_requires_auth():
    app = create_app(_settings(), _FakeGuard())
    client = TestClient(app)

    response = client.post("/trigger/retrain")

    assert response.status_code == 401


def test_trigger_endpoint_rejects_wrong_token():
    app = create_app(_settings(), _FakeGuard())
    client = TestClient(app)

    response = client.post("/trigger/retrain", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401


def test_trigger_endpoint_starts_a_run():
    guard = _FakeGuard(result=True)
    app = create_app(_settings(), guard)
    client = TestClient(app)

    response = client.post("/trigger/retrain", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 202
    assert response.json() == {"status": "started"}
    assert guard.calls == ["webhook"]


def test_trigger_endpoint_returns_429_when_debounced():
    guard = _FakeGuard(result=False)
    app = create_app(_settings(), guard)
    client = TestClient(app)

    response = client.post("/trigger/retrain", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 429


def test_trigger_endpoint_health_requires_no_auth():
    app = create_app(_settings(), _FakeGuard())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200


def test_poll_input_forever_disabled_for_db_input(caplog):
    input_settings = IOSettings(kind="db", options={"database_url": "postgresql+psycopg://u:p@h/d"})
    guard = _FakeGuard()

    poll_input_forever(input_settings, guard, interval_seconds=0.01)

    assert guard.calls == []


def test_poll_input_forever_triggers_on_change(tmp_path, monkeypatch):
    input_settings = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    guard = _FakeGuard()

    markers = iter(["v1", "v2"])
    monkeypatch.setattr("cicerone.trigger._current_marker", lambda _settings: next(markers, "v2"))

    sleep_calls = {"n": 0}

    def fake_sleep(_seconds):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            raise SystemExit("stop after two ticks")

    monkeypatch.setattr("cicerone.trigger.time.sleep", fake_sleep)

    with pytest.raises(SystemExit):
        poll_input_forever(input_settings, guard, interval_seconds=0.01)

    assert guard.calls == ["s3-poll"]


def test_current_marker_local_backend_reflects_mtime(tmp_path):
    from cicerone.trigger import _current_marker

    input_settings = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    assert _current_marker(input_settings) is None

    (tmp_path / "events.parquet").write_bytes(b"data")
    assert _current_marker(input_settings) is not None


@pytest.fixture
def s3_input_options():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")
        yield {
            "storage_backend": "s3",
            "access_key_id": "test",
            "secret_access_key": "test",
            "bucket": "test-bucket",
        }


def test_current_marker_s3_backend_returns_last_modified(s3_input_options):
    from cicerone.trigger import _current_marker

    client = boto3.client("s3", region_name="us-east-1")
    client.put_object(Bucket=s3_input_options["bucket"], Key="events.parquet", Body=b"data")

    input_settings = IOSettings(kind="dataset", options=s3_input_options)

    assert _current_marker(input_settings) is not None


def test_current_marker_s3_backend_missing_object_returns_none(s3_input_options):
    from cicerone.trigger import _current_marker

    input_settings = IOSettings(kind="dataset", options=s3_input_options)

    assert _current_marker(input_settings) is None
