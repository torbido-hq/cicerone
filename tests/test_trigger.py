from __future__ import annotations

import threading

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from support.fakes import FakeGuard, fake_monotonic, patch_module_ref, raising
from support.prometheus_metrics import metric_value
from support.settings import local_output, make_settings
from support.waits import WAIT_SECONDS, wait_until

from cicerone.config import IOSettings, Settings
from cicerone.trigger import RunGuard, create_app, poll_input_forever


def _settings(**overrides) -> Settings:
    return make_settings(**{"trigger_enabled": True, "trigger_auth_token": "secret", **overrides})


def test_run_guard_records_retrain_trigger_metrics():
    done = threading.Event()

    def fake_run(triggered_by: str) -> None:
        del triggered_by
        done.set()

    guard = RunGuard(debounce_seconds=60, run_fn=fake_run)
    assert guard.trigger("webhook") is True
    assert done.wait(timeout=5)
    assert guard.trigger("webhook") is False

    from prometheus_client import generate_latest

    body = generate_latest().decode()
    accepted = metric_value(
        body, "cicerone_retrain_trigger_total", {"source": "webhook", "status": "accepted"}
    )
    debounced = metric_value(
        body, "cicerone_retrain_trigger_total", {"source": "webhook", "status": "debounced"}
    )
    assert accepted >= 1
    assert debounced >= 1


def test_run_guard_records_cron_retrain_source():
    done = threading.Event()

    def fake_run(triggered_by: str) -> None:
        del triggered_by
        done.set()

    guard = RunGuard(debounce_seconds=0, run_fn=fake_run)
    assert guard.trigger("cron") is True
    assert done.wait(timeout=5)

    from prometheus_client import generate_latest

    body = generate_latest().decode()
    accepted = metric_value(body, "cicerone_retrain_trigger_total", {"source": "cron", "status": "accepted"})
    assert accepted >= 1


def test_run_guard_busy_property():
    started = threading.Event()
    release = threading.Event()

    def fake_run(triggered_by: str) -> None:
        del triggered_by
        started.set()
        release.wait(timeout=2)

    guard = RunGuard(debounce_seconds=0, run_fn=fake_run)
    assert guard.busy is False
    assert guard.trigger("webhook") is True
    assert started.wait(timeout=2)
    assert guard.busy is True
    release.set()
    assert wait_until(lambda: not guard.busy)


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


def test_run_guard_allows_trigger_after_debounce_window_elapses(monkeypatch):
    clock = fake_monotonic(monkeypatch, "cicerone.trigger.time")
    done = threading.Event()

    def fake_run(triggered_by: str) -> None:
        done.set()

    guard = RunGuard(debounce_seconds=60, run_fn=fake_run)
    assert guard.trigger("webhook") is True
    assert done.wait(WAIT_SECONDS)
    assert wait_until(lambda: not guard.busy)
    clock.now += 61

    assert guard.trigger("webhook") is True


def test_run_guard_recovers_after_run_fn_raises():
    def failing_run(triggered_by: str) -> None:
        raise ValueError("boom")

    guard = RunGuard(debounce_seconds=0, run_fn=failing_run)
    assert guard.trigger("webhook") is True

    assert wait_until(lambda: not guard.busy)
    assert isinstance(guard.last_error, ValueError)
    assert "boom" in str(guard.last_error)


def test_trigger_endpoint_requires_auth():
    app = create_app(_settings(), FakeGuard())
    client = TestClient(app)

    response = client.post("/trigger/retrain")

    assert response.status_code == 401


def test_trigger_endpoint_rejects_wrong_token():
    app = create_app(_settings(), FakeGuard())
    client = TestClient(app)

    response = client.post("/trigger/retrain", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401


def test_trigger_endpoint_starts_a_run():
    guard = FakeGuard(result=True)
    app = create_app(_settings(), guard)
    client = TestClient(app)

    response = client.post("/trigger/retrain", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 202
    assert response.json() == {"status": "started"}
    assert guard.calls == ["webhook"]


def test_trigger_endpoint_returns_429_when_debounced():
    guard = FakeGuard(result=False)
    app = create_app(_settings(), guard)
    client = TestClient(app)

    response = client.post("/trigger/retrain", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 429


def test_trigger_endpoint_health_requires_no_auth():
    app = create_app(_settings(), FakeGuard())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200


def test_trigger_docs_are_disabled():
    client = TestClient(create_app(_settings(), FakeGuard()))
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_trigger_security_headers_on_health():
    response = TestClient(create_app(_settings(), FakeGuard())).get("/health")
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_poll_input_forever_disabled_for_db_input(caplog):
    input_settings = IOSettings(kind="db", options={"database_url": "postgresql+psycopg://u:p@h/d"})
    guard = FakeGuard()

    poll_input_forever(input_settings, guard, interval_seconds=0.01)

    assert guard.calls == []


def test_poll_input_forever_triggers_on_change(tmp_path, monkeypatch):
    input_settings = local_output(tmp_path)
    guard = FakeGuard()

    markers = iter(["v1", "v2"])
    monkeypatch.setattr("cicerone.trigger._current_marker", lambda _settings: next(markers, "v2"))

    sleep_calls = {"n": 0}

    def fake_sleep(_seconds):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            raise SystemExit("stop after two ticks")

    patch_module_ref(monkeypatch, "cicerone.trigger.time", sleep=fake_sleep)

    with pytest.raises(SystemExit):
        poll_input_forever(input_settings, guard, interval_seconds=0.01)

    assert guard.calls == ["s3-poll"]


def test_poll_input_forever_logs_an_unexpected_check_error_and_keeps_polling(tmp_path, monkeypatch, caplog):
    validate = raising(TypeError("marker bug"))
    monkeypatch.setattr("cicerone.trigger.validate_storage_options", validate)
    sleeps = {"n": 0}

    def fake_sleep(_seconds):
        sleeps["n"] += 1
        if sleeps["n"] == 3:
            raise SystemExit("stop")

    patch_module_ref(monkeypatch, "cicerone.trigger.time", sleep=fake_sleep)
    guard = FakeGuard()
    with pytest.raises(SystemExit):
        poll_input_forever(local_output(tmp_path), guard, interval_seconds=0.01)
    assert validate.calls == 3
    assert caplog.text.count("Failed to check input source for changes") == 3
    assert guard.calls == []


def test_current_marker_local_backend_reflects_mtime(tmp_path):
    from cicerone.trigger import _current_marker

    input_settings = local_output(tmp_path)
    assert _current_marker(input_settings) is None

    (tmp_path / "events.parquet").write_bytes(b"data")
    assert _current_marker(input_settings) is not None


def test_current_marker_local_stat_error_propagates_to_the_poller(tmp_path, monkeypatch):
    from cicerone.trigger import _current_marker

    input_settings = local_output(tmp_path)
    (tmp_path / "events.parquet").write_bytes(b"data")

    def boom(self):
        raise PermissionError("denied")

    monkeypatch.setattr("pathlib.Path.stat", boom)
    with pytest.raises(PermissionError, match="denied"):
        _current_marker(input_settings)


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
