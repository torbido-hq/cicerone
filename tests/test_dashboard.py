from __future__ import annotations

from datetime import UTC, datetime

import pytest
from conftest import make_settings
from fastapi.testclient import TestClient

from cicerone.config import Settings
from cicerone.dashboard import create_app, main
from cicerone.http_auth import require_basic_auth


def _settings(**overrides) -> Settings:
    return make_settings(**{"dashboard_enabled": True, **overrides})


class _FakeReader:
    def __init__(self, latest: dict | None, history: list[dict] | None = None):
        self._latest = latest
        self._history = history if history is not None else ([latest] if latest else [])

    def read_latest(self) -> dict | None:
        return self._latest

    def read_recent(self, limit: int) -> list[dict]:
        return self._history[:limit]


def _users_with(username: str, password: str) -> dict[str, str]:
    import bcrypt

    return {username: bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")}


def test_health_requires_no_auth():
    app = create_app(_settings(), _FakeReader(None), _users_with("alice", "s3cret"))
    client = TestClient(app)

    assert client.get("/health").status_code == 200


def test_dashboard_page_requires_auth():
    app = create_app(_settings(), _FakeReader(None), _users_with("alice", "s3cret"))
    client = TestClient(app)

    response = client.get("/dashboard")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Basic"


def test_dashboard_page_rejects_wrong_password():
    app = create_app(_settings(), _FakeReader(None), _users_with("alice", "s3cret"))
    client = TestClient(app)

    response = client.get("/dashboard", auth=("alice", "wrong"))

    assert response.status_code == 401


def test_dashboard_page_rejects_unknown_username():
    app = create_app(_settings(), _FakeReader(None), _users_with("alice", "s3cret"))
    client = TestClient(app)

    response = client.get("/dashboard", auth=("ghost", "s3cret"))

    assert response.status_code == 401


def test_dashboard_page_renders_with_valid_credentials():
    app = create_app(_settings(), _FakeReader(None), _users_with("alice", "s3cret"))
    client = TestClient(app)

    response = client.get("/dashboard", auth=("alice", "s3cret"))

    assert response.status_code == 200
    assert "Cicerone" in response.text
    assert "No job runs recorded yet." in response.text


def test_status_partial_renders_latest_manifest():
    manifest = {
        "status": "success",
        "generated_at": "2026-07-28T00:00:00+00:00",
        "triggered_by": "cron",
        "n_events": 42,
        "n_users_with_recommendations": 7,
        "models": "collaborative,popular",
        "error": None,
    }
    app = create_app(_settings(), _FakeReader(manifest), _users_with("alice", "s3cret"))
    client = TestClient(app)

    response = client.get("/partials/status", auth=("alice", "s3cret"))

    assert response.status_code == 200
    assert "success" in response.text
    assert "cron" in response.text
    assert "42" in response.text


def test_status_partial_renders_failed_run_with_error():
    manifest = {
        "status": "failed",
        "generated_at": "2026-07-28T00:00:00+00:00",
        "triggered_by": "webhook",
        "error": "boom: something broke",
    }
    app = create_app(_settings(), _FakeReader(manifest), _users_with("alice", "s3cret"))
    client = TestClient(app)

    response = client.get("/partials/status", auth=("alice", "s3cret"))

    assert response.status_code == 200
    assert "failed" in response.text
    assert "boom: something broke" in response.text


def test_status_partial_marks_stale_run_as_stale():
    manifest = {"status": "success", "generated_at": "2020-01-01T00:00:00+00:00"}
    app = create_app(_settings(), _FakeReader(manifest), _users_with("alice", "s3cret"))
    client = TestClient(app)

    response = client.get("/partials/status", auth=("alice", "s3cret"))

    assert "Stale" in response.text


def test_status_partial_history_limited_to_a_single_run_shows_dataset_backend_note():
    manifest = {"status": "success", "generated_at": "2026-07-28T00:00:00+00:00"}
    app = create_app(_settings(), _FakeReader(manifest), _users_with("alice", "s3cret"))
    client = TestClient(app)

    response = client.get("/partials/status", auth=("alice", "s3cret"))

    assert "Only the latest run is available" in response.text


def test_status_partial_multiple_history_rows_no_dataset_backend_note():
    manifest = {"status": "success", "generated_at": "2026-07-28T00:00:00+00:00"}
    older = {"status": "failed", "generated_at": "2026-07-27T00:00:00+00:00", "error": "boom"}
    app = create_app(
        _settings(),
        _FakeReader(manifest, history=[manifest, older]),
        _users_with("alice", "s3cret"),
    )
    client = TestClient(app)

    response = client.get("/partials/status", auth=("alice", "s3cret"))

    assert "Only the latest run is available" not in response.text
    assert "boom" in response.text


def test_main_raises_when_dashboard_not_enabled(monkeypatch):
    monkeypatch.setattr("cicerone.dashboard.load_settings", lambda: _settings(dashboard_enabled=False))

    with pytest.raises(RuntimeError, match="dashboard.enabled"):
        main()


def test_main_raises_when_no_users_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cicerone.dashboard.load_settings",
        lambda: _settings(dashboard_users_path=str(tmp_path / "dashboard_users.toml")),
    )

    with pytest.raises(RuntimeError, match="No dashboard users configured"):
        main()


def test_require_basic_auth_used_directly_rejects_unknown_user():
    # Call dependency directly for the timing-safe unknown-username branch.
    from fastapi import HTTPException
    from fastapi.security import HTTPBasicCredentials

    dependency = require_basic_auth(_users_with("alice", "s3cret"))
    with pytest.raises(HTTPException):
        dependency(HTTPBasicCredentials(username="ghost", password="whatever"))


def test_compute_staleness_no_manifest_is_stale():
    from cicerone.dashboard import _compute_staleness

    result = _compute_staleness(None, "0 3 * * *", datetime.now(UTC))

    assert result == {"is_stale": True, "expected_next_run": None, "error": None}


def test_compute_staleness_invalid_cron_schedule_is_unknown_not_a_crash():
    from cicerone.dashboard import _compute_staleness

    manifest = {"status": "success", "generated_at": "2026-07-28T00:00:00+00:00"}
    result = _compute_staleness(manifest, "not a cron expression", datetime.now(UTC))

    assert result["is_stale"] is None
    assert result["expected_next_run"] is None
    assert result["error"]


def test_compute_staleness_accepts_a_datetime_generated_at():
    # Db-backed manifests may yield datetime/Timestamp, not ISO strings.
    from cicerone.dashboard import _compute_staleness

    manifest = {"status": "success", "generated_at": datetime(2026, 7, 28, tzinfo=UTC)}
    result = _compute_staleness(manifest, "0 3 * * *", datetime.now(UTC))

    assert result["error"] is None
    assert result["is_stale"] is not None


def test_compute_staleness_malformed_generated_at_is_unknown_not_a_crash():
    from cicerone.dashboard import _compute_staleness

    manifest = {"status": "success", "generated_at": "not-a-timestamp"}
    result = _compute_staleness(manifest, "0 3 * * *", datetime.now(UTC))

    assert result["is_stale"] is None
    assert result["expected_next_run"] is None
    assert result["error"]


def test_compute_staleness_naive_generated_at_is_treated_as_utc_not_a_crash():
    # Naive generated_at must not TypeError against aware `now` — degrade to unknown.
    from cicerone.dashboard import _compute_staleness

    manifest = {"status": "success", "generated_at": "2026-07-28T00:00:00"}
    result = _compute_staleness(manifest, "0 3 * * *", datetime.now(UTC))

    assert result["error"] is None
    assert result["is_stale"] is not None

    manifest_datetime = {"status": "success", "generated_at": datetime(2026, 7, 28)}
    result_datetime = _compute_staleness(manifest_datetime, "0 3 * * *", datetime.now(UTC))

    assert result_datetime["error"] is None
    assert result_datetime["is_stale"] is not None


def test_status_partial_shows_unknown_staleness_for_invalid_cron_schedule():
    manifest = {"status": "success", "generated_at": "2026-07-28T00:00:00+00:00"}
    app = create_app(
        _settings(cron_schedule="not a cron expression"),
        _FakeReader(manifest),
        _users_with("alice", "s3cret"),
    )
    client = TestClient(app)

    response = client.get("/partials/status", auth=("alice", "s3cret"))

    assert response.status_code == 200
    assert "misconfigured" in response.text
