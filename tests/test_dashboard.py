from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pandas as pd
import pytest
from conftest import make_settings
from fastapi.testclient import TestClient
from support.fakes import FakeReader

from cicerone.config import Settings
from cicerone.dashboard import ROBOTS_TAG, ROBOTS_TXT, create_app, main
from cicerone.dashboard_lookup import HISTORY_UNAVAILABLE, LOOKUP_FAILED, MISSING
from cicerone.http_auth import require_basic_auth


def _settings(**overrides) -> Settings:
    return make_settings(**{"dashboard_enabled": True, **overrides})


def _flash_cookie(response) -> str:
    return (response.cookies.get("cicerone_flash") or "").strip('"')


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


def test_robots_txt_disallows_all_without_auth():
    app = create_app(_settings(), _FakeReader(None), _users_with("alice", "s3cret"))
    client = TestClient(app)

    response = client.get("/robots.txt")

    assert response.status_code == 200
    assert response.text == ROBOTS_TXT
    assert response.headers["x-robots-tag"] == ROBOTS_TAG
    assert "no-store" in response.headers["cache-control"]


def test_dashboard_responses_are_not_indexable():
    app = create_app(_settings(), _FakeReader(None), _users_with("alice", "s3cret"))
    client = TestClient(app)
    auth = ("alice", "s3cret")

    for path in (
        "/health",
        "/dashboard",
        "/dashboard/config",
        "/dashboard/quality",
        "/dashboard/experiments",
    ):
        response = client.get(path, auth=auth)
        assert response.status_code == 200, path
        assert response.headers["x-robots-tag"] == ROBOTS_TAG
        assert "no-store" in response.headers["cache-control"]

    unauthorized = client.get("/dashboard")
    assert unauthorized.status_code == 401
    assert unauthorized.headers["x-robots-tag"] == ROBOTS_TAG

    static = client.get("/static/cicerone-logo.svg")
    assert static.status_code == 200
    assert static.headers["x-robots-tag"] == ROBOTS_TAG
    assert "no-store" not in static.headers.get("cache-control", "")

    prefix_collision = client.get("/staticx")
    assert prefix_collision.status_code == 404
    assert "no-store" in prefix_collision.headers.get("cache-control", "")


def test_dashboard_openapi_docs_are_disabled():
    app = create_app(_settings(), _FakeReader(None), _users_with("alice", "s3cret"))
    client = TestClient(app)

    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


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
    assert "Inspect user" in response.text
    assert "Enter a user id to inspect their recent events and current top-K." in response.text
    assert 'for="user-id"' in response.text
    assert 'aria-live="polite"' in response.text
    assert 'id="recommendation-announcer"' in response.text
    assert 'id="status-announcer"' in response.text
    assert 'href="#main"' in response.text
    assert "hx-disabled-elt=\"button[type='submit']\"" in response.text
    assert "Pause updates" in response.text
    assert "Refresh" in response.text
    assert ">Job status<" in response.text
    assert 'hx-trigger="refresh"' in response.text
    assert 'hx-trigger="load, refresh, every' not in response.text
    assert "<title>Cicerone dashboard</title>" in response.text
    assert 'id="recommendation-results" class="mt-3"' in response.text
    assert 'href="/dashboard/experiments"' in response.text
    assert 'href="/dashboard/quality"' in response.text
    assert 'href="/dashboard/config"' in response.text
    assert ">Quality" in response.text
    assert ">Config<" in response.text
    assert "Sign out" in response.text
    assert "flex-wrap" in response.text
    assert 'href="/dashboard/logout"' in response.text
    assert 'name="robots"' in response.text
    assert f'content="{ROBOTS_TAG}"' in response.text
    assert "noindex" in response.text
    assert 'aria-label="Main"' in response.text
    assert 'id="main"' in response.text
    assert "Skip to content" in response.text


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
    assert "Users with recommendations" in response.text
    assert "Latest run" in response.text
    assert "Recent runs" in response.text
    assert 'data-run-status="success"' in response.text
    assert "overflow-x-auto" in response.text
    assert '<span class="text-slate-500">—</span>' in response.text
    assert "current" in response.text
    assert 'data-controller="time-ago"' in response.text


def test_status_partial_shows_incremental_panel_when_events_enabled():
    from cicerone.config import EventsSettings

    history = [
        {
            "status": "success",
            "generated_at": "2026-08-13T15:00:00+00:00",
            "triggered_by": "incremental",
            "last_incremental_at": "2026-08-13T15:00:00+00:00",
            "incremental_events_applied": 3,
            "online_users_refreshed": 2,
            "online_fit_partial_epochs": 1,
            "n_events": 3,
            "error": None,
        },
        {
            "status": "success",
            "generated_at": "2026-08-13T03:00:00+00:00",
            "triggered_by": "cron",
            "n_events": 100,
            "error": None,
        },
    ]
    app = create_app(
        _settings(events=EventsSettings(enabled=True, kind="webhook")),
        _FakeReader(history[1], history=history),
        _users_with("alice", "s3cret"),
    )
    response = TestClient(app).get("/partials/status", auth=("alice", "s3cret"))
    assert response.status_code == 200
    assert "Incremental events" in response.text
    assert "Source: webhook" in response.text
    assert "3" in response.text
    assert "Online users refreshed" in response.text
    assert "cicerone_events_source_lag" in response.text


def test_status_partial_hides_incremental_panel_when_events_disabled():
    history = [
        {
            "status": "success",
            "generated_at": "2026-08-13T15:00:00+00:00",
            "triggered_by": "incremental",
            "incremental_events_applied": 3,
            "error": None,
        }
    ]
    app = create_app(
        _settings(),
        _FakeReader(history[0], history=history),
        _users_with("alice", "s3cret"),
    )
    response = TestClient(app).get("/partials/status", auth=("alice", "s3cret"))
    assert "Incremental events" not in response.text


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
    assert 'data-run-stale="true"' in response.text
    assert "border-amber-300 bg-amber-50" in response.text
    assert "border-emerald-300 bg-emerald-50" not in response.text


def test_status_partial_history_limited_to_a_single_run_shows_dataset_backend_note():
    manifest = {"status": "success", "generated_at": "2026-07-28T00:00:00+00:00"}
    app = create_app(_settings(), _FakeReader(manifest), _users_with("alice", "s3cret"))
    client = TestClient(app)

    response = client.get("/partials/status", auth=("alice", "s3cret"))

    assert "Only the latest run is available" in response.text
    assert "Recent job runs" in response.text


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


def _run_main(monkeypatch, *, settings=None, rec_reader=None, history_reader=object) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_create_app(settings, reader, users, rec_reader=None, history_reader=None, **kwargs):
        captured.update(rec_reader=rec_reader, history_reader=history_reader, **kwargs)
        return object()

    monkeypatch.setattr("cicerone.dashboard.load_settings", lambda: settings or _settings())
    monkeypatch.setattr("cicerone.dashboard.load_users", lambda _path: {"alice": "hash"})
    monkeypatch.setattr("cicerone.dashboard.build_manifest_reader", lambda _output: _FakeReader(None))
    monkeypatch.setattr(
        "cicerone.dashboard.build_recommendation_reader",
        rec_reader or (lambda _output: FakeReader(pd.DataFrame())),
    )
    monkeypatch.setattr("cicerone.dashboard.build_input_source", lambda _input: history_reader())
    monkeypatch.setattr("cicerone.dashboard.create_app", fake_create_app)
    monkeypatch.setattr(
        "cicerone.dashboard.uvicorn",
        type("_Uvicorn", (), {"run": staticmethod(lambda *_a, **_k: None)}),
    )
    main()
    return captured


def _raise(exc: Exception):
    def boom(*_args, **_kwargs):
        raise exc

    return boom


def _with_features(tmp_path, text: str) -> Settings:
    features = tmp_path / "features.toml"
    features.write_text(text)
    return dataclasses.replace(_settings(), feature_config_path=str(features))


_MALFORMED_FEATURES = '[[item_features]]\ncolumn = "style"\ntype = "bogus"\n'


def test_main_starts_when_recommendation_reader_fails(monkeypatch):
    monkeypatch.setenv("CICERONE_CONFIG_PATH", "/tmp/cicerone.dashboard.toml")

    captured = _run_main(monkeypatch, rec_reader=_raise(OSError("bad store")))

    assert captured["rec_reader"] is None
    assert captured["config_path"] == "/tmp/cicerone.dashboard.toml"


def test_main_starts_when_history_reader_fails(monkeypatch):
    captured = _run_main(monkeypatch, history_reader=_raise(ValueError("bad events")))

    assert captured["history_reader"] is None


def test_main_propagates_unexpected_store_errors(monkeypatch):
    with pytest.raises(RuntimeError, match="bug"):
        _run_main(monkeypatch, rec_reader=_raise(RuntimeError("bug")))


def test_main_fails_on_a_malformed_feature_config(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Unknown feature type"):
        _run_main(monkeypatch, settings=_with_features(tmp_path, _MALFORMED_FEATURES))


def test_main_fails_on_a_malformed_feature_config_when_the_store_is_down(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Unknown feature type"):
        _run_main(
            monkeypatch,
            settings=_with_features(tmp_path, _MALFORMED_FEATURES),
            rec_reader=_raise(OSError("bad store")),
        )


def test_main_passes_the_feature_config_to_create_app(tmp_path, monkeypatch):
    configured: list[object] = []
    rec_reader = FakeReader(pd.DataFrame())
    monkeypatch.setattr(
        rec_reader, "configure_item_filters", lambda **kw: configured.append(kw["availability_filters"])
    )

    captured = _run_main(
        monkeypatch,
        settings=_with_features(tmp_path, 'item_availability_filters = ["in_stock"]\n'),
        rec_reader=lambda _output: rec_reader,
    )

    assert captured["feature_config"].item_availability_filters == ["in_stock"]
    assert captured["rec_reader"] is rec_reader
    assert configured == [["in_stock"]]


def test_main_warns_and_continues_without_a_feature_config(tmp_path, monkeypatch, caplog):
    settings = dataclasses.replace(_settings(), feature_config_path=str(tmp_path / "missing.toml"))

    with caplog.at_level("WARNING", logger="cicerone.serve.item_filters"):
        captured = _run_main(monkeypatch, settings=settings)

    assert captured["feature_config"] is None
    assert captured["rec_reader"] is not None
    assert "feature config missing" in caplog.text


def test_main_disables_lookup_when_configuring_item_filters_fails(monkeypatch):
    rec_reader = FakeReader(pd.DataFrame())
    monkeypatch.setattr(rec_reader, "configure_item_filters", _raise(OSError("s3 blip")))

    captured = _run_main(monkeypatch, rec_reader=lambda _output: rec_reader)

    assert captured["rec_reader"] is None


def test_main_uses_package_default_config_path(monkeypatch):
    monkeypatch.delenv("CICERONE_CONFIG_PATH", raising=False)
    monkeypatch.setattr("cicerone.dashboard.DEFAULT_CONFIG_PATH", "/patched/cicerone.toml")

    captured = _run_main(monkeypatch)

    assert captured["config_path"] == "/patched/cicerone.toml"


def test_require_basic_auth_used_directly_rejects_unknown_user():
    # Call dependency directly for the timing-safe unknown-username branch.
    from fastapi import HTTPException
    from fastapi.security import HTTPBasicCredentials

    dependency = require_basic_auth(_users_with("alice", "s3cret"))
    with pytest.raises(HTTPException):
        dependency(HTTPBasicCredentials(username="ghost", password="whatever"))


def test_require_basic_auth_rejects_malformed_hash_and_overlong_password():
    from fastapi import HTTPException
    from fastapi.security import HTTPBasicCredentials

    malformed = require_basic_auth({"alice": "not-a-valid-bcrypt-hash"})
    with pytest.raises(HTTPException) as bad_hash:
        malformed(HTTPBasicCredentials(username="alice", password="s3cret"))
    assert bad_hash.value.status_code == 401

    overlong = require_basic_auth(_users_with("alice", "s3cret"))
    with pytest.raises(HTTPException) as too_long:
        overlong(HTTPBasicCredentials(username="alice", password="x" * 80))
    assert too_long.value.status_code == 401


def test_require_basic_auth_propagates_unexpected_check_error(monkeypatch):
    from fastapi.security import HTTPBasicCredentials

    def _boom(*_args, **_kwargs):
        raise RuntimeError("bcrypt bug")

    monkeypatch.setattr("cicerone.http_auth.bcrypt.checkpw", _boom)
    dependency = require_basic_auth(_users_with("alice", "s3cret"))
    with pytest.raises(RuntimeError, match="bcrypt bug"):
        dependency(HTTPBasicCredentials(username="alice", password="s3cret"))


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
    assert "border-amber-300" in response.text
    assert 'data-run-stale="unknown"' in response.text


def test_page_title_orders_status_then_user():
    from cicerone.dashboard import page_title

    assert page_title() == "Cicerone dashboard"
    assert page_title(user_id=" alice ") == "alice · Cicerone dashboard"
    assert page_title(manifest={"status": "failed"}, user_id="alice") == "failed · alice · Cicerone dashboard"
    assert (
        page_title(manifest={"status": "success"}, staleness={"is_stale": True})
        == "stale · Cicerone dashboard"
    )
    assert page_title(staleness={"is_stale": True}) == "Cicerone dashboard"


def test_status_partial_unknown_status_is_not_success_green():
    manifest = {"status": None, "generated_at": datetime.now(UTC).isoformat()}
    app = create_app(_settings(), _FakeReader(manifest), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/partials/status", auth=("alice", "s3cret"))

    assert response.status_code == 200
    assert "unknown" in response.text
    assert 'data-run-status="unknown"' in response.text
    assert "bg-amber-100" in response.text
    assert "text-amber-800" in response.text
    assert "bg-emerald-100" not in response.text
    assert "border-emerald-300 bg-emerald-50" not in response.text


def test_dashboard_page_failed_run_title():
    manifest = {
        "status": "failed",
        "generated_at": datetime.now(UTC).isoformat(),
        "error": "boom",
    }
    app = create_app(_settings(), _FakeReader(manifest), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard", auth=("alice", "s3cret"))

    assert "<title>failed · Cicerone dashboard</title>" in response.text
    assert "border-red-300 bg-red-50" in response.text


def test_status_partial_history_error_uses_disclosure_not_red_dash():
    manifest = {"status": "success", "generated_at": "2026-07-28T00:00:00+00:00"}
    older = {
        "status": "failed",
        "generated_at": "2026-07-27T00:00:00+00:00",
        "error": "S3 bucket unreachable",
    }
    app = create_app(
        _settings(),
        _FakeReader(manifest, history=[manifest, older]),
        _users_with("alice", "s3cret"),
    )
    response = TestClient(app).get("/partials/status", auth=("alice", "s3cret"))

    assert "<details" in response.text
    assert "S3 bucket unreachable" in response.text
    assert response.text.count('<span class="text-slate-500">—</span>') == 1


class _FailingRecReader(FakeReader):
    def __init__(
        self,
        recs: pd.DataFrame,
        *,
        refresh_error: Exception | None = None,
        lookup_error: Exception | None = None,
    ):
        super().__init__(recs)
        self._refresh_error = refresh_error
        self._lookup_error = lookup_error

    def refresh(self) -> None:
        super().refresh()
        if self._refresh_error is not None:
            raise self._refresh_error

    def get_recommendations(self, user_id: str, k: int, *, variant: str | None = None) -> pd.DataFrame:
        if self._lookup_error is not None:
            raise self._lookup_error
        return super().get_recommendations(user_id, k, variant=variant)


def _recs_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"},
            {"user_id": "u1", "item_id": "i2", "rank": 2, "score": 0.5, "source": "personalized"},
        ]
    )


def _recs_client(
    rec_reader: FakeReader | None = None,
    history_reader: object | None = None,
    **settings_overrides: object,
) -> TestClient:
    app = create_app(
        _settings(**settings_overrides),
        _FakeReader(None),
        _users_with("alice", "s3cret"),
        rec_reader,
        history_reader,
    )
    return TestClient(app)


def _csrf_token(client: TestClient) -> str:
    from cicerone.http_security import CSRF_COOKIE

    response = client.get("/dashboard/experiments", auth=("alice", "s3cret"))
    assert response.status_code == 200
    return str(client.cookies[CSRF_COOKIE])


def _same_origin() -> dict[str, str]:
    return {"origin": "http://testserver"}


def test_recommendations_partial_requires_auth():
    response = _recs_client(FakeReader(_recs_df())).get("/partials/recommendations")

    assert response.status_code == 401


def test_recommendations_partial_empty_user_id_prompts():
    response = _recs_client(FakeReader(_recs_df())).get("/partials/recommendations", auth=("alice", "s3cret"))

    assert response.status_code == 200
    assert "Enter a user id to inspect their recent events and current top-K." in response.text


def test_recommendations_partial_whitespace_user_id_prompts():
    response = _recs_client(FakeReader(_recs_df())).get(
        "/partials/recommendations",
        params={"user_id": "  "},
        auth=("alice", "s3cret"),
    )

    assert "Enter a user id to inspect their recent events and current top-K." in response.text


def test_recommendations_partial_renders_known_user():
    rec_reader = FakeReader(_recs_df())
    response = _recs_client(rec_reader).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert response.status_code == 200
    assert rec_reader.refresh_calls == 1
    assert "Recommendations for" in response.text
    assert ">i1<" in response.text
    assert ">i2<" in response.text
    assert "0.9000" in response.text
    assert "personalized" in response.text
    assert "cold-start fallback" not in response.text


def test_dashboard_page_user_id_query_renders_lookup_results():
    rec_reader = FakeReader(_recs_df())
    response = _recs_client(rec_reader).get(
        "/dashboard",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert response.status_code == 200
    assert 'value="u1"' in response.text
    assert ">i1<" in response.text
    assert "Inspect user" in response.text
    assert "Current top-K recommendations for u1" in response.text
    assert 'scope="col"' in response.text
    assert "<title>u1 · Cicerone dashboard</title>" in response.text


def test_recommendations_partial_unknown_user_uses_fallback():
    fallback = pd.DataFrame(
        [
            {
                "user_id": "__cold_start__",
                "item_id": "i9",
                "rank": 1,
                "score": 0.4,
                "source": "popular_fallback",
            }
        ]
    )
    response = _recs_client(FakeReader(pd.concat([_recs_df(), fallback], ignore_index=True))).get(
        "/partials/recommendations",
        params={"user_id": "ghost"},
        auth=("alice", "s3cret"),
    )

    assert response.status_code == 200
    assert "cold-start fallback" in response.text
    assert "popular-item stand-ins" in response.text
    assert ">i9<" in response.text
    assert "popular_fallback" in response.text


def test_recommendations_partial_unknown_user_without_fallback():
    response = _recs_client(FakeReader(_recs_df())).get(
        "/partials/recommendations",
        params={"user_id": "ghost"},
        auth=("alice", "s3cret"),
    )

    assert "No recommendations for ghost." in response.text
    assert "cold-start fallback" not in response.text


def test_recommendations_partial_joins_category_from_items():
    items = pd.DataFrame([{"item_id": "i1", "category": "beer"}, {"item_id": "i2", "category": "wine"}])
    response = _recs_client(FakeReader(_recs_df(), items=items)).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert "Category" in response.text
    assert "beer" in response.text
    assert "wine" in response.text


def test_recommendations_partial_uses_category_already_on_recs():
    recs = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "rank": 1,
                "score": 0.9,
                "source": "personalized",
                "category": "stout",
            }
        ]
    )
    response = _recs_client(FakeReader(recs)).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert "Category" in response.text
    assert "stout" in response.text


def test_recommendations_partial_items_without_category_omits_column():
    items = pd.DataFrame([{"item_id": "i1", "name": "lager"}])
    response = _recs_client(FakeReader(_recs_df(), items=items)).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert "Category" not in response.text
    assert "lager" not in response.text


def test_recommendations_partial_empty_items_snapshot_omits_category():
    items = pd.DataFrame(columns=["item_id", "category"])
    response = _recs_client(FakeReader(_recs_df(), items=items)).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert "Category" not in response.text


@pytest.mark.parametrize("overrides", [{"top_k": 50}, {"top_k": 50, "dashboard_lookup_k": 5}])
def test_recommendations_partial_shows_serve_default_k_rows(overrides):
    recs = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": f"i{i}",
                "rank": i,
                "score": 1.0,
                "source": "personalized",
            }
            for i in range(1, 26)
        ]
    )
    response = _recs_client(FakeReader(recs), serve_default_k=7, **overrides).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert ">i7<" in response.text
    assert ">i8<" not in response.text


def test_recommendations_partial_without_reader_shows_error():
    response = _recs_client(None).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert "Recommendation store is not available." in response.text


def test_recommendations_partial_refreshes_reader_once_per_ttl(refresh_clock):
    from cicerone.dashboard import _LOOKUP_REFRESH_TTL_SECONDS

    rec_reader = FakeReader(_recs_df())
    client = _recs_client(rec_reader)

    def lookup(user_id: str) -> None:
        response = client.get(
            "/partials/recommendations", params={"user_id": user_id}, auth=("alice", "s3cret")
        )
        assert response.status_code == 200

    lookup("u1")
    lookup("u1")
    lookup("")
    assert rec_reader.refresh_calls == 1
    refresh_clock.now += _LOOKUP_REFRESH_TTL_SECONDS
    lookup("u1")
    assert rec_reader.refresh_calls == 2


def test_recommendations_partial_refresh_error_still_looks_up():
    rec_reader = _FailingRecReader(_recs_df(), refresh_error=OSError("cache boom"))
    response = _recs_client(rec_reader).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert response.status_code == 200
    assert ">i1<" in response.text


def test_recommendations_partial_lookup_error_shows_message():
    rec_reader = _FailingRecReader(_recs_df(), lookup_error=OSError("store boom"))
    response = _recs_client(rec_reader).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert LOOKUP_FAILED in response.text
    assert "store boom" not in response.text
    assert ">i1<" not in response.text


def test_recommendations_partial_missing_score_source_render_dashes():
    recs = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": None, "score": None, "source": None}])
    response = _recs_client(FakeReader(recs)).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert ">i1<" in response.text
    # Rank is renumbered like serve; score and source stay missing.
    assert response.text.count(f">{MISSING}<") == 2
    assert "None" not in response.text
    assert "nan" not in response.text


class _FakeHistory:
    def __init__(self, events: pd.DataFrame, user: dict | None = None):
        self._events = events
        self._user = user

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
        rows = self._events[self._events["user_id"] == user_id]
        return rows.head(limit).reset_index(drop=True)

    def get_user(self, user_id: str) -> dict | None:
        if self._user is None or self._user.get("user_id") != user_id:
            return None
        return self._user


def _events_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 2,
                "occurred_at": "2026-08-21T12:00:00+00:00",
            },
            {
                "user_id": "u1",
                "item_id": "i9",
                "event_type": "view",
                "quantity": 1,
                "occurred_at": "2026-08-20T12:00:00+00:00",
            },
        ]
    )


def test_recommendations_partial_without_history_reader_shows_unavailable():
    response = _recs_client(FakeReader(_recs_df())).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert HISTORY_UNAVAILABLE in response.text
    assert ">i1<" in response.text


def test_recommendations_partial_renders_events_and_overlap():
    response = _recs_client(
        FakeReader(_recs_df()),
        _FakeHistory(_events_df(), {"user_id": "u1", "region_slug": "lazio"}),
        dashboard_lookup_user_attrs=("region_slug",),
    ).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert response.status_code == 200
    assert "Recent events" in response.text
    assert "purchase" in response.text
    assert ">i9<" in response.text
    assert "warm" in response.text
    assert "personalized 2" in response.text
    assert "lazio" in response.text
    assert "bg-amber-50" in response.text
    assert "Also in recommendations" in response.text
    assert "Also in recent events" in response.text
    assert "Highlighted rows appear in both events and recommendations." in response.text


def test_recommendations_partial_no_events_badge():
    empty = pd.DataFrame(columns=["user_id", "item_id", "event_type", "quantity", "occurred_at"])
    response = _recs_client(FakeReader(_recs_df()), _FakeHistory(empty)).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert "no events" in response.text
    assert "No events for u1." in response.text
    assert HISTORY_UNAVAILABLE not in response.text


def test_dashboard_experiments_page_disabled():
    response = _recs_client().get("/dashboard/experiments", auth=("alice", "s3cret"))
    assert response.status_code == 200
    assert "No experiment is running." in response.text
    assert "How to enable one" in response.text
    assert "<title>Experiments · Cicerone dashboard</title>" in response.text
    assert "Sign out" in response.text
    assert ">off<" in response.text
    assert 'href="/dashboard/experiments"' in response.text
    assert 'href="/dashboard/quality"' in response.text
    assert 'href="/dashboard/config"' in response.text


def test_dashboard_lookup_shows_assigned_variant():
    recs = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "control-item",
                "rank": 1,
                "score": 0.9,
                "source": "personalized",
                "variant": "control",
            },
            {
                "user_id": "u1",
                "item_id": "treatment-item",
                "rank": 1,
                "score": 0.8,
                "source": "personalized",
                "variant": "treatment",
            },
        ]
    )
    from cicerone.config.settings import ExperimentSettings, VariantSettings
    from cicerone.experiment.assignment import assign_variant

    experiment = ExperimentSettings(
        enabled=True,
        id="exp-1",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    assigned = assign_variant("exp-1", "u1", (("control", 0.5), ("treatment", 0.5)))
    response = _recs_client(FakeReader(recs), experiment=experiment).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )
    assert response.status_code == 200
    assert f"exp-1 / {assigned}" in response.text
    assert f">{assigned}-item<" in response.text
    other = "treatment" if assigned == "control" else "control"
    assert f">{other}-item<" not in response.text


def test_dashboard_lookup_omits_variant_without_column():
    recs = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "control-item",
                "rank": 1,
                "score": 0.9,
                "source": "personalized",
            },
            {
                "user_id": "u1",
                "item_id": "treatment-item",
                "rank": 2,
                "score": 0.8,
                "source": "personalized",
            },
        ]
    )
    from cicerone.config.settings import ExperimentSettings, VariantSettings

    experiment = ExperimentSettings(
        enabled=True,
        id="exp-1",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    response = _recs_client(FakeReader(recs), experiment=experiment).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )
    assert response.status_code == 200
    assert "exp-1 /" not in response.text
    assert ">control-item<" in response.text
    assert ">treatment-item<" in response.text


def test_dashboard_inspect_matches_serve_response(tmp_path):
    import re

    from cicerone.config.settings import ExperimentSettings, IOSettings, VariantSettings
    from cicerone.experiment.assignment import assign_variant
    from cicerone.feature_config import FeatureConfig
    from cicerone.serve import create_app as create_serve_app

    variants = (("control", 0.5), ("treatment", 0.5))
    user_id = next(f"u{i}" for i in range(100) if assign_variant("exp-1", f"u{i}", variants) == "treatment")
    recs = pd.DataFrame(
        [
            {
                "user_id": user_id,
                "item_id": item,
                "rank": rank,
                "score": score,
                "source": "als",
                "variant": "control",
            }
            for item, rank, score in (("i1", 10, 0.9), ("i3", 20, 0.8), ("i2", 30, 0.7))
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "published": True},
            {"item_id": "i2", "category": "wine", "published": True},
            {"item_id": "i3", "category": "beer", "published": False},
        ]
    )
    reader = FakeReader(recs, items=items)
    feature_config = FeatureConfig(
        event_weights={},
        quantity_scaled_events=set(),
        event_caps={},
        user_features=[],
        item_features=[],
        item_availability_filters=["published"],
    )
    settings = _settings(
        mode="serve",
        serve_auth_token="secret",
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
        experiment=ExperimentSettings(
            enabled=True,
            id="exp-1",
            variants=tuple(VariantSettings(name=name, traffic=traffic) for name, traffic in variants),
        ),
    )
    served = TestClient(create_serve_app(settings, reader, feature_config=feature_config)).get(
        f"/recommendations/{user_id}", headers={"Authorization": "Bearer secret"}
    )
    inspect = TestClient(
        create_app(
            settings, _FakeReader(None), _users_with("alice", "s3cret"), reader, feature_config=feature_config
        )
    ).get("/partials/recommendations", params={"user_id": user_id}, auth=("alice", "s3cret"))

    body = served.json()
    rows = re.findall(
        r'<td class="py-2 px-2 text-slate-700">(\S+)</td>\s*<th scope="row"[^>]*>([^<]+)</th>\s*'
        r'<td class="py-2 px-2 text-slate-700">(\S+)</td>\s*'
        r'<td class="py-2 px-2 text-slate-700">(\S+)</td>\s*'
        r'<td class="py-2 px-2 text-slate-700">(\S+)</td>',
        inspect.text,
    )
    assert [(rank, item) for rank, item, _category, _score, _source in rows] == [
        (str(row["rank"]), row["item_id"]) for row in body["items"]
    ]
    assert [(score, source) for _rank, _item, _category, score, source in rows] == [
        (f"{row['score']:.4f}", row["source"]) for row in body["items"]
    ]
    assert [row["item_id"] for row in body["items"]] == ["i1", "i2"]
    assert [row["rank"] for row in body["items"]] == [1, 2]
    assert [category for _rank, _item, category, _score, _source in rows] == ["beer", "wine"]
    assert (body["fallback"], body["experiment_id"], body["variant"]) == (False, "exp-1", "control")
    assert 'data-fallback="false"' in inspect.text
    assert "exp-1 / control" in inspect.text


def test_dashboard_promote_unknown_variant_redirects():
    from cicerone.config.settings import ExperimentSettings, VariantSettings

    experiment = ExperimentSettings(
        enabled=True,
        id="exp-1",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    client = _recs_client(experiment=experiment)
    token = _csrf_token(client)
    response = client.post(
        "/dashboard/experiments/promote",
        data={"variant": "ghost", "csrf_token": token},
        auth=("alice", "s3cret"),
        headers=_same_origin(),
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard/experiments"
    assert _flash_cookie(response).startswith("err:")


@pytest.mark.parametrize(
    ("error", "cookie"),
    [
        ("Writer lock is busy", "err:Writer%20lock%20is%20busy"),
        ("Writer lock was lost", "err:Writer%20lock%20was%20lost"),
    ],
)
def test_dashboard_promote_lock_error_flashes_the_lock_message(monkeypatch, error, cookie):
    import cicerone.dashboard as dash
    from cicerone.config.settings import ExperimentSettings, VariantSettings

    monkeypatch.setattr(dash, "promote_winner", lambda _settings, _variant: error)
    experiment = ExperimentSettings(
        enabled=True,
        id="exp-1",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    client = _recs_client(experiment=experiment)
    response = client.post(
        "/dashboard/experiments/promote",
        data={"variant": "control", "csrf_token": _csrf_token(client)},
        auth=("alice", "s3cret"),
        headers=_same_origin(),
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert _flash_cookie(response) == cookie
    page = client.get("/dashboard/experiments", auth=("alice", "s3cret"))
    assert error in page.text


def test_dashboard_promote_hostile_variant_stays_on_experiments():
    from cicerone.config.settings import ExperimentSettings, VariantSettings

    experiment = ExperimentSettings(
        enabled=True,
        id="exp-1",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    client = _recs_client(experiment=experiment)
    token = _csrf_token(client)
    response = client.post(
        "/dashboard/experiments/promote",
        data={"variant": "https://evil.example/phish", "csrf_token": token},
        auth=("alice", "s3cret"),
        headers=_same_origin(),
        follow_redirects=False,
    )
    assert response.status_code == 303
    location = response.headers["location"]
    assert location == "/dashboard/experiments"
    assert "://" not in location


def test_dashboard_promote_success_redirects(monkeypatch):
    import cicerone.dashboard as dash
    from cicerone.config.settings import ExperimentSettings, VariantSettings

    monkeypatch.setattr(dash, "promote_winner", lambda _settings, _variant: None)
    experiment = ExperimentSettings(
        enabled=True,
        id="exp-1",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    client = _recs_client(experiment=experiment)
    token = _csrf_token(client)
    response = client.post(
        "/dashboard/experiments/promote",
        data={"variant": "control", "csrf_token": token},
        auth=("alice", "s3cret"),
        headers=_same_origin(),
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard/experiments"
    flash = _flash_cookie(response)
    assert flash.startswith("ok:Promoted")
    assert "control" not in flash


def test_dashboard_unpromote_resumes_split(tmp_path):
    from cicerone.config import IOSettings
    from cicerone.config.settings import ExperimentSettings, VariantSettings

    out = tmp_path / "out"
    out.mkdir()
    experiment = ExperimentSettings(
        enabled=True,
        id="exp-1",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    client = _recs_client(
        experiment=experiment,
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
    )
    token = _csrf_token(client)
    response = client.post(
        "/dashboard/experiments/unpromote",
        data={"csrf_token": token},
        auth=("alice", "s3cret"),
        headers=_same_origin(),
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard/experiments"
    assert "Resumed" in (response.cookies.get("cicerone_flash") or "")


def test_dashboard_unpromote_disabled_experiment_redirects():
    client = _recs_client()
    token = _csrf_token(client)
    response = client.post(
        "/dashboard/experiments/unpromote",
        data={"csrf_token": token},
        auth=("alice", "s3cret"),
        headers=_same_origin(),
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard/experiments"
    assert _flash_cookie(response).startswith("err:")


def test_dashboard_promote_rejects_missing_csrf():
    from cicerone.config.settings import ExperimentSettings, VariantSettings

    experiment = ExperimentSettings(
        enabled=True,
        id="exp-1",
        variants=(
            VariantSettings(name="control", traffic=0.5),
            VariantSettings(name="treatment", traffic=0.5),
        ),
    )
    client = _recs_client(experiment=experiment)
    _csrf_token(client)
    response = client.post(
        "/dashboard/experiments/promote",
        data={"variant": "control"},
        auth=("alice", "s3cret"),
        headers=_same_origin(),
        follow_redirects=False,
    )
    assert response.status_code == 403


def test_dashboard_docs_are_disabled():
    client = _recs_client()
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_dashboard_security_headers_on_health():
    response = TestClient(create_app(_settings(), _FakeReader(None), _users_with("alice", "s3cret"))).get(
        "/health"
    )
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_dashboard_malformed_bcrypt_hash_is_unauthorized():
    app = create_app(_settings(), _FakeReader(None), {"alice": "not-a-valid-bcrypt-hash"})
    response = TestClient(app).get("/dashboard", auth=("alice", "s3cret"))
    assert response.status_code == 401


def test_dashboard_logout_returns_401():
    app = create_app(_settings(), _FakeReader(None), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard/logout")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Basic"
    assert "Signed out" in response.text
    assert "clear the saved password" in response.text


def test_experiments_flash_cookie_is_one_shot():
    client = _recs_client()
    client.cookies.set("cicerone_flash", "ok:Promoted control")
    first = client.get("/dashboard/experiments", auth=("alice", "s3cret"))
    assert first.status_code == 200
    assert "Promoted control" in first.text
    assert "cicerone_flash" in first.headers.get("set-cookie", "").lower()
    client.cookies.pop("cicerone_flash", None)
    second = client.get("/dashboard/experiments", auth=("alice", "s3cret"))
    assert "Promoted control" not in second.text


def test_as_percent_formats_rates():
    from cicerone.dashboard import _as_percent, _chrome
    from cicerone.dashboard_experiments import _lift_label

    assert _as_percent(0.2) == "20.00%"
    assert _as_percent("0.0123") == "1.23%"
    assert _as_percent(None) == "—"
    assert _as_percent("nope") == "—"
    chrome = _chrome(_settings())
    assert chrome["nav_quality_available"] is False
    assert chrome["nav_experiments_available"] is False
    assert _lift_label("ctr") == "CTR lift"
    assert _lift_label("conversion") == "Conversion lift"
    assert _lift_label("weighted") == "Mean lift"
