from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from conftest import make_settings
from fastapi.testclient import TestClient

from cicerone.config import Settings
from cicerone.dashboard import create_app, main
from cicerone.dashboard_lookup import LOOKUP_FAILED, MISSING
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
    assert "Look up recommendations" in response.text
    assert "Enter a user id to inspect their current top-K." in response.text
    assert 'for="user-id"' in response.text
    assert 'aria-live="polite"' in response.text
    assert 'href="#main"' in response.text
    assert "hx-disabled-elt=\"button[type='submit']\"" in response.text


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


def test_status_partial_shows_incremental_panel_when_events_enabled():
    from cicerone.config import EventsSettings

    history = [
        {
            "status": "success",
            "generated_at": "2026-08-13T15:00:00+00:00",
            "triggered_by": "incremental",
            "last_incremental_at": "2026-08-13T15:00:00+00:00",
            "incremental_events_applied": 3,
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
    assert "kind=webhook" in response.text
    assert "3" in response.text
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


def test_main_starts_when_recommendation_reader_fails(monkeypatch):
    captured: dict[str, object] = {}

    def fake_create_app(settings, reader, users, rec_reader=None):
        captured["rec_reader"] = rec_reader
        return object()

    def boom(_output):
        raise RuntimeError("bad store")

    monkeypatch.setattr("cicerone.dashboard.load_settings", lambda: _settings())
    monkeypatch.setattr("cicerone.dashboard.load_users", lambda _path: {"alice": "hash"})
    monkeypatch.setattr("cicerone.dashboard.build_manifest_reader", lambda _output: _FakeReader(None))
    monkeypatch.setattr("cicerone.dashboard.build_recommendation_reader", boom)
    monkeypatch.setattr("cicerone.dashboard.create_app", fake_create_app)
    monkeypatch.setattr(
        "cicerone.dashboard.uvicorn",
        type("_Uvicorn", (), {"run": staticmethod(lambda *_a, **_k: None)}),
    )

    main()

    assert captured["rec_reader"] is None


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


class _FakeRecReader:
    def __init__(
        self,
        recs: pd.DataFrame,
        items: pd.DataFrame | None = None,
        fallback: pd.DataFrame | None = None,
        *,
        refresh_error: Exception | None = None,
        lookup_error: Exception | None = None,
    ):
        self._recs = recs
        self._items = items
        self._fallback = fallback if fallback is not None else pd.DataFrame()
        self._refresh_error = refresh_error
        self._lookup_error = lookup_error
        self.refresh_calls = 0

    def refresh(self) -> None:
        self.refresh_calls += 1
        if self._refresh_error is not None:
            raise self._refresh_error

    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame:
        if self._lookup_error is not None:
            raise self._lookup_error
        rows = self._recs[self._recs["user_id"] == user_id]
        if "rank" in rows.columns:
            rows = rows.sort_values("rank")
        return rows.head(k).reset_index(drop=True)

    def get_items(self) -> pd.DataFrame | None:
        return self._items

    def get_cold_start_fallback(self, k: int) -> pd.DataFrame:
        return self._fallback.head(k).reset_index(drop=True)


def _recs_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"},
            {"user_id": "u1", "item_id": "i2", "rank": 2, "score": 0.5, "source": "personalized"},
        ]
    )


def _recs_client(rec_reader: _FakeRecReader | None = None, **settings_overrides: object) -> TestClient:
    app = create_app(
        _settings(**settings_overrides),
        _FakeReader(None),
        _users_with("alice", "s3cret"),
        rec_reader,
    )
    return TestClient(app)


def test_recommendations_partial_requires_auth():
    response = _recs_client(_FakeRecReader(_recs_df())).get("/partials/recommendations")

    assert response.status_code == 401


def test_recommendations_partial_empty_user_id_prompts():
    response = _recs_client(_FakeRecReader(_recs_df())).get(
        "/partials/recommendations", auth=("alice", "s3cret")
    )

    assert response.status_code == 200
    assert "Enter a user id to inspect their current top-K." in response.text


def test_recommendations_partial_whitespace_user_id_prompts():
    response = _recs_client(_FakeRecReader(_recs_df())).get(
        "/partials/recommendations",
        params={"user_id": "  "},
        auth=("alice", "s3cret"),
    )

    assert "Enter a user id to inspect their current top-K." in response.text


def test_recommendations_partial_renders_known_user():
    rec_reader = _FakeRecReader(_recs_df())
    response = _recs_client(rec_reader).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert response.status_code == 200
    assert rec_reader.refresh_calls == 1
    assert "user_id=" in response.text
    assert ">i1<" in response.text
    assert ">i2<" in response.text
    assert "0.9000" in response.text
    assert "personalized" in response.text
    assert "cold-start fallback" not in response.text


def test_dashboard_page_user_id_query_renders_lookup_results():
    rec_reader = _FakeRecReader(_recs_df())
    response = _recs_client(rec_reader).get(
        "/dashboard",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert response.status_code == 200
    assert 'value="u1"' in response.text
    assert ">i1<" in response.text
    assert "Look up recommendations" in response.text
    assert "Current top-K recommendations for u1" in response.text
    assert 'scope="col"' in response.text


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
    response = _recs_client(_FakeRecReader(_recs_df(), fallback=fallback)).get(
        "/partials/recommendations",
        params={"user_id": "ghost"},
        auth=("alice", "s3cret"),
    )

    assert response.status_code == 200
    assert "cold-start fallback" in response.text
    assert ">i9<" in response.text
    assert "popular_fallback" in response.text


def test_recommendations_partial_unknown_user_without_fallback():
    response = _recs_client(_FakeRecReader(_recs_df())).get(
        "/partials/recommendations",
        params={"user_id": "ghost"},
        auth=("alice", "s3cret"),
    )

    assert "No recommendations for user_id=ghost." in response.text
    assert "cold-start fallback" not in response.text


def test_recommendations_partial_joins_category_from_items():
    items = pd.DataFrame([{"item_id": "i1", "category": "beer"}, {"item_id": "i2", "category": "wine"}])
    response = _recs_client(_FakeRecReader(_recs_df(), items=items)).get(
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
    response = _recs_client(_FakeRecReader(recs)).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert "Category" in response.text
    assert "stout" in response.text


def test_recommendations_partial_items_without_category_omits_column():
    items = pd.DataFrame([{"item_id": "i1", "name": "lager"}])
    response = _recs_client(_FakeRecReader(_recs_df(), items=items)).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert "Category" not in response.text
    assert "lager" not in response.text


def test_recommendations_partial_empty_items_snapshot_omits_category():
    items = pd.DataFrame(columns=["item_id", "category"])
    response = _recs_client(_FakeRecReader(_recs_df(), items=items)).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert "Category" not in response.text


def test_recommendations_partial_caps_k_at_20():
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
    response = _recs_client(_FakeRecReader(recs), top_k=50).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert ">i20<" in response.text
    assert ">i21<" not in response.text


def test_recommendations_partial_respects_dashboard_lookup_k():
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
    response = _recs_client(_FakeRecReader(recs), top_k=50, dashboard_lookup_k=5).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert ">i5<" in response.text
    assert ">i6<" not in response.text


def test_recommendations_partial_without_reader_shows_error():
    response = _recs_client(None).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert "Recommendation store is not available." in response.text


def test_recommendations_partial_refresh_error_still_looks_up():
    rec_reader = _FakeRecReader(_recs_df(), refresh_error=RuntimeError("cache boom"))
    response = _recs_client(rec_reader).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert response.status_code == 200
    assert ">i1<" in response.text


def test_recommendations_partial_lookup_error_shows_message():
    rec_reader = _FakeRecReader(_recs_df(), lookup_error=RuntimeError("store boom"))
    response = _recs_client(rec_reader).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert LOOKUP_FAILED in response.text
    assert "store boom" not in response.text
    assert ">i1<" not in response.text


def test_recommendations_partial_missing_rank_score_source_render_dashes():
    recs = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": None, "score": None, "source": None}])
    response = _recs_client(_FakeRecReader(recs)).get(
        "/partials/recommendations",
        params={"user_id": "u1"},
        auth=("alice", "s3cret"),
    )

    assert ">i1<" in response.text
    assert response.text.count(f">{MISSING}<") == 3
    assert "None" not in response.text
    assert "nan" not in response.text
