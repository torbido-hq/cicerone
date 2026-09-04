from __future__ import annotations

import re

from conftest import make_settings
from fastapi.testclient import TestClient

from cicerone.config import IOSettings
from cicerone.dashboard import ROBOTS_TAG, create_app
from cicerone.http_security import CSRF_COOKIE
from cicerone.track.store import TrackStore


def _users_with(username: str, password: str) -> dict[str, str]:
    import bcrypt

    return {username: bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")}


class _FakeReader:
    def read_latest(self):
        return None

    def read_recent(self, limit: int):
        del limit
        return []


def _settings(tmp_path, **overrides):
    return make_settings(
        **{
            "dashboard_enabled": True,
            "output": IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
            "input": IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
            **overrides,
        }
    )


_QUALITY_CURRENT = re.compile(
    r'<a\b[^>]*href="/dashboard/quality"[^>]*aria-current="page"[^>]*>\s*Quality\s*</a>',
    re.DOTALL,
)


def _assert_quality_chrome(response) -> None:
    assert response.status_code == 200
    assert "<title>Cicerone — Quality</title>" in response.text
    assert _QUALITY_CURRENT.search(response.text)
    assert 'href="/dashboard"' in response.text
    assert 'href="/dashboard/experiments"' in response.text
    assert 'href="/dashboard/config"' in response.text
    assert 'aria-label="Main"' in response.text
    assert 'name="robots"' in response.text
    assert f'content="{ROBOTS_TAG}"' in response.text
    assert CSRF_COOKIE in response.cookies


def test_quality_page_empty_when_track_off(tmp_path):
    app = create_app(_settings(tmp_path), _FakeReader(), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard/quality", auth=("alice", "s3cret"))
    _assert_quality_chrome(response)
    assert "[track]" in response.text
    assert "/track" in response.text
    assert 'aria-labelledby="quality-track-heading"' in response.text


def test_quality_page_requires_auth(tmp_path):
    app = create_app(_settings(tmp_path), _FakeReader(), _users_with("alice", "s3cret"))
    assert TestClient(app).get("/dashboard/quality").status_code == 401


def test_quality_page_shows_stored_metrics(tmp_path):
    settings = _settings(tmp_path, track={"enabled": True})
    TrackStore(settings.output).write_eval(
        {
            "track_eval": {
                "overall": {
                    "n_impressions": 10,
                    "n_clicks": 2,
                    "n_conversions_click": 1,
                    "n_conversions_view": 1,
                    "ctr": 0.2,
                    "cvr_click": 0.1,
                    "cvr_view": 0.1,
                    "n_users": 4,
                },
                "by_rank": {
                    "1": {
                        "n_impressions": 5,
                        "n_clicks": 2,
                        "n_conversions_click": 1,
                        "n_conversions_view": 1,
                        "ctr": 0.4,
                        "cvr_click": 0.2,
                        "cvr_view": 0.2,
                        "n_users": 4,
                    }
                },
                "by_source": {},
                "by_variant": {},
            }
        }
    )
    app = create_app(settings, _FakeReader(), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard/quality", auth=("alice", "s3cret"))
    _assert_quality_chrome(response)
    assert "Impressions: 10" in response.text
    assert "By rank" in response.text
    assert "0.2000" in response.text
    assert "CTR and conversion by rank" in response.text


def test_quality_page_live_metrics_from_track_rows(tmp_path):
    from cicerone.track.normalize import normalize_track

    settings = _settings(tmp_path, track={"enabled": True})
    TrackStore(settings.output).append_rows(
        [
            normalize_track(
                {
                    "kind": "impression",
                    "user_id": "u1",
                    "item_id": "i1",
                    "rank": 1,
                    "occurred_at": "2026-08-28T12:00:00Z",
                    "event_id": "imp-1",
                }
            ).as_row()
        ]
    )
    app = create_app(settings, _FakeReader(), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard/quality", auth=("alice", "s3cret"))
    assert response.status_code == 200
    assert "Impressions: 1" in response.text


def test_quality_live_eval_error_falls_back_to_empty(tmp_path, monkeypatch):
    from cicerone.dashboard_quality import quality_context
    from cicerone.track.normalize import normalize_track

    settings = _settings(tmp_path, track={"enabled": True})
    TrackStore(settings.output).append_rows(
        [
            normalize_track(
                {
                    "kind": "impression",
                    "user_id": "u1",
                    "item_id": "i1",
                    "rank": 1,
                    "occurred_at": "2026-08-28T12:00:00Z",
                }
            ).as_row()
        ]
    )
    monkeypatch.setattr(
        "cicerone.dashboard_quality.evaluate_tracking",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("eval failed")),
    )
    context = quality_context(settings)
    assert context["empty_track"] is True
    assert context["track_eval"] is None


def test_quality_live_eval_read_rows_error(tmp_path, monkeypatch):
    from cicerone.dashboard_quality import quality_context

    settings = _settings(tmp_path, track={"enabled": True})
    monkeypatch.setattr(
        "cicerone.track.store.TrackStore.read_rows",
        lambda self, **_kwargs: (_ for _ in ()).throw(RuntimeError("rows")),
    )
    context = quality_context(settings)
    assert context["empty_track"] is True
    assert context["track_eval"] is None


def test_quality_live_eval_conversion_load_error_still_scores(tmp_path, monkeypatch):
    from cicerone.dashboard_quality import quality_context
    from cicerone.track.normalize import normalize_track

    settings = _settings(tmp_path, track={"enabled": True})
    TrackStore(settings.output).append_rows(
        [
            normalize_track(
                {
                    "kind": "impression",
                    "user_id": "u1",
                    "item_id": "i1",
                    "rank": 1,
                    "occurred_at": "2026-08-28T12:00:00Z",
                    "event_id": "imp-live",
                }
            ).as_row()
        ]
    )
    monkeypatch.setattr(
        "cicerone.dashboard_experiments._load_metric_events",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("events")),
    )
    context = quality_context(settings)
    assert context["empty_track"] is False
    assert context["track_eval"]["overall"]["n_impressions"] == 1


def test_quality_page_shows_stored_served_eval(tmp_path):
    settings = _settings(tmp_path, track={"enabled": True}, eval={"enabled": True})
    TrackStore(settings.output).write_eval(
        {
            "track_eval": {
                "overall": {
                    "n_impressions": 8,
                    "n_clicks": 2,
                    "n_conversions_click": 1,
                    "n_conversions_view": 1,
                    "ctr": 0.25,
                    "cvr_click": 0.125,
                    "cvr_view": 0.125,
                    "n_users": 3,
                },
                "by_rank": {},
                "by_source": {
                    "personalized": {
                        "n_impressions": 8,
                        "n_clicks": 2,
                        "n_conversions_click": 1,
                        "n_conversions_view": 1,
                        "ctr": 0.25,
                        "cvr_click": 0.125,
                        "cvr_view": 0.125,
                        "n_users": 3,
                    }
                },
                "by_variant": {
                    "control": {
                        "n_impressions": 8,
                        "n_clicks": 2,
                        "n_conversions_click": 1,
                        "n_conversions_view": 1,
                        "ctr": 0.25,
                        "cvr_click": 0.125,
                        "cvr_view": 0.125,
                        "n_users": 3,
                    }
                },
            },
            "served_eval": {
                "n_users": 3,
                "n_users_with_events": 1,
                "generated_at": "2026-08-28T03:00:00+00:00",
                "metrics": {"HitRate@10": 0.5, "NDCG@10": 0.4},
                "by_source": {"personalized": {"HitRate@10": 0.5}},
            },
        }
    )
    app = create_app(settings, _FakeReader(), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard/quality", auth=("alice", "s3cret"))
    _assert_quality_chrome(response)
    assert "By source" in response.text
    assert "By variant" in response.text
    assert "Production replay" in response.text
    assert "HitRate@10" in response.text
    assert "CTR and conversion by source" in response.text
    assert "CTR and conversion by variant" in response.text
    assert "Production replay ranking metrics" in response.text
    assert "Production replay hit rate by source" in response.text
    assert 'aria-labelledby="quality-replay-heading"' in response.text


def test_quality_eval_enabled_empty_replay_copy(tmp_path):
    settings = _settings(tmp_path, eval={"enabled": True})
    app = create_app(settings, _FakeReader(), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard/quality", auth=("alice", "s3cret"))
    assert "Production replay is on" in response.text


def test_quality_context_handles_read_errors(tmp_path, monkeypatch):
    from cicerone.dashboard_quality import quality_context

    settings = _settings(tmp_path, track={"enabled": True})

    def _boom(self):
        raise RuntimeError("nope")

    monkeypatch.setattr("cicerone.track.store.TrackStore.read_eval", _boom)
    context = quality_context(settings)
    assert context["error"]
    assert context["empty_track"] is True


def test_quality_live_eval_with_conversions(tmp_path):
    import pandas as pd

    from cicerone.dashboard_quality import quality_context
    from cicerone.track.normalize import normalize_track

    settings = _settings(tmp_path, track={"enabled": True})
    pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": "2026-08-28T13:00:00Z",
            }
        ]
    ).to_parquet(tmp_path / "events.parquet", index=False)
    TrackStore(settings.output).append_rows(
        [
            normalize_track(
                {
                    "kind": "impression",
                    "user_id": "u1",
                    "item_id": "i1",
                    "rank": 1,
                    "occurred_at": "2026-08-28T12:00:00Z",
                    "event_id": "imp-conv",
                }
            ).as_row()
        ]
    )
    context = quality_context(settings)
    assert context["track_eval"]["overall"]["n_impressions"] == 1
    assert context["track_eval"]["overall"]["n_conversions_view"] == 1


def test_quality_live_eval_joins_history_when_current_recs_missing(tmp_path):
    import pandas as pd

    from cicerone.dashboard_quality import quality_context
    from cicerone.track.normalize import normalize_track

    settings = _settings(tmp_path, track={"enabled": True})
    stamp = "2026-08-28T03:00:00+00:00"
    store = TrackStore(settings.output)
    store.append_rows(
        [
            normalize_track(
                {
                    "kind": "impression",
                    "user_id": "u1",
                    "item_id": "i1",
                    "rank": 1,
                    "occurred_at": "2026-08-28T12:00:00Z",
                    "generated_at": stamp,
                    "event_id": "imp-hist",
                }
            ).as_row()
        ]
    )
    store.append_history(
        pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "source": "personalized"}]),
        generated_at=stamp,
    )
    context = quality_context(settings)
    assert context["empty_track"] is False
    assert context["track_eval"]["by_source"]["personalized"]["n_impressions"] == 1


def test_quality_live_eval_concats_history_with_current_recs(tmp_path):
    import pandas as pd

    from cicerone.dashboard_quality import quality_context
    from cicerone.track.normalize import normalize_track

    settings = _settings(tmp_path, track={"enabled": True})
    stamp = "2026-08-28T03:00:00+00:00"
    store = TrackStore(settings.output)
    store.append_rows(
        [
            normalize_track(
                {
                    "kind": "impression",
                    "user_id": "u1",
                    "item_id": "hist-item",
                    "rank": 1,
                    "occurred_at": "2026-08-28T12:00:00Z",
                    "generated_at": stamp,
                    "event_id": "imp-hist-concat",
                }
            ).as_row()
        ]
    )
    store.append_history(
        pd.DataFrame([{"user_id": "u1", "item_id": "hist-item", "rank": 1, "source": "personalized"}]),
        generated_at=stamp,
    )
    pd.DataFrame([{"user_id": "u2", "item_id": "live-item", "rank": 1, "source": "popular"}]).to_parquet(
        tmp_path / "recommendations.parquet", index=False
    )
    context = quality_context(settings)
    assert context["empty_track"] is False
    assert context["track_eval"]["by_source"]["personalized"]["n_impressions"] == 1


def test_quality_context_no_impressions_malformed_overall(tmp_path):
    from cicerone.dashboard_quality import quality_context

    settings = _settings(tmp_path, track={"enabled": True})
    TrackStore(settings.output).write_eval({"track_eval": {"overall": "nope"}})
    context = quality_context(settings)
    assert context["empty_track"] is True
