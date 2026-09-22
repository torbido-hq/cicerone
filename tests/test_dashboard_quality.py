from __future__ import annotations

import re

import pandas as pd
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
    def __init__(self, history: list[dict] | None = None):
        self._history = history or []

    def read_latest(self):
        return self._history[0] if self._history else None

    def read_recent(self, limit: int):
        return self._history[:limit]


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
    r'<a\b[^>]*href="/dashboard/quality"[^>]*aria-current="page"[^>]*>\s*Quality',
    re.DOTALL,
)


def _assert_quality_chrome(response) -> None:
    assert response.status_code == 200
    assert "<title>Quality · Cicerone dashboard</title>" in response.text
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
            "generated_at": "2026-09-04T12:00:00+00:00",
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
            },
        }
    )
    app = create_app(settings, _FakeReader(), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard/quality", auth=("alice", "s3cret"))
    _assert_quality_chrome(response)
    assert "Impressions" in response.text
    assert ">10<" in response.text
    assert "By rank" in response.text
    assert "20.00%" in response.text
    assert "CTR and conversion by rank" in response.text
    assert ">Clicks<" in response.text
    assert "2026-09-04T12:00:00+00:00" in response.text
    assert "As of" in response.text


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
    assert "Impressions" in response.text
    assert ">1<" in response.text
    assert "Live from the track store." in response.text


def test_quality_live_label_when_stored_eval_lacks_track_eval(tmp_path):
    from cicerone.dashboard_quality import quality_context
    from cicerone.track.normalize import normalize_track

    settings = _settings(tmp_path, track={"enabled": True})
    store = TrackStore(settings.output)
    store.write_eval(
        {
            "generated_at": "2026-09-04T12:00:00+00:00",
            "served_eval": {
                "generated_at": "2026-09-01T00:00:00+00:00",
                "metrics": {"HitRate@10": 0.1},
            },
        }
    )
    store.append_rows(
        [
            normalize_track(
                {
                    "kind": "impression",
                    "user_id": "u1",
                    "item_id": "i1",
                    "rank": 1,
                    "occurred_at": "2026-08-28T12:00:00Z",
                    "event_id": "imp-live-fallback",
                }
            ).as_row()
        ]
    )
    context = quality_context(settings)
    assert context["track_live"] is True
    assert context["track_as_of"] is None
    assert context["track_eval"]["overall"]["n_impressions"] == 1
    app = create_app(settings, _FakeReader(), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard/quality", auth=("alice", "s3cret"))
    _assert_quality_chrome(response)
    assert "Live from the track store." in response.text
    assert "As of" not in response.text
    assert "2026-09-04T12:00:00+00:00" not in response.text
    assert "Lists from" in response.text


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


def test_quality_live_eval_pushes_conversion_event_types(tmp_path, monkeypatch):
    from cicerone.dashboard_quality import quality_context
    from cicerone.evaluation import DEFAULT_CONVERSION_TYPE
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
                    "event_id": "imp-types",
                }
            ).as_row()
        ]
    )
    seen: dict[str, object] = {}

    def _load_events(_settings, *, event_types=None, since=None):
        del since
        seen["event_types"] = event_types
        return pd.DataFrame(columns=["user_id", "item_id", "event_type", "quantity", "occurred_at"])

    monkeypatch.setattr("cicerone.dashboard_quality.load_metric_events", _load_events)
    quality_context(settings)
    assert seen["event_types"] == (DEFAULT_CONVERSION_TYPE,)


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
        "cicerone.dashboard_quality.load_metric_events",
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
    assert ">HitRate@10<" in response.text
    assert "name=0.5000" not in response.text


def test_quality_as_of_falls_back_to_served_eval(tmp_path):
    from cicerone.dashboard_quality import quality_context

    settings = _settings(tmp_path, track={"enabled": True})
    TrackStore(settings.output).write_eval(
        {
            "track_eval": {
                "overall": {
                    "n_impressions": 2,
                    "n_clicks": 1,
                    "n_conversions_click": 0,
                    "n_conversions_view": 0,
                    "ctr": 0.5,
                    "cvr_click": 0.0,
                    "cvr_view": 0.0,
                    "n_users": 1,
                }
            },
            "served_eval": {"generated_at": "2026-09-01T00:00:00+00:00", "metrics": {"HitRate@10": 0.1}},
        }
    )
    context = quality_context(settings)
    assert context["track_as_of"] == "2026-09-01T00:00:00+00:00"
    assert context["replay_metric_names"] == []
    assert context["ranking_metrics"] == {"HitRate@10": 0.1}
    assert context["catalog_metrics"] == {}


def test_replay_metric_names_are_source_keys_only():
    from cicerone.dashboard_quality import _replay_metric_names

    assert _replay_metric_names(None) == []
    assert _replay_metric_names({"metrics": {"NDCG@10": 0.2}, "by_source": {"a": {"Recall@10": 0.1}}}) == [
        "Recall@10",
    ]
    assert _replay_metric_names({"by_source": {"a": "bad"}}) == []


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


def test_quality_context_clears_error_when_live_track_succeeds(tmp_path, monkeypatch):
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

    def _boom(self):
        raise RuntimeError("nope")

    monkeypatch.setattr("cicerone.track.store.TrackStore.read_eval", _boom)
    context = quality_context(settings)
    assert context["error"] is None
    assert context["track_live"] is True
    assert context["empty_track"] is False


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


def test_quality_live_eval_history_error_keeps_current_recs(tmp_path, monkeypatch):
    import pandas as pd

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
                    "generated_at": "2026-08-28T03:00:00+00:00",
                    "event_id": "imp-keep-recs",
                }
            ).as_row()
        ]
    )
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}]
    ).to_parquet(tmp_path / "recommendations.parquet", index=False)
    monkeypatch.setattr(
        "cicerone.track.store.TrackStore.read_history",
        lambda self, **_kwargs: (_ for _ in ()).throw(RuntimeError("history")),
    )
    context = quality_context(settings)
    assert context["empty_track"] is False
    assert context["track_eval"]["by_source"]["popular"]["n_impressions"] == 1


def test_quality_context_no_impressions_malformed_overall(tmp_path):
    from cicerone.dashboard_quality import quality_context

    settings = _settings(tmp_path, track={"enabled": True})
    TrackStore(settings.output).write_eval({"track_eval": {"overall": "nope"}})
    context = quality_context(settings)
    assert context["empty_track"] is True


def test_quality_page_splits_catalog_metrics(tmp_path):
    settings = _settings(tmp_path, track={"enabled": True}, eval={"enabled": True})
    TrackStore(settings.output).write_eval(
        {
            "served_eval": {
                "n_users": 3,
                "n_users_with_events": 1,
                "generated_at": "2026-08-28T03:00:00+00:00",
                "metrics": {
                    "HitRate@10": 0.5,
                    "NDCG@10": 0.4,
                    "CatalogCoverage@10": 0.2,
                    "MeanInvUserFreq@10": 3.1,
                    "AvgRecPopularity@10": 12.0,
                },
                "by_source": {"personalized": {"HitRate@10": 0.5}},
            }
        }
    )
    app = create_app(settings, _FakeReader(), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard/quality", auth=("alice", "s3cret"))
    assert response.status_code == 200
    assert "Catalog" in response.text
    assert "Production replay catalog metrics" in response.text
    assert "Coverage is the share of the catalog in lists." in response.text
    assert "CatalogCoverage@10" in response.text
    html = response.text
    source_caption = html.split("Production replay hit rate by source", 1)[1]
    assert "HitRate@10" in source_caption
    assert "NDCG@10" not in source_caption.split("</tr>", 1)[0]


def test_quality_page_shows_previous_run_delta(tmp_path):
    from cicerone.dashboard_quality import quality_context

    settings = _settings(tmp_path, track={"enabled": True}, eval={"enabled": True})
    current = {
        "overall": {
            "n_impressions": 200,
            "n_clicks": 40,
            "n_conversions_click": 10,
            "n_conversions_view": 12,
            "ctr": 0.2,
            "cvr_click": 0.05,
            "cvr_view": 0.06,
            "n_users": 20,
        }
    }
    previous = {
        "overall": {
            "n_impressions": 200,
            "n_clicks": 20,
            "n_conversions_click": 4,
            "n_conversions_view": 4,
            "ctr": 0.1,
            "cvr_click": 0.02,
            "cvr_view": 0.02,
            "n_users": 20,
        }
    }
    TrackStore(settings.output).write_eval(
        {
            "generated_at": "2026-09-07T12:00:00+00:00",
            "track_eval": current,
            "served_eval": {
                "metrics": {"NDCG@10": 0.4, "CatalogCoverage@10": 0.25},
                "by_source": {},
            },
        }
    )
    history = [
        {
            "status": "success",
            "triggered_by": "cron",
            "generated_at": "2026-09-08T12:00:00+00:00",
            "track_eval": current,
            "served_eval": {"metrics": {"NDCG@10": 0.4, "CatalogCoverage@10": 0.25}},
        },
        {
            "status": "success",
            "triggered_by": "cron",
            "generated_at": "2026-09-07T12:00:00+00:00",
            "track_eval": previous,
            "served_eval": {"metrics": {"NDCG@10": 0.3, "CatalogCoverage@10": 0.2}},
        },
        {
            "status": "success",
            "triggered_by": "incremental",
            "generated_at": "2026-09-07T18:00:00+00:00",
            "track_eval": previous,
        },
    ]
    context = quality_context(settings, _FakeReader(history))
    assert context["quality_deltas"]["ctr"] == "+10.00 pp"
    assert context["quality_deltas"]["cvr_click"] == "+3.00 pp"
    assert context["quality_deltas"]["ndcg"] == "+0.1000"
    assert context["quality_deltas"]["coverage"] == "+0.0500"
    assert len(context["quality_history"]) == 2
    app = create_app(settings, _FakeReader(history), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard/quality", auth=("alice", "s3cret"))
    assert "+10.00 pp" in response.text
    assert "Recent quality" in response.text
    assert "2026-09-07T12:00:00+00:00" in response.text
    assert "Only the latest run is available" not in response.text


def test_quality_deltas_live_track_skips_latest_replay_row():
    from cicerone.dashboard_quality import _quality_deltas

    latest = {
        "ctr": 0.1,
        "cvr_click": 0.02,
        "served_eval": {"metrics": {"NDCG@10": 0.4, "CatalogCoverage@10": 0.25}},
    }
    previous = {
        "ctr": 0.05,
        "cvr_click": 0.01,
        "served_eval": {"metrics": {"NDCG@10": 0.3, "CatalogCoverage@10": 0.2}},
    }
    live_track = {"overall": {"ctr": 0.2, "cvr_click": 0.05}}
    stored_served = {"metrics": {"NDCG@10": 0.4, "CatalogCoverage@10": 0.25}}
    deltas = _quality_deltas(
        live_track,
        stored_served,
        [latest, previous],
        skip_first_track=False,
        skip_first_replay=True,
    )
    assert deltas["ctr"] == "+10.00 pp"
    assert deltas["cvr_click"] == "+3.00 pp"
    assert deltas["ndcg"] == "+0.1000"
    assert deltas["coverage"] == "+0.0500"
    single = _quality_deltas(
        live_track,
        stored_served,
        [latest],
        skip_first_track=False,
        skip_first_replay=True,
    )
    assert single["ctr"] == "+10.00 pp"
    assert single["ndcg"] == "—"
    assert single["coverage"] == "—"


def test_quality_deltas_skip_newer_manifest_when_report_lags():
    from cicerone.dashboard_quality import _quality_deltas

    current_served = {
        "generated_at": "2026-09-07T12:00:00+00:00",
        "metrics": {"NDCG@10": 0.4, "CatalogCoverage@10": 0.25},
    }
    newer = {
        "generated_at": "2026-09-09T12:00:00+00:00",
        "ctr": 0.3,
        "cvr_click": 0.06,
        "served_eval": {
            "generated_at": "2026-09-08T12:00:00+00:00",
            "metrics": {"NDCG@10": 0.5, "CatalogCoverage@10": 0.3},
        },
    }
    matching = {
        "generated_at": "2026-09-08T12:00:00+00:00",
        "ctr": 0.2,
        "cvr_click": 0.05,
        "served_eval": current_served,
    }
    previous = {
        "generated_at": "2026-09-07T12:00:00+00:00",
        "ctr": 0.1,
        "cvr_click": 0.02,
        "served_eval": {
            "generated_at": "2026-09-06T12:00:00+00:00",
            "metrics": {"NDCG@10": 0.3, "CatalogCoverage@10": 0.2},
        },
    }
    deltas = _quality_deltas(
        {"overall": {"ctr": 0.2, "cvr_click": 0.05}},
        current_served,
        [newer, matching, previous],
        skip_first_track=True,
        skip_first_replay=True,
        current_stamp="2026-09-07T12:00:00+00:00",
    )
    assert deltas["ndcg"] == "+0.1000"
    assert deltas["coverage"] == "+0.0500"
    assert deltas["ctr"] == "+10.00 pp"
    assert deltas["cvr_click"] == "+3.00 pp"


def test_quality_context_stale_report_skips_newer_manifest(tmp_path):
    from cicerone.dashboard_quality import quality_context

    settings = _settings(tmp_path, track={"enabled": True}, eval={"enabled": True})
    current_served = {
        "generated_at": "2026-09-07T12:00:00+00:00",
        "metrics": {"NDCG@10": 0.4, "CatalogCoverage@10": 0.25},
        "by_source": {},
    }
    TrackStore(settings.output).write_eval(
        {
            "generated_at": "2026-09-07T12:00:00+00:00",
            "track_eval": {
                "overall": {
                    "n_impressions": 200,
                    "n_clicks": 40,
                    "n_conversions_click": 10,
                    "n_conversions_view": 0,
                    "ctr": 0.2,
                    "cvr_click": 0.05,
                    "cvr_view": 0.0,
                    "n_users": 20,
                }
            },
            "served_eval": current_served,
        }
    )
    history = [
        {
            "status": "success",
            "triggered_by": "cron",
            "generated_at": "2026-09-09T12:00:00+00:00",
            "track_eval": {"overall": {"ctr": 0.3, "cvr_click": 0.06}},
            "served_eval": {
                "generated_at": "2026-09-08T12:00:00+00:00",
                "metrics": {"NDCG@10": 0.5, "CatalogCoverage@10": 0.3},
            },
        },
        {
            "status": "success",
            "triggered_by": "cron",
            "generated_at": "2026-09-08T12:00:00+00:00",
            "track_eval": {"overall": {"ctr": 0.2, "cvr_click": 0.05}},
            "served_eval": current_served,
        },
        {
            "status": "success",
            "triggered_by": "cron",
            "generated_at": "2026-09-07T12:00:00+00:00",
            "track_eval": {"overall": {"ctr": 0.1, "cvr_click": 0.02}},
            "served_eval": {
                "generated_at": "2026-09-06T12:00:00+00:00",
                "metrics": {"NDCG@10": 0.3, "CatalogCoverage@10": 0.2},
            },
        },
    ]
    context = quality_context(settings, _FakeReader(history))
    assert context["quality_deltas"]["ndcg"] == "+0.1000"
    assert context["quality_deltas"]["coverage"] == "+0.0500"
    assert context["quality_deltas"]["ctr"] == "+10.00 pp"


def test_quality_live_track_replay_delta_skips_current_manifest(tmp_path):
    from cicerone.dashboard_quality import quality_context
    from cicerone.track.normalize import normalize_track

    settings = _settings(tmp_path, track={"enabled": True}, eval={"enabled": True})
    stored_served = {"metrics": {"NDCG@10": 0.4, "CatalogCoverage@10": 0.25}, "by_source": {}}
    store = TrackStore(settings.output)
    store.write_eval({"served_eval": stored_served})
    store.append_rows(
        [
            normalize_track(
                {
                    "kind": "impression",
                    "user_id": "u1",
                    "item_id": "i1",
                    "rank": 1,
                    "occurred_at": "2026-08-28T12:00:00Z",
                    "event_id": "imp-live-delta",
                }
            ).as_row()
        ]
    )
    history = [
        {
            "status": "success",
            "triggered_by": "cron",
            "generated_at": "2026-09-08T12:00:00+00:00",
            "track_eval": {"overall": {"ctr": 0.2, "cvr_click": 0.05}},
            "served_eval": stored_served,
        },
        {
            "status": "success",
            "triggered_by": "cron",
            "generated_at": "2026-09-07T12:00:00+00:00",
            "track_eval": {"overall": {"ctr": 0.1, "cvr_click": 0.02}},
            "served_eval": {"metrics": {"NDCG@10": 0.3, "CatalogCoverage@10": 0.2}},
        },
    ]
    context = quality_context(settings, _FakeReader(history))
    assert context["track_live"] is True
    assert context["quality_deltas"]["ndcg"] == "+0.1000"
    assert context["quality_deltas"]["coverage"] == "+0.0500"
    assert context["quality_deltas"]["ndcg"] != "+0.0000"


def test_quality_catalog_delta_requires_matching_cutoff():
    from cicerone.dashboard_quality import _quality_deltas

    deltas = _quality_deltas(
        None,
        {"metrics": {"NDCG@10": 0.4, "CatalogCoverage@10": 0.3, "MeanInvUserFreq@10": 2.0}},
        [
            {"served_eval": {"metrics": {"NDCG@10": 0.4, "CatalogCoverage@10": 0.3}}},
            {"served_eval": {"metrics": {"NDCG@5": 0.2, "CatalogCoverage@5": 0.1, "MeanInvUserFreq@5": 1.0}}},
        ],
        skip_first_track=True,
        skip_first_replay=True,
    )
    assert deltas["ndcg"] == "—"
    assert deltas["coverage"] == "—"
    assert deltas["miuf"] == "—"
    assert deltas["coverage_name"] == "CatalogCoverage@10"
    assert deltas["ndcg_name"] == "NDCG@10"


def test_quality_page_catalog_delta_only_on_selected_cutoff(tmp_path):
    settings = _settings(tmp_path, track={"enabled": True}, eval={"enabled": True})
    TrackStore(settings.output).write_eval(
        {
            "track_eval": {
                "overall": {
                    "n_impressions": 10,
                    "n_clicks": 2,
                    "n_conversions_click": 0,
                    "n_conversions_view": 0,
                    "ctr": 0.2,
                    "cvr_click": 0.0,
                    "cvr_view": 0.0,
                    "n_users": 2,
                }
            },
            "served_eval": {
                "n_users": 3,
                "n_users_with_events": 1,
                "metrics": {
                    "NDCG@5": 0.2,
                    "NDCG@10": 0.4,
                    "CatalogCoverage@5": 0.15,
                    "CatalogCoverage@10": 0.25,
                    "MeanInvUserFreq@5": 1.5,
                    "MeanInvUserFreq@10": 2.0,
                    "AvgRecPopularity@5": 8.0,
                    "AvgRecPopularity@10": 10.0,
                },
                "by_source": {},
            },
        }
    )
    history = [
        {
            "status": "success",
            "triggered_by": "cron",
            "generated_at": "2026-09-08T12:00:00+00:00",
            "served_eval": {
                "metrics": {
                    "NDCG@10": 0.4,
                    "CatalogCoverage@10": 0.25,
                    "MeanInvUserFreq@10": 2.0,
                    "AvgRecPopularity@10": 10.0,
                }
            },
        },
        {
            "status": "success",
            "triggered_by": "cron",
            "generated_at": "2026-09-07T12:00:00+00:00",
            "served_eval": {
                "metrics": {
                    "NDCG@10": 0.3,
                    "CatalogCoverage@10": 0.2,
                    "MeanInvUserFreq@10": 1.5,
                    "AvgRecPopularity@10": 9.0,
                }
            },
        },
    ]
    app = create_app(settings, _FakeReader(history), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard/quality", auth=("alice", "s3cret"))
    catalog = response.text.split("Production replay catalog metrics", 1)[1]
    coverage_five = catalog.split("CatalogCoverage@5", 1)[1].split("</tr>", 1)[0]
    coverage_ten = catalog.split("CatalogCoverage@10", 1)[1].split("</tr>", 1)[0]
    assert "+0.0500" not in coverage_five
    assert "—" in coverage_five
    assert "+0.0500" in coverage_ten
    ranking = response.text.split("Production replay ranking metrics", 1)[1]
    ndcg_five = ranking.split("NDCG@5", 1)[1].split("</tr>", 1)[0]
    ndcg_ten = ranking.split("NDCG@10", 1)[1].split("</tr>", 1)[0]
    assert "+0.1000" not in ndcg_five
    assert "+0.1000" in ndcg_ten


def test_quality_history_single_run_footnote(tmp_path):
    settings = _settings(tmp_path, track={"enabled": True})
    TrackStore(settings.output).write_eval(
        {
            "track_eval": {
                "overall": {
                    "n_impressions": 10,
                    "n_clicks": 1,
                    "n_conversions_click": 0,
                    "n_conversions_view": 0,
                    "ctr": 0.1,
                    "cvr_click": 0.0,
                    "cvr_view": 0.0,
                    "n_users": 2,
                }
            }
        }
    )
    history = [
        {
            "status": "success",
            "triggered_by": "cron",
            "generated_at": "2026-09-08T12:00:00+00:00",
            "track_eval": {"overall": {"ctr": 0.1, "cvr_click": 0.0}},
        }
    ]
    app = create_app(settings, _FakeReader(history), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard/quality", auth=("alice", "s3cret"))
    assert "Recent quality" in response.text
    assert "Only the latest run is available" in response.text


def test_quality_history_single_footnote_skips_db_output(tmp_path):
    from cicerone.dashboard_quality import quality_context

    db_path = tmp_path / "quality.db"
    url = f"sqlite+pysqlite:///{db_path}"
    settings = _settings(
        tmp_path,
        track={"enabled": True},
        output=IOSettings(kind="db", options={"database_url": url}),
        input=IOSettings(kind="db", options={"database_url": url}),
    )
    history = [
        {
            "status": "success",
            "triggered_by": "cron",
            "generated_at": "2026-09-08T12:00:00+00:00",
            "track_eval": {"overall": {"ctr": 0.1, "cvr_click": 0.0}},
        }
    ]
    context = quality_context(settings, _FakeReader(history))
    assert len(context["quality_history"]) == 1
    assert context["quality_history_single"] is False
    app = create_app(settings, _FakeReader(history), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard/quality", auth=("alice", "s3cret"))
    assert "Recent quality" in response.text
    assert "Only the latest run is available" not in response.text


def test_quality_history_single_ignores_filtered_manifests(tmp_path):
    from cicerone.dashboard_quality import quality_context

    settings = _settings(tmp_path, track={"enabled": True})
    TrackStore(settings.output).write_eval(
        {
            "track_eval": {
                "overall": {
                    "n_impressions": 10,
                    "n_clicks": 1,
                    "n_conversions_click": 0,
                    "n_conversions_view": 0,
                    "ctr": 0.1,
                    "cvr_click": 0.0,
                    "cvr_view": 0.0,
                    "n_users": 2,
                }
            }
        }
    )
    evaluable = {
        "status": "success",
        "triggered_by": "cron",
        "generated_at": "2026-09-08T12:00:00+00:00",
        "track_eval": {"overall": {"ctr": 0.1, "cvr_click": 0.0}},
    }
    incremental = {
        "status": "success",
        "triggered_by": "incremental",
        "generated_at": "2026-09-08T13:00:00+00:00",
        "track_eval": {"overall": {"ctr": 0.2, "cvr_click": 0.0}},
    }
    failed = {
        "status": "failed",
        "triggered_by": "cron",
        "generated_at": "2026-09-07T12:00:00+00:00",
        "track_eval": {"overall": {"ctr": 0.3, "cvr_click": 0.0}},
    }
    empty_eval = {
        "status": "success",
        "triggered_by": "cron",
        "generated_at": "2026-09-06T12:00:00+00:00",
    }
    context = quality_context(settings, _FakeReader([evaluable, incremental, failed, empty_eval]))
    assert context["quality_history_single"] is True
    assert len(context["quality_history"]) == 1
    app = create_app(
        settings, _FakeReader([evaluable, incremental, failed, empty_eval]), _users_with("alice", "s3cret")
    )
    response = TestClient(app).get("/dashboard/quality", auth=("alice", "s3cret"))
    assert "Recent quality" in response.text
    assert "Only the latest run is available" in response.text


def test_quality_history_single_false_when_only_filtered_manifests(tmp_path):
    from cicerone.dashboard_quality import quality_context

    settings = _settings(tmp_path, track={"enabled": True})
    TrackStore(settings.output).write_eval(
        {
            "track_eval": {
                "overall": {
                    "n_impressions": 10,
                    "n_clicks": 1,
                    "n_conversions_click": 0,
                    "n_conversions_view": 0,
                    "ctr": 0.1,
                    "cvr_click": 0.0,
                    "cvr_view": 0.0,
                    "n_users": 2,
                }
            }
        }
    )
    incremental = {
        "status": "success",
        "triggered_by": "incremental",
        "generated_at": "2026-09-08T12:00:00+00:00",
        "track_eval": {"overall": {"ctr": 0.2, "cvr_click": 0.0}},
    }
    failed = {
        "status": "failed",
        "triggered_by": "cron",
        "generated_at": "2026-09-07T12:00:00+00:00",
        "track_eval": {"overall": {"ctr": 0.3, "cvr_click": 0.0}},
    }
    for history in ([incremental], [failed]):
        context = quality_context(settings, _FakeReader(history))
        assert context["quality_history_single"] is False
        assert context["quality_history"] == []
        app = create_app(settings, _FakeReader(history), _users_with("alice", "s3cret"))
        response = TestClient(app).get("/dashboard/quality", auth=("alice", "s3cret"))
        assert "Recent quality" not in response.text
        assert "Only the latest run is available" not in response.text


def test_rank_curve_inverted_note(tmp_path):
    settings = _settings(tmp_path, track={"enabled": True, "min_impressions": 100})
    TrackStore(settings.output).write_eval(
        {
            "track_eval": {
                "overall": {
                    "n_impressions": 200,
                    "n_clicks": 30,
                    "n_conversions_click": 0,
                    "n_conversions_view": 0,
                    "ctr": 0.15,
                    "cvr_click": 0.0,
                    "cvr_view": 0.0,
                    "n_users": 10,
                },
                "by_rank": {
                    "1": {
                        "n_impressions": 100,
                        "n_clicks": 10,
                        "n_conversions_click": 0,
                        "n_conversions_view": 0,
                        "ctr": 0.1,
                        "cvr_click": 0.0,
                        "cvr_view": 0.0,
                        "n_users": 10,
                    },
                    "2": {
                        "n_impressions": 100,
                        "n_clicks": 20,
                        "n_conversions_click": 0,
                        "n_conversions_view": 0,
                        "ctr": 0.2,
                        "cvr_click": 0.0,
                        "cvr_view": 0.0,
                        "n_users": 10,
                    },
                },
            }
        }
    )
    app = create_app(settings, _FakeReader(), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard/quality", auth=("alice", "s3cret"))
    assert "Rank CTR is not weakly decreasing" in response.text


def test_rank_curve_ignores_sparse_ranks():
    from cicerone.dashboard_quality import _rank_curve_inverted

    track_eval = {
        "by_rank": {
            "1": {"n_impressions": 100, "ctr": 0.2},
            "2": {"n_impressions": 5, "ctr": 0.9},
        }
    }
    assert _rank_curve_inverted(track_eval, 100) is False
    assert _rank_curve_inverted(track_eval, 1) is True


def test_quality_history_parses_json_strings_and_skips_failures():
    from cicerone.dashboard_quality import _quality_history

    rows = _quality_history(
        [
            {
                "status": "failed",
                "triggered_by": "cron",
                "track_eval": {"overall": {"ctr": 0.9}},
            },
            {
                "status": "success",
                "triggered_by": "cron",
                "generated_at": "2026-09-08T12:00:00+00:00",
                "track_eval": '{"overall":{"ctr":0.2,"cvr_click":0.1}}',
                "served_eval": '{"metrics":{"NDCG@5":0.1,"NDCG@10":0.2,"CatalogCoverage@10":0.3}}',
            },
        ]
    )
    assert len(rows) == 1
    assert rows[0]["ctr"] == 0.2
    assert rows[0]["ndcg"] == 0.2
    assert rows[0]["ndcg_name"] == "NDCG@10"
    assert rows[0]["coverage"] == 0.3
