from __future__ import annotations

import pandas as pd
import pytest

from cicerone.blending import COLD_START_USER_ID
from cicerone.evaluation import (
    conversion_event_types,
    evaluate_served,
    evaluate_tracking,
    filter_events_to_recommended,
    replay_ks,
    user_track_outcomes,
)


def _track(*rows: dict) -> list[dict]:
    return list(rows)


def test_evaluate_tracking_ctr_and_conversion_window() -> None:
    impressions_clicks = _track(
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "ipa",
            "rank": 1,
            "occurred_at": "2026-08-28T12:00:00Z",
            "event_id": "imp-1",
            "source": "personalized",
            "variant": "control",
        },
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "stout",
            "rank": 2,
            "occurred_at": "2026-08-28T12:00:00Z",
            "event_id": "imp-2",
            "source": "popular_fallback",
            "variant": "control",
        },
        {
            "kind": "click",
            "user_id": "alice",
            "item_id": "ipa",
            "occurred_at": "2026-08-28T12:05:00Z",
            "event_id": "clk-1",
        },
        {
            "kind": "click",
            "user_id": "bob",
            "item_id": "orphan",
            "occurred_at": "2026-08-28T12:05:00Z",
            "event_id": "clk-orphan",
        },
    )
    conversions = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "ipa",
                "event_type": "purchase",
                "occurred_at": "2026-08-28T13:00:00Z",
            },
            {
                "user_id": "alice",
                "item_id": "stout",
                "event_type": "purchase",
                "occurred_at": "2026-08-30T12:00:00Z",
            },
        ]
    )
    recs = pd.DataFrame(
        [
            {"user_id": "alice", "item_id": "ipa", "source": "personalized", "rank": 1},
            {"user_id": "alice", "item_id": "stout", "source": "popular_fallback", "rank": 2},
        ]
    )
    report = evaluate_tracking(
        track_rows=impressions_clicks,
        conversions=conversions,
        recommendations=recs,
        window_hours=24.0,
    )
    assert report.overall.n_impressions == 2
    assert report.overall.n_clicks == 1
    assert report.overall.ctr == pytest.approx(0.5)
    assert report.overall.n_conversions_click == 1
    assert report.overall.n_conversions_view == 1
    assert "1" in report.by_rank
    assert report.by_rank["1"].ctr == pytest.approx(1.0)
    assert report.by_source["personalized"].n_impressions == 1
    late = evaluate_tracking(
        track_rows=impressions_clicks,
        conversions=conversions,
        window_hours=0.5,
    )
    assert late.overall.n_conversions_view == 0
    assert late.overall.n_conversions_click == 0


def test_evaluate_tracking_empty_and_helpers() -> None:
    empty = evaluate_tracking(track_rows=[], conversions=pd.DataFrame())
    assert empty.overall.n_impressions == 0
    assert conversion_event_types((), primary_metric="weighted") == ("purchase",)
    assert conversion_event_types((), primary_metric="ctr") == ("purchase",)
    assert conversion_event_types((), primary_metric="view") == ("view",)
    assert conversion_event_types(("purchase", "save"), primary_metric="ctr") == ("purchase", "save")
    assert replay_ks((), top_k=10) == (5, 10)
    assert replay_ks((3, 3, 20), top_k=10) == (3,)
    assert replay_ks((0, 99), top_k=10) == (10,)


def test_user_track_outcomes_ctr_and_conversion() -> None:
    rows = _track(
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "ipa",
            "occurred_at": "2026-08-28T12:00:00Z",
            "event_id": "imp-1",
        },
        {
            "kind": "click",
            "user_id": "alice",
            "item_id": "ipa",
            "occurred_at": "2026-08-28T12:01:00Z",
            "event_id": "clk-1",
        },
    )
    conversions = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "ipa",
                "event_type": "purchase",
                "occurred_at": "2026-08-28T12:10:00Z",
            }
        ]
    )
    ctr = user_track_outcomes(
        track_rows=rows,
        conversions=conversions,
        primary_metric="ctr",
        attribution="click",
        window_hours=24,
    )
    assert ctr["alice"] == pytest.approx(1.0)
    conv = user_track_outcomes(
        track_rows=rows,
        conversions=conversions,
        primary_metric="conversion",
        attribution="impression",
        window_hours=24,
    )
    assert conv["alice"] == pytest.approx(1.0)


def test_filter_events_to_recommended_excludes_cold_start() -> None:
    recs = pd.DataFrame(
        [
            {"user_id": "alice", "item_id": "ipa", "variant": "control"},
            {"user_id": COLD_START_USER_ID, "item_id": "stout", "variant": "control"},
            {"user_id": "bob", "item_id": "lager", "variant": "treatment"},
        ]
    )
    events = pd.DataFrame(
        [
            {"user_id": "alice", "item_id": "ipa", "event_type": "purchase"},
            {"user_id": "alice", "item_id": "other", "event_type": "purchase"},
            {"user_id": COLD_START_USER_ID, "item_id": "stout", "event_type": "purchase"},
            {"user_id": "bob", "item_id": "lager", "event_type": "purchase"},
        ]
    )
    filtered = filter_events_to_recommended(events, recs, assigned={"alice": "control", "bob": "control"})
    assert list(filtered["item_id"]) == ["ipa"]


def test_evaluate_served_hit_rate_and_history() -> None:
    recs = pd.DataFrame(
        [
            {"user_id": "alice", "item_id": "ipa", "rank": 1, "source": "personalized"},
            {"user_id": "alice", "item_id": "stout", "rank": 2, "source": "personalized"},
            {"user_id": COLD_START_USER_ID, "item_id": "lager", "rank": 1, "source": "popular_fallback"},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "ipa",
                "event_type": "purchase",
                "occurred_at": "2026-08-28T12:00:00Z",
            },
            {
                "user_id": "alice",
                "item_id": "old",
                "event_type": "purchase",
                "occurred_at": "2026-08-27T12:00:00Z",
            },
        ]
    )
    report = evaluate_served(
        recs,
        events,
        generated_at="2026-08-28T03:00:00+00:00",
        ks=(1, 2),
        event_types=("purchase",),
    )
    assert report is not None
    assert report.n_users == 1
    assert report.metrics["HitRate@1"] == pytest.approx(1.0)
    old_recs = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "miss",
                "rank": 1,
                "source": "personalized",
                "generated_at": "2026-08-20T00:00:00Z",
            }
        ]
    )
    new_recs = recs.copy()
    new_recs["generated_at"] = "2026-08-28T03:00:00+00:00"
    history = pd.concat([old_recs, new_recs], ignore_index=True)
    later_events = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "ipa",
                "event_type": "purchase",
                "occurred_at": "2026-08-28T12:00:00Z",
            }
        ]
    )
    replayed = evaluate_served(
        old_recs,
        later_events,
        generated_at="2026-08-20T00:00:00Z",
        ks=(1,),
        event_types=("purchase",),
        history=history,
    )
    assert replayed is not None
    assert replayed.metrics["HitRate@1"] == pytest.approx(1.0)


def test_evaluate_served_history_without_generated_at() -> None:
    recs = pd.DataFrame([{"user_id": "alice", "item_id": "ipa", "rank": 1, "source": "personalized"}])
    history = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "stout",
                "rank": 1,
                "source": "personalized",
                "generated_at": "2026-08-20T00:00:00Z",
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "stout",
                "event_type": "purchase",
                "occurred_at": "2026-08-21T00:00:00Z",
            }
        ]
    )
    report = evaluate_served(
        recs,
        events,
        generated_at=None,
        ks=(1,),
        event_types=("purchase",),
        history=history,
    )
    assert report is not None
    assert report.metrics["HitRate@1"] == pytest.approx(1.0)


def test_score_previous_run_fail_open(tmp_path) -> None:
    from cicerone.config import IOSettings, make_settings
    from cicerone.job import _score_previous_run

    settings = make_settings(
        track={"enabled": True},
        eval={"enabled": True},
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    track, served = _score_previous_run(settings, pd.DataFrame(), None)
    assert track is not None
    assert track["overall"]["n_impressions"] == 0
    assert served is None


def test_score_previous_run_swallows_errors(tmp_path, monkeypatch) -> None:
    from cicerone.config import IOSettings, make_settings
    from cicerone.job import _score_previous_run

    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    settings = make_settings(track={"enabled": True}, eval={"enabled": True}, output=output)
    monkeypatch.setattr(
        "cicerone.job.load_recommendations_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("recs")),
    )
    monkeypatch.setattr(
        "cicerone.job.evaluate_tracking",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("track")),
    )
    track, served = _score_previous_run(settings, pd.DataFrame(), {"generated_at": "t"})
    assert track is None
    assert served is None


def test_evaluate_served_empty_and_as_dict() -> None:
    from cicerone.evaluation import TrackEvalReport

    assert evaluate_served(pd.DataFrame(), pd.DataFrame(), generated_at=None, ks=(5,), event_types=()) is None
    only_cold = pd.DataFrame(
        [{"user_id": COLD_START_USER_ID, "item_id": "x", "rank": 1, "source": "popular_fallback"}]
    )
    assert evaluate_served(only_cold, pd.DataFrame(), generated_at=None, ks=(5,), event_types=()) is None
    recs = pd.DataFrame([{"user_id": "alice", "item_id": "ipa", "rank": 1, "source": "personalized"}])
    empty_events = evaluate_served(recs, pd.DataFrame(), generated_at="not-a-date", ks=(1,), event_types=())
    assert empty_events is not None
    assert empty_events.n_users == 1
    dumped = empty_events.as_dict()
    assert dumped["n_users"] == 1
    metrics = TrackEvalReport(overall=evaluate_tracking(track_rows=[], conversions=pd.DataFrame()).overall)
    assert metrics.as_dict()["overall"]["n_impressions"] == 0
    no_rank = evaluate_served(
        pd.DataFrame([{"user_id": "alice", "item_id": "ipa", "source": "personalized"}]),
        pd.DataFrame(
            [
                {
                    "user_id": "bob",
                    "item_id": "other",
                    "event_type": "purchase",
                    "occurred_at": "2026-08-28T12:00:00Z",
                }
            ]
        ),
        generated_at=None,
        ks=(1,),
        event_types=("purchase",),
    )
    assert no_rank is not None
    assert no_rank.metrics["HitRate@1"] == 0.0
    filtered = filter_events_to_recommended(pd.DataFrame(), recs)
    assert filtered.empty
    clicks_only = evaluate_tracking(
        track_rows=[{"kind": "click", "user_id": "a"}],
        conversions=pd.DataFrame(),
    )
    assert clicks_only.overall.n_impressions == 0


def test_score_previous_run_served_eval(tmp_path) -> None:
    from cicerone.config import IOSettings, make_settings
    from cicerone.job import _score_previous_run
    from cicerone.track.normalize import normalize_track
    from cicerone.track.store import TrackStore

    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    recs = pd.DataFrame(
        [
            {"user_id": "alice", "item_id": "ipa", "rank": 1, "score": 1.0, "source": "personalized"},
        ]
    )
    recs.to_parquet(tmp_path / "recommendations.parquet", index=False)
    store = TrackStore(output)
    store.append_rows(
        [
            normalize_track(
                {
                    "kind": "impression",
                    "user_id": "alice",
                    "item_id": "ipa",
                    "rank": 1,
                    "occurred_at": "2026-08-28T04:00:00Z",
                    "event_id": "imp-a",
                }
            ).as_row()
        ]
    )
    store.append_history(recs, generated_at="2026-08-28T03:00:00+00:00")
    settings = make_settings(track={"enabled": True}, eval={"enabled": True}, output=output)
    events = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "ipa",
                "event_type": "purchase",
                "occurred_at": "2026-08-28T12:00:00Z",
            }
        ]
    )
    track, served = _score_previous_run(settings, events, {"generated_at": "2026-08-28T03:00:00+00:00"})
    assert track is not None
    assert track["overall"]["n_impressions"] == 1
    assert served is not None
    assert served["n_users"] == 1
    assert "HitRate@5" in served["metrics"] or "HitRate@10" in served["metrics"] or served["metrics"]


def test_score_previous_run_empty_history(tmp_path) -> None:
    from cicerone.config import IOSettings, make_settings
    from cicerone.job import _score_previous_run

    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    recs = pd.DataFrame(
        [{"user_id": "alice", "item_id": "ipa", "rank": 1, "score": 1.0, "source": "personalized"}]
    )
    recs.to_parquet(tmp_path / "recommendations.parquet", index=False)
    settings = make_settings(eval={"enabled": True}, output=output)
    events = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "ipa",
                "event_type": "purchase",
                "occurred_at": "2026-08-28T12:00:00Z",
            }
        ]
    )
    track, served = _score_previous_run(settings, events, {"generated_at": "2026-08-28T03:00:00+00:00"})
    assert track is None
    assert served is not None


def test_score_previous_run_history_and_served_errors(tmp_path, monkeypatch) -> None:
    from cicerone.config import IOSettings, make_settings
    from cicerone.job import _score_previous_run

    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    recs = pd.DataFrame(
        [{"user_id": "alice", "item_id": "ipa", "rank": 1, "score": 1.0, "source": "personalized"}]
    )
    recs.to_parquet(tmp_path / "recommendations.parquet", index=False)
    settings = make_settings(track={"enabled": True}, eval={"enabled": True}, output=output)
    events = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "ipa",
                "event_type": "purchase",
                "occurred_at": "2026-08-28T12:00:00Z",
            }
        ]
    )
    monkeypatch.setattr(
        "cicerone.track.store.TrackStore.read_history",
        lambda self: (_ for _ in ()).throw(RuntimeError("history")),
    )
    track, served = _score_previous_run(settings, events, {"generated_at": "2026-08-28T03:00:00+00:00"})
    assert track is not None
    assert served is not None
    monkeypatch.setattr(
        "cicerone.job.evaluate_served",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("served")),
    )
    track, served = _score_previous_run(settings, events, {"generated_at": "2026-08-28T03:00:00+00:00"})
    assert track is not None
    assert served is None


def test_evaluation_remaining_branches(monkeypatch) -> None:
    from datetime import timedelta

    from cicerone.evaluation import _annotate_source, _merge_asof_events, _recs_from_history

    empty = _merge_asof_events(pd.DataFrame(), pd.DataFrame({"user_id": ["a"]}), window=timedelta(hours=1))
    assert empty.empty
    earlier = pd.DataFrame(
        {
            "user_id": ["alice"],
            "item_id": ["ipa"],
            "occurred_at": [pd.Timestamp("2026-08-28T12:00:00Z")],
        }
    )
    later = pd.DataFrame(
        {
            "user_id": ["alice"],
            "item_id": ["ipa"],
            "occurred_at": [pd.Timestamp("2026-08-28T13:00:00Z")],
        }
    )
    matched = _merge_asof_events(later, earlier, window=timedelta(hours=24))
    assert len(matched) == 1
    assert _annotate_source(pd.DataFrame(), None).empty
    recs = pd.DataFrame(
        [{"user_id": "alice", "item_id": "ipa", "source": "personalized", "variant": "control"}]
    )
    impressions = pd.DataFrame(
        [
            {
                "kind": "impression",
                "user_id": "alice",
                "item_id": "ipa",
                "rank": 1,
                "occurred_at": "2026-08-28T12:00:00Z",
            }
        ]
    )
    annotated = _annotate_source(impressions, recs)
    assert annotated.iloc[0]["source"] == "personalized"
    both = impressions.copy()
    both["source"] = None
    filled = _annotate_source(both, recs)
    assert filled.iloc[0]["source"] == "personalized"
    snapshots = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "ipa",
                "source": "popular_fallback",
                "variant": "control",
                "generated_at": "2026-08-20T00:00:00Z",
            },
            {
                "user_id": "alice",
                "item_id": "ipa",
                "source": "personalized",
                "variant": "treatment",
                "generated_at": "2026-08-28T00:00:00Z",
            },
        ]
    )
    later_imp = pd.DataFrame([{"user_id": "alice", "item_id": "ipa", "generated_at": "2026-08-28T00:00:00Z"}])
    by_snap = _annotate_source(later_imp, snapshots)
    assert by_snap.iloc[0]["source"] == "personalized"
    assert by_snap.iloc[0]["variant"] == "treatment"
    missing_time = evaluate_tracking(
        track_rows=[{"kind": "impression", "user_id": "a", "item_id": "i"}],
        conversions=pd.DataFrame(),
    )
    assert missing_time.overall.n_impressions == 0
    clicks_no_id = evaluate_tracking(
        track_rows=[
            {
                "kind": "impression",
                "user_id": "alice",
                "item_id": "ipa",
                "rank": 1,
                "occurred_at": "2026-08-28T12:00:00Z",
            },
            {
                "kind": "click",
                "user_id": "alice",
                "item_id": "ipa",
                "occurred_at": "2026-08-28T12:01:00Z",
            },
        ],
        conversions=pd.DataFrame(),
    )
    assert clicks_no_id.overall.n_clicks == 1
    assert (
        user_track_outcomes(
            track_rows=[],
            conversions=pd.DataFrame(),
            primary_metric="ctr",
            attribution="user",
            window_hours=24,
        )
        == {}
    )
    assert (
        user_track_outcomes(
            track_rows=[
                {"kind": "click", "user_id": "a", "item_id": "i", "occurred_at": "2026-08-28T12:00:00Z"}
            ],
            conversions=pd.DataFrame(),
            primary_metric="ctr",
            attribution="user",
            window_hours=24,
        )
        == {}
    )
    assert (
        user_track_outcomes(
            track_rows=[{"kind": "impression", "user_id": "alice", "item_id": "ipa"}],
            conversions=pd.DataFrame(),
            primary_metric="ctr",
            attribution="impression",
            window_hours=24,
        )
        == {}
    )
    assert (
        user_track_outcomes(
            track_rows=[
                {
                    "kind": "impression",
                    "user_id": "alice",
                    "item_id": "ipa",
                    "occurred_at": "2026-08-28T12:00:00Z",
                }
            ],
            conversions=pd.DataFrame([{"user_id": "alice", "item_id": "ipa", "event_type": "purchase"}]),
            primary_metric="conversion",
            attribution="impression",
            window_hours=24,
        )["alice"]
        == 0.0
    )
    outcomes = user_track_outcomes(
        track_rows=[
            {
                "kind": "impression",
                "user_id": "alice",
                "item_id": "ipa",
                "occurred_at": "2026-08-28T12:00:00Z",
            },
            {
                "kind": "click",
                "user_id": "alice",
                "item_id": "ipa",
                "occurred_at": "2026-08-28T12:01:00Z",
            },
        ],
        conversions=pd.DataFrame(),
        primary_metric="ctr",
        attribution="click",
        window_hours=24,
    )
    assert outcomes["alice"] == pytest.approx(1.0)
    recs_none_source = pd.DataFrame([{"user_id": "alice", "item_id": "ipa", "rank": 1, "source": None}])
    report = evaluate_served(
        recs_none_source,
        pd.DataFrame(
            [
                {
                    "user_id": "alice",
                    "item_id": "ipa",
                    "event_type": "purchase",
                    "occurred_at": "2026-08-28T12:00:00Z",
                }
            ]
        ),
        generated_at=None,
        ks=(1,),
        event_types=("purchase",),
    )
    assert report is not None
    hist = pd.DataFrame(
        [{"user_id": "alice", "item_id": "ipa", "rank": 1, "generated_at": "2026-08-29T00:00:00Z"}]
    )
    events = pd.DataFrame(
        [{"user_id": "alice", "item_id": "ipa", "occurred_at": pd.Timestamp("2026-08-28T00:00:00Z")}]
    )
    assert _recs_from_history(hist, events).empty
    monkeypatch.setattr(
        "cicerone.evaluation.calc_metrics", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("x"))
    )
    failed = evaluate_served(
        pd.DataFrame([{"user_id": "alice", "item_id": "ipa", "rank": 1, "source": "personalized"}]),
        pd.DataFrame(
            [
                {
                    "user_id": "alice",
                    "item_id": "ipa",
                    "event_type": "purchase",
                    "occurred_at": "2026-08-28T12:00:00Z",
                }
            ]
        ),
        generated_at=None,
        ks=(1,),
        event_types=("purchase",),
    )
    assert failed is not None
    assert "HitRate@1" in failed.metrics
    monkeypatch.setattr(
        pd,
        "merge_asof",
        lambda *args, **kwargs: pd.DataFrame({"user_id": ["alice"], "item_id": ["ipa"]}),
    )
    empty_asof = _merge_asof_events(later, earlier, window=timedelta(hours=24))
    assert empty_asof.empty
