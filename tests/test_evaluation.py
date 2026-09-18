from __future__ import annotations

import pandas as pd
import pytest

from cicerone.blending import COLD_START_USER_ID
from cicerone.evaluation import (
    DEFAULT_CONVERSION_TYPE,
    OCCURRED_AT,
    conversion_event_types,
    conversion_events,
    evaluate_served,
    evaluate_tracking,
    filter_events_by_types,
    filter_events_to_recommended,
    generated_ats_from_track,
    replay_ks,
    user_track_outcomes,
)
from cicerone.evaluation.context import (
    EVENT_METRIC_COLUMNS,
    _filter_events_since,
    concat_history,
    load_metric_events,
    prefer_history,
    stamp_recommendations,
)


def test_evaluation_package_reexports_prior_constants() -> None:
    assert OCCURRED_AT == "occurred_at"
    assert DEFAULT_CONVERSION_TYPE == "purchase"


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


def test_evaluate_tracking_slices_by_impression_event_id() -> None:
    rows = _track(
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "ipa",
            "rank": 1,
            "occurred_at": "2026-08-28T12:00:00Z",
            "event_id": "imp-rank-1",
            "source": "personalized",
            "variant": "control",
        },
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "ipa",
            "rank": 5,
            "occurred_at": "2026-08-28T12:10:00Z",
            "event_id": "imp-rank-5",
            "source": "popular_fallback",
            "variant": "treatment",
        },
        {
            "kind": "click",
            "user_id": "alice",
            "item_id": "ipa",
            "occurred_at": "2026-08-28T12:11:00Z",
            "event_id": "clk-1",
        },
    )
    conversions = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "ipa",
                "event_type": "purchase",
                "occurred_at": "2026-08-28T12:12:00Z",
            }
        ]
    )
    report = evaluate_tracking(track_rows=rows, conversions=conversions, window_hours=24.0)
    assert report.overall.n_impressions == 2
    assert report.overall.n_clicks == 1
    assert report.by_rank["1"].n_clicks == 0
    assert report.by_rank["5"].n_clicks == 1
    assert report.by_source["personalized"].n_clicks == 0
    assert report.by_source["popular_fallback"].n_clicks == 1
    assert report.by_variant["control"].n_clicks == 0
    assert report.by_variant["treatment"].n_clicks == 1
    assert report.by_rank["5"].n_conversions_click == 1
    assert report.by_rank["1"].n_conversions_click == 0
    assert report.overall.n_conversions_click == 1
    assert report.overall.cvr_click == pytest.approx(0.5)


def test_evaluate_tracking_empty_and_helpers() -> None:
    empty = evaluate_tracking(track_rows=[], conversions=pd.DataFrame())
    assert empty.overall.n_impressions == 0
    assert conversion_event_types((), primary_metric="weighted") == ("purchase",)
    assert conversion_event_types((), primary_metric="ctr") == ("purchase",)
    assert conversion_event_types((), primary_metric="view") == ("view",)
    assert conversion_event_types(("purchase", "save"), primary_metric="ctr") == ("purchase", "save")
    events = pd.DataFrame(
        [
            {"user_id": "a", "item_id": "i1", "event_type": "purchase"},
            {"user_id": "b", "item_id": "i2", "event_type": "view"},
        ]
    )
    purchases = conversion_events(events, (), primary_metric="ctr")
    assert list(purchases["event_type"]) == ["purchase"]
    assert filter_events_by_types(events, None).equals(events)
    assert generated_ats_from_track(
        [{"generated_at": "2026-09-01T00:00:00Z"}, {"generated_at": ""}],
        "2026-09-02T00:00:00Z",
        None,
    ) == {"2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z"}
    current = pd.DataFrame([{"user_id": "a", "item_id": "i1"}])
    stamped = stamp_recommendations(current, "2026-09-01T00:00:00Z")
    assert stamped is not None
    assert list(stamped["generated_at"]) == ["2026-09-01T00:00:00Z"]
    history = pd.DataFrame([{"user_id": "b", "item_id": "i2"}])
    combined = concat_history(history, stamped)
    assert combined is not None
    assert len(combined) == 2
    assert prefer_history(history, current).equals(history)
    assert prefer_history(pd.DataFrame(), current).equals(current)
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


def test_user_track_outcomes_conversion_is_rate_not_count() -> None:
    rows = _track(
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "ipa",
            "occurred_at": "2026-08-28T12:00:00Z",
            "event_id": "imp-1",
        },
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "stout",
            "occurred_at": "2026-08-28T12:00:01Z",
            "event_id": "imp-2",
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
    conv = user_track_outcomes(
        track_rows=rows,
        conversions=conversions,
        primary_metric="conversion",
        attribution="impression",
        window_hours=24,
    )
    assert conv["alice"] == pytest.approx(0.5)


def test_evaluate_tracking_caps_conversions_at_impressions() -> None:
    rows = _track(
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "ipa",
            "rank": 1,
            "occurred_at": "2026-08-28T12:00:00Z",
            "event_id": "imp-1",
        }
    )
    conversions = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "ipa",
                "event_type": "purchase",
                "occurred_at": "2026-08-28T12:10:00Z",
            },
            {
                "user_id": "alice",
                "item_id": "ipa",
                "event_type": "purchase",
                "occurred_at": "2026-08-28T12:20:00Z",
            },
        ]
    )
    report = evaluate_tracking(track_rows=rows, conversions=conversions, window_hours=24)
    assert report.overall.n_impressions == 1
    assert report.overall.n_conversions_view == 1
    assert report.overall.cvr_view == pytest.approx(1.0)


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
    assert "MAP@1" in report.metrics
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


def test_evaluate_served_catalog_coverage_uses_item_catalog() -> None:
    recs = pd.DataFrame([{"user_id": "alice", "item_id": "ipa", "rank": 1, "source": "personalized"}])
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
    items = pd.DataFrame([{"item_id": "ipa"}, {"item_id": "stout"}, {"item_id": "lager"}])
    report = evaluate_served(
        recs,
        events,
        generated_at=None,
        ks=(1,),
        event_types=("purchase",),
        catalog=items,
    )
    assert report is not None
    assert report.metrics["CatalogCoverage@1"] == pytest.approx(1.0 / 3.0)


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


def test_score_previous_run_swallows_errors(tmp_path, monkeypatch, caplog) -> None:
    from cicerone.config import IOSettings, make_settings
    from cicerone.job import _score_previous_run

    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    settings = make_settings(track={"enabled": True}, eval={"enabled": True}, output=output)
    monkeypatch.setattr(
        "cicerone.job_eval.load_recommendations_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("recs")),
    )
    monkeypatch.setattr(
        "cicerone.job_eval.evaluate_tracking",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("track")),
    )
    with caplog.at_level("ERROR", logger="cicerone.job_eval"):
        track, served = _score_previous_run(settings, pd.DataFrame(), {"generated_at": "t"})
    assert track is None
    assert served is None
    assert any(
        "Failed to load previous recommendations for eval (OSError: recs)" in record.getMessage()
        for record in caplog.records
    )


def test_score_previous_run_swallows_track_eval_errors(tmp_path, monkeypatch, caplog) -> None:
    from cicerone.config import IOSettings, make_settings
    from cicerone.job import _score_previous_run

    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    pd.DataFrame(
        [{"user_id": "alice", "item_id": "ipa", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(tmp_path / "recommendations.parquet", index=False)
    settings = make_settings(track={"enabled": True}, eval={"enabled": True}, output=output)
    monkeypatch.setattr(
        "cicerone.job_eval.evaluate_tracking",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("track")),
    )
    with caplog.at_level("ERROR", logger="cicerone.job_eval"):
        track, served = _score_previous_run(settings, pd.DataFrame(), {"generated_at": "t"})
    assert track is None
    assert any(
        "Failed to compute track eval (ValueError: track)" in record.getMessage() for record in caplog.records
    )


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


def test_score_previous_run_assignment_overlay_on_caller_thread(tmp_path, monkeypatch) -> None:
    import threading

    from cicerone.config import IOSettings, make_settings
    from cicerone.config.settings import ExperimentSettings, VariantSettings
    from cicerone.job import _score_previous_run
    from cicerone.track.normalize import normalize_track
    from cicerone.track.store import TrackStore

    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    recs = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "ipa",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                "variant": "control",
            },
            {
                "user_id": "bob",
                "item_id": "stout",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                "variant": "treatment",
            },
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
                    "variant": "control",
                    "occurred_at": "2026-08-28T04:00:00Z",
                    "event_id": "imp-a",
                }
            ).as_row()
        ]
    )
    caller = threading.get_ident()
    seen: list[int] = []

    def _overlay(self, experiment_id: str):
        del self, experiment_id
        seen.append(threading.get_ident())
        return None, None

    monkeypatch.setattr("cicerone.job.ExperimentStore.assignment_overlay", _overlay)
    settings = make_settings(
        track={"enabled": True},
        eval={"enabled": True},
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        output=output,
    )
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
    assert seen == [caller]
    assert track is not None
    assert served is not None


def test_score_previous_run_reraises_unexpected_overlay_error(tmp_path, monkeypatch) -> None:
    from cicerone.config import IOSettings, make_settings
    from cicerone.config.settings import ExperimentSettings, VariantSettings
    from cicerone.job import _score_previous_run
    from cicerone.track.normalize import normalize_track
    from cicerone.track.store import TrackStore

    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    recs = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "ipa",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                "variant": "control",
            },
            {
                "user_id": "bob",
                "item_id": "stout",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                "variant": "treatment",
            },
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
                    "variant": "control",
                    "occurred_at": "2026-08-28T04:00:00Z",
                    "event_id": "imp-a",
                }
            ).as_row()
        ]
    )

    def _boom(self):
        raise RuntimeError("state bug")

    monkeypatch.setattr("cicerone.experiment.store.ExperimentStore.read_state", _boom)
    settings = make_settings(
        track={"enabled": True},
        eval={"enabled": True},
        experiment=ExperimentSettings(
            enabled=True,
            id="ranking-cvr",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        output=output,
    )
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
    with pytest.raises(RuntimeError, match="state bug"):
        _score_previous_run(settings, events, {"generated_at": "2026-08-28T03:00:00+00:00"})


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
        lambda self, **_kwargs: (_ for _ in ()).throw(OSError("history")),
    )
    track, served = _score_previous_run(settings, events, {"generated_at": "2026-08-28T03:00:00+00:00"})
    assert track is not None
    assert served is not None
    monkeypatch.setattr(
        "cicerone.job_eval.evaluate_served",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("served")),
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
    assert annotated.iloc[0]["variant"] == "control"
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
        "cicerone.evaluation.served.calc_metrics",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("x")),
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


def test_annotate_source_latest_uses_newest_generated_at() -> None:
    from cicerone.evaluation import _annotate_source

    snapshots = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "ipa",
                "source": "personalized",
                "variant": "treatment",
                "generated_at": "2026-08-28T00:00:00Z",
            },
            {
                "user_id": "alice",
                "item_id": "ipa",
                "source": "popular_fallback",
                "variant": "control",
                "generated_at": "2026-08-20T00:00:00Z",
            },
        ]
    )
    impressions = pd.DataFrame([{"user_id": "alice", "item_id": "ipa"}])
    annotated = _annotate_source(impressions, snapshots)
    assert annotated.iloc[0]["source"] == "personalized"
    assert "variant" not in annotated.columns or pd.isna(annotated.iloc[0].get("variant"))


def test_evaluate_served_uses_assigned_variant_only() -> None:
    recs = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "control-item",
                "rank": 1,
                "source": "personalized",
                "variant": "control",
            },
            {
                "user_id": "alice",
                "item_id": "treat-item",
                "rank": 1,
                "source": "personalized",
                "variant": "treatment",
            },
        ]
    )
    events = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "treat-item",
                "event_type": "purchase",
                "occurred_at": "2026-08-28T12:00:00Z",
            }
        ]
    )
    report = evaluate_served(
        recs,
        events,
        generated_at="2026-08-28T03:00:00+00:00",
        ks=(1,),
        event_types=("purchase",),
        assigned={"alice": "control"},
    )
    assert report is not None
    assert report.metrics["HitRate@1"] == pytest.approx(0.0)


def test_evaluate_tracking_orphan_click_does_not_count_click_through() -> None:
    rows = _track(
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "ipa",
            "rank": 1,
            "occurred_at": "2026-08-28T12:00:00Z",
            "event_id": "imp-1",
        },
        {
            "kind": "click",
            "user_id": "bob",
            "item_id": "other",
            "occurred_at": "2026-08-28T12:05:00Z",
            "event_id": "clk-orphan",
        },
    )
    conversions = pd.DataFrame(
        [
            {
                "user_id": "bob",
                "item_id": "other",
                "event_type": "purchase",
                "occurred_at": "2026-08-28T12:10:00Z",
            }
        ]
    )
    report = evaluate_tracking(track_rows=rows, conversions=conversions, window_hours=24.0)
    assert report.overall.n_impressions == 1
    assert report.overall.n_clicks == 0
    assert report.overall.n_conversions_click == 0
    assert report.overall.n_conversions_view == 0
    assert report.overall.cvr_click == 0.0


def test_evaluate_tracking_overall_excludes_unattributed_click_and_conversion() -> None:
    rows = _track(
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "ipa",
            "rank": 1,
            "occurred_at": "2026-08-28T12:00:00Z",
            "event_id": "imp-rank-1",
        },
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "ipa",
            "rank": 5,
            "occurred_at": "2026-08-28T12:10:00Z",
            "event_id": "imp-rank-5",
        },
        {
            "kind": "click",
            "user_id": "alice",
            "item_id": "ipa",
            "occurred_at": "2026-08-28T12:11:00Z",
            "event_id": "clk-1",
        },
        {
            "kind": "click",
            "user_id": "bob",
            "item_id": "other",
            "occurred_at": "2026-08-28T12:11:00Z",
            "event_id": "clk-orphan",
        },
    )
    conversions = pd.DataFrame(
        [
            {
                "user_id": "bob",
                "item_id": "other",
                "event_type": "purchase",
                "occurred_at": "2026-08-28T12:12:00Z",
            },
            {
                "user_id": "alice",
                "item_id": "ipa",
                "event_type": "purchase",
                "occurred_at": "2026-08-28T12:12:00Z",
            },
        ]
    )
    report = evaluate_tracking(track_rows=rows, conversions=conversions, window_hours=24.0)
    assert report.overall.n_impressions == 2
    assert report.overall.n_clicks == 1
    assert report.overall.n_conversions_click == 1
    assert report.overall.n_conversions_view == 1
    assert report.by_rank["1"].n_clicks == 0
    assert report.by_rank["5"].n_clicks == 1
    assert report.by_rank["5"].n_conversions_click == 1


def test_annotate_source_unmatched_generated_at_does_not_take_later_snap() -> None:
    from cicerone.evaluation import _annotate_source

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
    impressions = pd.DataFrame(
        [{"user_id": "alice", "item_id": "ipa", "generated_at": "2026-08-21T00:00:00Z"}]
    )
    annotated = _annotate_source(impressions, snapshots)
    assert pd.isna(annotated.iloc[0].get("source")) or annotated.iloc[0]["source"] != "personalized"


def test_evaluate_tracking_caps_ctr_at_one() -> None:
    rows = _track(
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "ipa",
            "rank": 1,
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
        {
            "kind": "click",
            "user_id": "alice",
            "item_id": "ipa",
            "occurred_at": "2026-08-28T12:02:00Z",
            "event_id": "clk-2",
        },
    )
    report = evaluate_tracking(track_rows=rows, conversions=pd.DataFrame(), window_hours=24.0)
    assert report.overall.n_impressions == 1
    assert report.overall.n_clicks == 1
    assert report.overall.ctr == pytest.approx(1.0)
    outcomes = user_track_outcomes(
        track_rows=rows,
        conversions=pd.DataFrame(),
        primary_metric="ctr",
        attribution="click",
        window_hours=24.0,
    )
    assert outcomes["alice"] == pytest.approx(1.0)


def test_evaluate_tracking_counts_unique_clicked_impressions() -> None:
    rows = _track(
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "ipa",
            "rank": 1,
            "occurred_at": "2026-08-28T12:00:00Z",
            "event_id": "imp-1",
        },
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "stout",
            "rank": 2,
            "occurred_at": "2026-08-28T12:00:00Z",
            "event_id": "imp-2",
        },
        {
            "kind": "click",
            "user_id": "alice",
            "item_id": "ipa",
            "occurred_at": "2026-08-28T12:01:00Z",
            "event_id": "clk-1",
        },
        {
            "kind": "click",
            "user_id": "alice",
            "item_id": "ipa",
            "occurred_at": "2026-08-28T12:02:00Z",
            "event_id": "clk-2",
        },
    )
    report = evaluate_tracking(track_rows=rows, conversions=pd.DataFrame(), window_hours=24.0)
    assert report.overall.n_impressions == 2
    assert report.overall.n_clicks == 1
    assert report.overall.ctr == pytest.approx(0.5)
    outcomes = user_track_outcomes(
        track_rows=rows,
        conversions=pd.DataFrame(),
        primary_metric="ctr",
        attribution="click",
        window_hours=24.0,
    )
    assert outcomes["alice"] == pytest.approx(0.5)


def test_evaluate_tracking_blank_event_ids_count_separately() -> None:
    rows = _track(
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "ipa",
            "rank": 1,
            "occurred_at": "2026-08-28T12:00:00Z",
            "event_id": "",
        },
        {
            "kind": "impression",
            "user_id": "alice",
            "item_id": "stout",
            "rank": 2,
            "occurred_at": "2026-08-28T12:00:00Z",
            "event_id": "",
        },
        {
            "kind": "click",
            "user_id": "alice",
            "item_id": "ipa",
            "occurred_at": "2026-08-28T12:01:00Z",
            "event_id": "clk-1",
        },
        {
            "kind": "click",
            "user_id": "alice",
            "item_id": "stout",
            "occurred_at": "2026-08-28T12:02:00Z",
            "event_id": "clk-2",
        },
    )
    report = evaluate_tracking(track_rows=rows, conversions=pd.DataFrame(), window_hours=24.0)
    assert report.overall.n_impressions == 2
    assert report.overall.n_clicks == 2
    outcomes = user_track_outcomes(
        track_rows=rows,
        conversions=pd.DataFrame(),
        primary_metric="ctr",
        attribution="click",
        window_hours=24.0,
    )
    assert outcomes["alice"] == pytest.approx(1.0)


def test_annotate_source_does_not_invent_variant_from_later_snap() -> None:
    from cicerone.evaluation import _annotate_source

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
    impressions = pd.DataFrame([{"user_id": "alice", "item_id": "ipa"}])
    annotated = _annotate_source(impressions, snapshots)
    assert "variant" not in annotated.columns or pd.isna(annotated.iloc[0].get("variant"))


def test_annotate_source_untimestamped_keeps_latest_source() -> None:
    from cicerone.evaluation import _annotate_source

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
    impressions = pd.DataFrame([{"user_id": "alice", "item_id": "ipa", "generated_at": None}])
    annotated = _annotate_source(impressions, snapshots)
    assert annotated.iloc[0]["source"] == "personalized"
    assert "variant" not in annotated.columns or pd.isna(annotated.iloc[0].get("variant"))


def test_annotate_source_untimestamped_keeps_existing_source() -> None:
    from cicerone.evaluation import _annotate_source

    snapshots = pd.DataFrame(
        [
            {
                "user_id": "alice",
                "item_id": "ipa",
                "source": "personalized",
                "variant": "treatment",
                "generated_at": "2026-08-28T00:00:00Z",
            }
        ]
    )
    impressions = pd.DataFrame(
        [{"user_id": "alice", "item_id": "ipa", "generated_at": None, "source": "logged"}]
    )
    annotated = _annotate_source(impressions, snapshots)
    assert annotated.iloc[0]["source"] == "logged"
    assert "variant" not in annotated.columns or pd.isna(annotated.iloc[0].get("variant"))


def test_annotate_source_untimestamped_does_not_clear_source_without_match() -> None:
    from cicerone.evaluation import _annotate_source

    snapshots = pd.DataFrame(
        [
            {
                "user_id": "bob",
                "item_id": "stout",
                "source": "personalized",
                "generated_at": "2026-08-28T00:00:00Z",
            }
        ]
    )
    impressions = pd.DataFrame(
        [{"user_id": "alice", "item_id": "ipa", "generated_at": None, "source": "logged"}]
    )
    annotated = _annotate_source(impressions, snapshots)
    assert annotated.iloc[0]["source"] == "logged"


def test_annotate_source_without_generated_at_keeps_variant() -> None:
    from cicerone.evaluation import _annotate_source

    recs = pd.DataFrame(
        [{"user_id": "alice", "item_id": "ipa", "source": "personalized", "variant": "control"}]
    )
    impressions = pd.DataFrame([{"user_id": "alice", "item_id": "ipa"}])
    annotated = _annotate_source(impressions, recs)
    assert annotated.iloc[0]["source"] == "personalized"
    assert annotated.iloc[0]["variant"] == "control"


def test_filter_events_since_without_occurred_at_is_empty() -> None:
    frame = pd.DataFrame([{"user_id": "u1", "event_type": "purchase", "quantity": 1}])
    filtered = _filter_events_since(frame, "2026-08-28T00:00:00Z")
    assert filtered.empty


def test_load_metric_events_invalid_since_is_empty(tmp_path) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(
        input=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    frame = load_metric_events(settings, since="not-a-date")
    assert frame.empty
    assert list(frame.columns) == list(EVENT_METRIC_COLUMNS)


def test_load_metric_events_dataset_pushes_since_filter(tmp_path, monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(
        input=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    seen: dict[str, object] = {}

    def _read(_options, _filename, **kwargs):
        seen.update(kwargs)
        return pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "i1",
                    "event_type": "purchase",
                    "quantity": 1,
                    "occurred_at": "2026-08-29T06:00:00Z",
                }
            ]
        )

    monkeypatch.setattr("cicerone.evaluation.context.read_parquet", _read)
    frame = load_metric_events(settings, event_types=("purchase",), since="2026-08-29T05:00:00+00:00")
    filters = list(seen.get("filters") or [])
    assert ("event_type", "in", ["purchase"]) in filters
    since_filters = [item for item in filters if item[0] == "occurred_at" and item[1] == ">="]
    assert since_filters
    assert not isinstance(since_filters[0][2], str)
    assert len(frame) == 1


def test_load_metric_events_dataset_retries_string_since_filter(tmp_path, monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(
        input=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    calls: list[object] = []

    def _read(_options, _filename, **kwargs):
        bound = next(
            (item[2] for item in (kwargs.get("filters") or []) if item[0] == "occurred_at"),
            None,
        )
        calls.append(bound)
        if not isinstance(bound, str):
            raise TypeError("datetime bound")
        return pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "i1",
                    "event_type": "purchase",
                    "quantity": 1,
                    "occurred_at": "2026-08-29T06:00:00Z",
                }
            ]
        )

    monkeypatch.setattr("cicerone.evaluation.context.read_parquet", _read)
    frame = load_metric_events(settings, event_types=("purchase",), since="2026-08-29T05:00:00+00:00")
    assert len(calls) == 2
    assert not isinstance(calls[0], str)
    assert calls[1] == "2026-08-28"
    assert len(frame) == 1


def test_load_metric_events_dataset_filter_failure_is_empty(tmp_path, monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(
        input=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    calls = {"n": 0}

    def _read(_options, _filename, **_kwargs):
        calls["n"] += 1
        raise TypeError("unsupported occurred_at")

    monkeypatch.setattr("cicerone.evaluation.context.read_parquet", _read)
    frame = load_metric_events(settings, event_types=("purchase",), since="2026-08-29T05:00:00+00:00")
    assert calls["n"] == 4
    assert frame.empty


def test_load_metric_events_dataset_s3_missing_is_empty(tmp_path, monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(
        input=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )

    def _read(*_args, **_kwargs):
        raise RuntimeError("NoSuchKey")

    monkeypatch.setattr("cicerone.evaluation.context.read_parquet", _read)
    monkeypatch.setattr("cicerone.evaluation.context.is_s3_not_found", lambda _exc: True)
    frame = load_metric_events(settings, since="2026-08-29T05:00:00+00:00")
    assert frame.empty


def test_load_metric_events_dataset_unbounded_retries(tmp_path, monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(
        input=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    calls: list[object] = []

    def _read(_options, _filename, **kwargs):
        calls.append(kwargs.get("filters"))
        if kwargs.get("filters"):
            raise TypeError("event_type filter")
        return pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "i1",
                    "event_type": "purchase",
                    "quantity": 1,
                    "occurred_at": "2026-08-29T06:00:00Z",
                }
            ]
        )

    monkeypatch.setattr("cicerone.evaluation.context.read_parquet", _read)
    frame = load_metric_events(settings, event_types=("purchase",))
    assert calls[0] == [("event_type", "in", ["purchase"])]
    assert calls[-1] is None
    assert len(frame) == 1


def test_load_metric_events_db_pushes_since_predicate(monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(input=IOSettings(kind="db", options={"database_url": "sqlite+pysqlite://"}))
    captured: dict[str, object] = {}

    class _Engine:
        def dispose(self) -> None:
            return None

    def _read_sql(stmt, _engine, params=None):
        captured["sql"] = str(stmt)
        captured["params"] = params
        return pd.DataFrame(columns=["user_id", "item_id", "event_type", "occurred_at"])

    monkeypatch.setattr("cicerone.evaluation.context.create_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr("cicerone.evaluation.context.pd.read_sql", _read_sql)
    frame = load_metric_events(settings, event_types=("purchase",), since="2026-08-29T05:00:00+00:00")
    assert '"occurred_at" >= :since' in str(captured["sql"])
    assert '"quantity"' in str(captured["sql"])
    assert captured["params"]["since"] == "2026-08-28"
    assert captured["params"]["types"] == ["purchase"]
    assert frame.empty


def test_load_metric_events_query_pushes_since_predicate(monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(
        input=IOSettings(
            kind="db",
            options={
                "database_url": "sqlite+pysqlite://",
                "events_query": "SELECT user_id, item_id, event_type, quantity, occurred_at FROM events",
            },
        )
    )
    captured: dict[str, object] = {}

    class _Engine:
        def dispose(self) -> None:
            return None

    def _read_sql(stmt, _engine, params=None):
        captured["sql"] = str(stmt)
        captured["params"] = params
        return pd.DataFrame(columns=list(EVENT_METRIC_COLUMNS))

    monkeypatch.setattr("cicerone.evaluation.context.create_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr("cicerone.evaluation.context.pd.read_sql", _read_sql)
    frame = load_metric_events(settings, event_types=("purchase",), since="2026-08-29T05:00:00+00:00")
    assert "_cicerone_metric_events" in str(captured["sql"])
    assert '"occurred_at" >= :since' in str(captured["sql"])
    assert captured["params"]["since"] == "2026-08-28"
    assert captured["params"]["types"] == ["purchase"]
    assert frame.empty


def test_load_metric_events_query_keeps_limit(monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(
        input=IOSettings(
            kind="db",
            options={
                "database_url": "sqlite+pysqlite://",
                "events_query": (
                    "SELECT user_id, item_id, event_type, quantity, occurred_at "
                    "FROM events ORDER BY occurred_at DESC LIMIT 100"
                ),
            },
        )
    )
    captured: dict[str, object] = {}

    class _Engine:
        def dispose(self) -> None:
            return None

    def _read_sql(stmt, _engine, params=None):
        captured["sql"] = str(stmt)
        captured["params"] = params
        return pd.DataFrame(columns=list(EVENT_METRIC_COLUMNS))

    monkeypatch.setattr("cicerone.evaluation.context.create_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr("cicerone.evaluation.context.pd.read_sql", _read_sql)
    load_metric_events(settings, event_types=("purchase",), since="2026-08-29T05:00:00+00:00")
    assert "_cicerone_metric_events" not in str(captured["sql"])
    assert "LIMIT 100" in str(captured["sql"])


def test_load_metric_events_query_quoted_offset_still_wraps(monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(
        input=IOSettings(
            kind="db",
            options={
                "database_url": "sqlite+pysqlite://",
                "events_query": (
                    'SELECT user_id, item_id, event_type, quantity, occurred_at AS "offset" FROM events'
                ),
            },
        )
    )
    captured: dict[str, object] = {}

    class _Engine:
        def dispose(self) -> None:
            return None

    def _read_sql(stmt, _engine, params=None):
        captured["sql"] = str(stmt)
        return pd.DataFrame(columns=list(EVENT_METRIC_COLUMNS))

    monkeypatch.setattr("cicerone.evaluation.context.create_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr("cicerone.evaluation.context.pd.read_sql", _read_sql)
    load_metric_events(settings, since="2026-08-29T05:00:00+00:00")
    assert "_cicerone_metric_events" in str(captured["sql"])


def test_load_metric_events_query_comment_limit_still_wraps(monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(
        input=IOSettings(
            kind="db",
            options={
                "database_url": "sqlite+pysqlite://",
                "events_query": (
                    "SELECT user_id, item_id, event_type, quantity, occurred_at FROM events -- LIMIT 100"
                ),
            },
        )
    )
    captured: dict[str, object] = {}

    class _Engine:
        def dispose(self) -> None:
            return None

    def _read_sql(stmt, _engine, params=None):
        captured["sql"] = str(stmt)
        captured["params"] = params
        return pd.DataFrame(columns=list(EVENT_METRIC_COLUMNS))

    monkeypatch.setattr("cicerone.evaluation.context.create_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr("cicerone.evaluation.context.pd.read_sql", _read_sql)
    load_metric_events(settings, event_types=("purchase",), since="2026-08-29T05:00:00+00:00")
    assert "_cicerone_metric_events" in str(captured["sql"])
    assert '"occurred_at" >= :since' in str(captured["sql"])
    assert captured["params"]["since"] == "2026-08-28"


def test_load_metric_events_query_block_comment_offset_still_wraps(monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(
        input=IOSettings(
            kind="db",
            options={
                "database_url": "sqlite+pysqlite://",
                "events_query": (
                    "SELECT user_id, item_id, event_type, quantity, occurred_at FROM events /* OFFSET 1 */"
                ),
            },
        )
    )
    captured: dict[str, object] = {}

    class _Engine:
        def dispose(self) -> None:
            return None

    def _read_sql(stmt, _engine, params=None):
        captured["sql"] = str(stmt)
        return pd.DataFrame(columns=list(EVENT_METRIC_COLUMNS))

    monkeypatch.setattr("cicerone.evaluation.context.create_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr("cicerone.evaluation.context.pd.read_sql", _read_sql)
    load_metric_events(settings, since="2026-08-29T05:00:00+00:00")
    assert "_cicerone_metric_events" in str(captured["sql"])


def test_load_metric_events_query_subquery_limit_still_wraps(monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(
        input=IOSettings(
            kind="db",
            options={
                "database_url": "sqlite+pysqlite://",
                "events_query": (
                    "SELECT user_id, item_id, event_type, quantity, occurred_at "
                    "FROM (SELECT * FROM events LIMIT 1) AS ev"
                ),
            },
        )
    )
    captured: dict[str, object] = {}

    class _Engine:
        def dispose(self) -> None:
            return None

    def _read_sql(stmt, _engine, params=None):
        captured["sql"] = str(stmt)
        return pd.DataFrame(columns=list(EVENT_METRIC_COLUMNS))

    monkeypatch.setattr("cicerone.evaluation.context.create_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr("cicerone.evaluation.context.pd.read_sql", _read_sql)
    load_metric_events(settings, since="2026-08-29T05:00:00+00:00")
    assert "_cicerone_metric_events" in str(captured["sql"])


def test_load_metric_events_db_bound_failure_is_empty(monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(input=IOSettings(kind="db", options={"database_url": "sqlite+pysqlite://"}))

    class _Engine:
        def dispose(self) -> None:
            return None

    def _boom(*_args, **_kwargs):
        raise RuntimeError("sql down")

    monkeypatch.setattr("cicerone.evaluation.context.create_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr("cicerone.evaluation.context.pd.read_sql", _boom)

    def _source(_inp):
        raise AssertionError("unbounded fallback")

    monkeypatch.setattr("cicerone.evaluation.context.build_input_source", _source)
    frame = load_metric_events(settings, event_types=("purchase",), since="2026-08-29T05:00:00+00:00")
    assert frame.empty


def test_load_metric_events_db_engine_failure_is_empty(monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(input=IOSettings(kind="db", options={"database_url": "sqlite+pysqlite://"}))

    def _boom(*_args, **_kwargs):
        raise RuntimeError("engine down")

    monkeypatch.setattr("cicerone.evaluation.context.create_engine", _boom)

    def _source(_inp):
        raise AssertionError("unbounded fallback")

    monkeypatch.setattr("cicerone.evaluation.context.build_input_source", _source)
    frame = load_metric_events(settings, event_types=("purchase",), since="2026-08-29T05:00:00+00:00")
    assert frame.empty


def test_load_metric_events_dataset_keeps_quantity(tmp_path, monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(
        input=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
    )
    seen: dict[str, object] = {}

    def _read(_options, _filename, **kwargs):
        seen["columns"] = kwargs.get("columns")
        return pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "i1",
                    "event_type": "purchase",
                    "quantity": 3,
                    "occurred_at": "2026-08-29T06:00:00Z",
                }
            ]
        )

    monkeypatch.setattr("cicerone.evaluation.context.read_parquet", _read)
    frame = load_metric_events(settings, event_types=("purchase",), since="2026-08-29T05:00:00+00:00")
    assert seen.get("columns") == list(EVENT_METRIC_COLUMNS)
    assert frame.iloc[0]["quantity"] == 3


def test_load_metric_events_db_retries_without_quantity(monkeypatch) -> None:
    from conftest import make_settings
    from sqlalchemy.exc import ProgrammingError

    from cicerone.config import IOSettings

    settings = make_settings(input=IOSettings(kind="db", options={"database_url": "sqlite+pysqlite://"}))
    calls: list[str] = []

    class _Engine:
        def dispose(self) -> None:
            return None

    def _read_sql(stmt, _engine, params=None):
        sql = str(stmt)
        calls.append(sql)
        if '"quantity"' in sql:
            raise ProgrammingError("SELECT", {}, Exception('column "quantity" does not exist'))
        return pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "i1",
                    "event_type": "purchase",
                    "occurred_at": "2026-08-29T06:00:00Z",
                }
            ]
        )

    monkeypatch.setattr("cicerone.evaluation.context.create_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr("cicerone.evaluation.context.pd.read_sql", _read_sql)
    frame = load_metric_events(settings, event_types=("purchase",), since="2026-08-29T05:00:00+00:00")
    assert len(calls) == 2
    assert '"quantity"' in calls[0]
    assert '"quantity"' not in calls[1]
    assert frame.iloc[0]["quantity"] == 1


def test_load_metric_events_query_limit_defaults_quantity(monkeypatch) -> None:
    from conftest import make_settings

    from cicerone.config import IOSettings

    settings = make_settings(
        input=IOSettings(
            kind="db",
            options={
                "database_url": "sqlite+pysqlite://",
                "events_query": ("SELECT user_id, item_id, event_type, occurred_at FROM events LIMIT 100"),
            },
        )
    )

    class _Engine:
        def dispose(self) -> None:
            return None

    def _read_sql(stmt, _engine, params=None):
        return pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "i1",
                    "event_type": "purchase",
                    "occurred_at": "2026-08-29T06:00:00Z",
                }
            ]
        )

    monkeypatch.setattr("cicerone.evaluation.context.create_engine", lambda *_args, **_kwargs: _Engine())
    monkeypatch.setattr("cicerone.evaluation.context.pd.read_sql", _read_sql)
    frame = load_metric_events(settings, since="2026-08-29T05:00:00+00:00")
    assert frame.iloc[0]["quantity"] == 1
