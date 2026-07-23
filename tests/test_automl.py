from __future__ import annotations

import pandas as pd
import pytest
from rectools.metrics import calc_metrics

from cicerone import automl
from cicerone.automl import (
    Candidate,
    CandidateResult,
    _parse_candidates,
    _time_based_folds,
    evaluate_candidates,
    select_best_candidate,
)
from cicerone.dataset import build_dataset, build_interactions
from cicerone.model import train_and_recommend


def _spread_events(n_days: int) -> pd.DataFrame:
    now = pd.Timestamp.utcnow()
    rows = []
    interactions = {
        "u1": ["i1", "i2"],
        "u2": ["i2", "i3"],
        "u3": ["i1", "i3"],
    }
    # Repeat the same purchases every few days across the window so both
    # sides of a time-based split have signal regardless of where the cut falls.
    for day_offset in range(0, n_days, 3):
        occurred_at = now - pd.Timedelta(days=day_offset)
        for user, items in interactions.items():
            for item in items:
                rows.append(
                    {
                        "user_id": user,
                        "item_id": item,
                        "event_type": "purchase",
                        "quantity": 1,
                        "occurred_at": occurred_at,
                    }
                )
    return pd.DataFrame(rows)


def test_time_based_folds_splits_oldest_test_window_first():
    events = _spread_events(n_days=30)
    folds = _time_based_folds(events, n_splits=2, test_days=7)

    assert len(folds) == 2
    for train_events, test_events in folds:
        assert not train_events.empty
        assert not test_events.empty
    assert folds[0][1]["occurred_at"].max() < folds[1][1]["occurred_at"].min()


def test_time_based_folds_skips_folds_without_enough_history():
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    )
    assert _time_based_folds(events, n_splits=5, test_days=14) == []


def test_time_based_folds_includes_most_recent_event_in_latest_fold():
    # Regression test: the most recent event's timestamp must not fall
    # exactly on the strict "< test_end" upper bound of the newest fold's
    # test window, or it would never be included in any fold's test set.
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": now - pd.Timedelta(days=10),
            },
            {
                "user_id": "u1",
                "item_id": "i2",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": now,
            },
        ]
    )

    folds = _time_based_folds(events, n_splits=1, test_days=7)

    assert len(folds) == 1
    _, test_events = folds[0]
    assert pd.to_datetime(now, utc=True) in set(pd.to_datetime(test_events["occurred_at"], utc=True))


def test_parse_candidates_rejects_unknown_model():
    with pytest.raises(ValueError, match="not_a_model"):
        _parse_candidates([{"models": ["not_a_model"]}])


def test_parse_candidates_rejects_empty_list():
    with pytest.raises(ValueError, match="empty list"):
        _parse_candidates([])


def test_parse_candidates_rejects_scalar_string_models():
    with pytest.raises(ValueError, match="must be a list of model names"):
        _parse_candidates([{"models": "popular"}])


def test_parse_candidates_rejects_non_string_model_entries():
    with pytest.raises(ValueError, match="must contain only strings"):
        _parse_candidates([{"models": ["popular", 1]}])


def test_parse_candidates_rejects_unknown_weight_key():
    with pytest.raises(ValueError, match="not_enabled"):
        _parse_candidates([{"models": ["popular"], "weights": {"not_enabled": 1.0}}])


def test_parse_candidates_rejects_non_mapping_weights():
    with pytest.raises(ValueError, match="must be a table of model name -> weight"):
        _parse_candidates([{"models": ["popular"], "weights": ["popular", 1.0]}])


def test_parse_candidates_rejects_partial_weights():
    with pytest.raises(ValueError, match="latest"):
        _parse_candidates([{"models": ["popular", "latest"], "weights": {"popular": 1.0}}])


def test_parse_candidates_accepts_weights_covering_every_model():
    entry = {"models": ["popular", "latest"], "weights": {"popular": 1.0, "latest": 0.5}}
    parsed = _parse_candidates([entry])
    assert parsed[0].weights == {"popular": 1.0, "latest": 0.5}


def test_parse_candidates_rejects_empty_models():
    with pytest.raises(ValueError, match="must not be empty"):
        _parse_candidates([{"models": []}])


def test_parse_candidates_rejects_negative_weight():
    entry = {"models": ["popular", "latest"], "weights": {"popular": -1.0, "latest": 0.5}}
    with pytest.raises(ValueError, match="non-negative"):
        _parse_candidates([entry])


def test_parse_candidates_rejects_non_positive_rrf_k():
    entry = {"models": ["popular"], "weights": {"popular": 1.0}, "rrf_k": 0}
    with pytest.raises(ValueError, match="rrf_k must be positive"):
        _parse_candidates([entry])


def test_parse_candidates_defaults_to_default_candidates():
    assert _parse_candidates(None) == _parse_candidates(automl.DEFAULT_CANDIDATES)


def test_candidate_label_for_priority_and_fusion():
    assert Candidate(models=["collaborative", "popular"]).label == "collaborative+popular"
    fusion = Candidate(models=["popular", "latest"], weights={"popular": 1.0, "latest": 0.5})
    assert fusion.label == "fusion(popular=1.0,latest=0.5)"


def test_evaluate_candidates_raises_without_enough_history(sample_items, feature_config):
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now}]
    )
    with pytest.raises(ValueError, match="Not enough event history"):
        evaluate_candidates(
            events, None, sample_items, feature_config, top_k=2, half_life_days=90, n_splits=3, test_days=14
        )


def test_evaluate_candidates_raises_on_empty_candidates_list(sample_items, feature_config):
    events = _spread_events(n_days=21)
    with pytest.raises(ValueError, match="empty list"):
        evaluate_candidates(
            events, None, sample_items, feature_config, top_k=2, half_life_days=90, candidates=[]
        )


def test_evaluate_candidates_warns_when_fewer_folds_than_requested(sample_items, feature_config, caplog):
    # 21 days of history only supports 1 fold of test_days=7 (needs history
    # before the test window too), but n_splits=3 is requested -- the run
    # should still succeed with the folds it can build, while logging that
    # backtest coverage is reduced instead of silently under-delivering.
    events = _spread_events(n_days=21)
    with caplog.at_level("WARNING", logger="cicerone.automl"):
        results = evaluate_candidates(
            events,
            None,
            sample_items,
            feature_config,
            top_k=2,
            half_life_days=90,
            candidates=[{"models": ["popular"]}],
            n_splits=3,
            test_days=7,
        )

    assert results[0].n_folds < 3
    assert any("only" in record.message and "fold" in record.message for record in caplog.records)


def test_evaluate_candidates_scores_each_candidate(sample_items, feature_config):
    events = _spread_events(n_days=21)
    candidates = [{"models": ["popular"]}, {"models": ["latest"]}]

    results = evaluate_candidates(
        events,
        None,
        sample_items,
        feature_config,
        top_k=2,
        half_life_days=90,
        candidates=candidates,
        n_splits=1,
        test_days=7,
    )

    assert len(results) == 2
    for result in results:
        assert result.n_folds >= 1
        assert any(key.startswith("MAP") for key in result.metrics)
        assert any(key.startswith("NDCG") for key in result.metrics)
        assert any(key.startswith("Recall") for key in result.metrics)


def test_evaluate_candidates_handles_weighted_rrf_and_averages_across_folds(sample_items, feature_config):
    # 35 days gives 3 non-overlapping 7-day test folds plus enough leftover
    # history for each fold's train side, so multi-fold averaging is
    # actually exercised (not just a single fold repeated).
    events = _spread_events(n_days=35)
    candidate_cfg = {
        "models": ["popular", "latest"],
        "weights": {"popular": 1.0, "latest": 0.5},
        "rrf_k": 30,
    }

    results = evaluate_candidates(
        events,
        None,
        sample_items,
        feature_config,
        top_k=2,
        half_life_days=90,
        candidates=[candidate_cfg],
        n_splits=3,
        test_days=7,
    )

    assert len(results) == 1
    result = results[0]
    assert result.candidate.models == ["popular", "latest"]
    assert result.candidate.weights == {"popular": 1.0, "latest": 0.5}
    assert result.candidate.rrf_k == 30.0
    assert result.n_folds > 1

    # Independently recompute each fold's own metrics and confirm
    # evaluate_candidates' averaged result is actually their mean -- i.e.
    # multi-fold averaging isn't just echoing a single fold's numbers.
    folds = _time_based_folds(events, n_splits=3, test_days=7)
    assert len(folds) == result.n_folds
    metrics_defs = automl._make_metrics(top_k=2)
    per_fold_metrics = []
    for train_events, test_events in folds:
        built = build_dataset(train_events, None, sample_items, feature_config, half_life_days=90)
        test_interactions = build_interactions(test_events, feature_config, half_life_days=90)
        test_users = sorted(set(test_events["user_id"]))
        reco = train_and_recommend(
            built,
            test_users,
            feature_config,
            top_k=2,
            enabled_models=candidate_cfg["models"],
            weights=candidate_cfg["weights"],
            rrf_k=candidate_cfg["rrf_k"],
        )
        per_fold_metrics.append(calc_metrics(metrics_defs, reco=reco, interactions=test_interactions))

    expected = pd.DataFrame(per_fold_metrics).mean()
    for key, expected_value in expected.items():
        assert result.metrics[key] == pytest.approx(expected_value)


def test_select_best_candidate_picks_highest_primary_metric():
    low = CandidateResult(candidate=Candidate(models=["popular"]), metrics={"MAP@5": 0.1}, n_folds=1)
    high = CandidateResult(candidate=Candidate(models=["latest"]), metrics={"MAP@5": 0.9}, n_folds=1)

    assert select_best_candidate([low, high], primary_metric="MAP") is high


def test_select_best_candidate_raises_on_empty_results():
    with pytest.raises(ValueError, match="No candidate results"):
        select_best_candidate([])


def test_select_best_candidate_raises_on_unknown_primary_metric():
    result = CandidateResult(candidate=Candidate(models=["popular"]), metrics={"MAP@5": 1.0}, n_folds=1)
    with pytest.raises(ValueError, match="No metric starting with"):
        select_best_candidate([result], primary_metric="NDCG")


def test_select_best_candidate_validates_metric_key_per_result():
    # Second result is missing "MAP@5" entirely (heterogeneous metrics) -- must
    # be caught even though the first result does have a "MAP@5" key, i.e. the
    # metric key can't just be derived once from results[0] and reused.
    has_map = CandidateResult(candidate=Candidate(models=["popular"]), metrics={"MAP@5": 0.5}, n_folds=1)
    missing_map = CandidateResult(candidate=Candidate(models=["latest"]), metrics={"NDCG@5": 0.9}, n_folds=1)
    with pytest.raises(ValueError, match="latest"):
        select_best_candidate([has_map, missing_map], primary_metric="MAP")
