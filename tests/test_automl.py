from __future__ import annotations

import pandas as pd
import pytest

from cicerone import automl
from cicerone.automl import (
    Candidate,
    CandidateResult,
    _parse_candidates,
    _time_based_folds,
    evaluate_candidates,
    select_best_candidate,
)


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
