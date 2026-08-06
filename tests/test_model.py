from __future__ import annotations

import pandas as pd
import pytest
from implicit.nearest_neighbours import TFIDFRecommender
from rectools import Columns

from cicerone.config import STRATEGY_NAMES, validate_model_weights
from cicerone.dataset import build_dataset
from cicerone.model import (
    DEFAULT_MODELS,
    STRATEGIES,
    RecommenderModel,
    Strategy,
    _as_recommender_model,
    _combine_by_priority,
    _validate_strategy_names,
    train_and_recommend,
)
from cicerone.policy import allowed_items_for_cohort, resolve_eligibility


def _synthetic_events() -> pd.DataFrame:
    now = pd.Timestamp.utcnow()
    rows = []
    # Enough interactions for LightFM to fit.
    interactions = {
        "u1": ["i1", "i2"],
        "u2": ["i2", "i3"],
        "u3": ["i1", "i3"],
    }
    for user, items in interactions.items():
        for item in items:
            rows.append(
                {
                    "user_id": user,
                    "item_id": item,
                    "event_type": "purchase",
                    "quantity": 1,
                    "occurred_at": now,
                }
            )
    return pd.DataFrame(rows)


def test_availability_filters_via_policy_allowlist(sample_items, feature_config):
    rules = resolve_eligibility(feature_config)
    allowed = allowed_items_for_cohort(["u1"], None, sample_items, rules, ["i1", "i2", "i3", "i4"])
    assert allowed == ["i1", "i2"]


def test_combine_by_priority_fills_from_earlier_strategies_first():
    frames = [
        pd.DataFrame(
            [
                {"user_id": "u1", "item_id": "a", "rank": 1, "score": 1.0, "source": "first", "_weight": 1.0},
                {"user_id": "u1", "item_id": "b", "rank": 2, "score": 0.5, "source": "first", "_weight": 1.0},
            ]
        ),
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "c",
                    "rank": 1,
                    "score": 9.0,
                    "source": "second",
                    "_weight": 1.0,
                },
                {
                    "user_id": "u1",
                    "item_id": "b",
                    "rank": 2,
                    "score": 8.0,
                    "source": "second",
                    "_weight": 1.0,
                },
            ]
        ),
    ]
    out = _combine_by_priority(frames, top_k=2)
    assert list(out[Columns.Item]) == ["a", "b"]
    assert list(out[Columns.Rank]) == [1, 2]
    assert list(out["source"]) == ["first", "first"]


def test_combine_by_priority_fills_from_later_strategies_when_earlier_insufficient():
    frames = [
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "a",
                    "rank": 1,
                    "score": 1.0,
                    "source": "first",
                    "_weight": 1.0,
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "b",
                    "rank": 1,
                    "score": 0.8,
                    "source": "second",
                    "_weight": 1.0,
                },
                {
                    "user_id": "u1",
                    "item_id": "c",
                    "rank": 2,
                    "score": 0.7,
                    "source": "second",
                    "_weight": 1.0,
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "d",
                    "rank": 1,
                    "score": 0.9,
                    "source": "third",
                    "_weight": 1.0,
                },
            ]
        ),
    ]

    combined = _combine_by_priority(frames, top_k=2)

    assert list(combined[Columns.User].unique()) == ["u1"]
    assert len(combined) == 2
    assert list(combined[Columns.Item]) == ["a", "b"]
    assert list(combined[Columns.Rank]) == [1, 2]


def test_combine_by_priority_recomputes_ranks_per_user():
    frames = [
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "a",
                    "rank": 1,
                    "score": 1.0,
                    "source": "first",
                    "_weight": 1.0,
                },
                {
                    "user_id": "u2",
                    "item_id": "b",
                    "rank": 1,
                    "score": 0.9,
                    "source": "first",
                    "_weight": 1.0,
                },
                {
                    "user_id": "u2",
                    "item_id": "c",
                    "rank": 2,
                    "score": 0.8,
                    "source": "first",
                    "_weight": 1.0,
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "item_id": "d",
                    "rank": 1,
                    "score": 0.7,
                    "source": "second",
                    "_weight": 1.0,
                },
                {
                    "user_id": "u1",
                    "item_id": "e",
                    "rank": 2,
                    "score": 0.6,
                    "source": "second",
                    "_weight": 1.0,
                },
                {
                    "user_id": "u2",
                    "item_id": "f",
                    "rank": 1,
                    "score": 0.85,
                    "source": "second",
                    "_weight": 1.0,
                },
            ]
        ),
    ]

    top_k = 2
    combined = _combine_by_priority(frames, top_k=top_k)

    counts = combined.groupby(Columns.User)[Columns.Item].count().to_dict()
    assert counts == {"u1": 2, "u2": 2}

    ranks_per_user = (
        combined.sort_values([Columns.User, Columns.Rank])
        .groupby(Columns.User)[Columns.Rank]
        .apply(list)
        .to_dict()
    )
    assert ranks_per_user == {"u1": [1, 2], "u2": [1, 2]}

    items_per_user = (
        combined.sort_values([Columns.User, Columns.Rank])
        .groupby(Columns.User)[Columns.Item]
        .apply(list)
        .to_dict()
    )
    assert items_per_user == {"u1": ["a", "d"], "u2": ["b", "c"]}

    for _, group in combined.groupby(Columns.User):
        ordered = group.sort_values(Columns.Rank)
        assert list(ordered[Columns.Rank]) == list(range(1, top_k + 1))


def test_train_and_recommend_respects_top_k_and_availability_filter(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    recommendations = train_and_recommend(
        built, target_users=["u1", "u2", "u3"], config=feature_config, top_k=2
    )

    assert set(recommendations[Columns.User]) == {"u1", "u2", "u3"}
    assert (recommendations.groupby(Columns.User).size() <= 2).all()
    assert not recommendations[Columns.Item].isin(["i3", "i4"]).any()
    assert set(recommendations["source"]) <= {"personalized", "item_based", "popular_fallback"}


def test_train_and_recommend_falls_back_to_popularity_for_cold_users(
    sample_items, feature_config, sample_users
):
    events = _synthetic_events()
    # u4: features only (hybrid cold-start via rectools).
    built = build_dataset(events, sample_users, sample_items, feature_config, half_life_days=90)

    recommendations = train_and_recommend(built, target_users=["u1", "u4"], config=feature_config, top_k=2)

    warm_via_features = recommendations[recommendations[Columns.User] == "u4"]
    assert not warm_via_features.empty


def test_train_and_recommend_falls_back_to_popularity_for_fully_unknown_users(sample_items, feature_config):
    events = _synthetic_events()
    # ghost: no interactions/features → popularity fallback.
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    recommendations = train_and_recommend(built, target_users=["u1", "ghost"], config=feature_config, top_k=2)

    cold_user_recos = recommendations[recommendations[Columns.User] == "ghost"]
    assert not cold_user_recos.empty
    assert (cold_user_recos["source"] == "popular_fallback").all()


def test_train_and_recommend_rejects_unknown_model(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    with pytest.raises(ValueError, match="not_a_real_model"):
        train_and_recommend(
            built, target_users=["u1"], config=feature_config, top_k=2, enabled_models=["not_a_real_model"]
        )


def test_train_and_recommend_rejects_empty_enabled_models(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    # Empty list ≠ omit; must raise, not succeed empty.
    with pytest.raises(ValueError, match="enabled_models is empty"):
        train_and_recommend(built, target_users=["u1"], config=feature_config, top_k=2, enabled_models=[])


def test_train_and_recommend_item_based_strategy(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    recommendations = train_and_recommend(
        built, target_users=["u1", "u2", "u3"], config=feature_config, top_k=2, enabled_models=["item_based"]
    )

    assert set(recommendations["source"]) == {"item_based"}


def test_train_and_recommend_latest_strategy(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    recommendations = train_and_recommend(
        built, target_users=["u1", "u2", "u3"], config=feature_config, top_k=2, enabled_models=["latest"]
    )

    assert set(recommendations[Columns.User]) == {"u1", "u2", "u3"}
    assert set(recommendations["source"]) == {"latest"}


def test_train_and_recommend_combines_multiple_personalized_strategies(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    recommendations = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=3,
        enabled_models=["collaborative", "item_based", "popular"],
    )

    assert set(recommendations["source"]) <= {"personalized", "item_based", "popular_fallback"}
    assert set(recommendations.columns) == {
        Columns.User,
        Columns.Item,
        Columns.Rank,
        Columns.Score,
        "source",
    }
    assert (recommendations.groupby(Columns.User).size() <= 3).all()
    assert not recommendations.duplicated(subset=[Columns.User, Columns.Item]).any()


def test_strategies_keys_match_config_strategy_names():
    # Keep STRATEGIES in sync with config STRATEGY_NAMES.
    assert set(STRATEGIES) == set(STRATEGY_NAMES)


def test_validate_strategy_names_raises_on_mismatch():
    with pytest.raises(RuntimeError, match="must match"):
        _validate_strategy_names({"popular": STRATEGIES["popular"]}, ("popular", "latest"))


def test_as_recommender_model_accepts_every_registered_strategy_factory():
    for name, strategy in STRATEGIES.items():
        model = strategy.factory()
        assert _as_recommender_model(model) is model, name


def test_as_recommender_model_rejects_object_missing_recommend():
    class NotAModel:
        def fit(self, dataset):
            return self

    with pytest.raises(TypeError, match="does not implement the RecommenderModel protocol"):
        _as_recommender_model(NotAModel())


def test_as_recommender_model_rejects_recommend_missing_expected_parameters():
    class WrongSignatureModel:
        def fit(self, dataset):
            return self

        def recommend(self, *, users, dataset, k):
            return pd.DataFrame()

    with pytest.raises(TypeError, match="missing expected parameter"):
        _as_recommender_model(WrongSignatureModel())


def test_validate_model_weights_no_op_when_none():
    validate_model_weights(None)


def test_train_and_recommend_no_warm_users_and_only_personalized_strategies_returns_empty(
    sample_items, feature_config, caplog
):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    with caplog.at_level("INFO"):
        recommendations = train_and_recommend(
            built, target_users=["ghost"], config=feature_config, top_k=2, enabled_models=["item_based"]
        )

    assert recommendations.empty
    assert list(recommendations.columns) == [
        Columns.User,
        Columns.Item,
        Columns.Rank,
        Columns.Score,
        "source",
    ]
    # Must say no fallback available, not "falling back".
    assert "no non-personalized strategy is enabled" in caplog.text
    assert "falling back" not in caplog.text


def test_train_and_recommend_rejects_unknown_weight_key(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    with pytest.raises(ValueError, match="not_enabled"):
        train_and_recommend(
            built,
            target_users=["u1"],
            config=feature_config,
            top_k=2,
            enabled_models=["popular"],
            weights={"not_enabled": 1.0},
        )


def test_train_and_recommend_rejects_negative_weight(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    with pytest.raises(ValueError, match="non-negative"):
        train_and_recommend(
            built,
            target_users=["u1"],
            config=feature_config,
            top_k=2,
            enabled_models=["popular"],
            weights={"popular": -1.0},
        )


def test_train_and_recommend_rejects_non_positive_rrf_k(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    with pytest.raises(ValueError, match="rrf_k must be positive"):
        train_and_recommend(
            built,
            target_users=["u1"],
            config=feature_config,
            top_k=2,
            enabled_models=["popular"],
            weights={"popular": 1.0},
            rrf_k=0,
        )


def test_train_and_recommend_rejects_weights_for_models_dropped_from_recommend(sample_items, feature_config):
    """Weights must match the resolved recommend set, not just the requested models list."""
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    with pytest.raises(ValueError, match="not in recommend models"):
        train_and_recommend(
            built,
            target_users=["u1"],
            config=feature_config,
            top_k=2,
            enabled_models=["collaborative", "content_fallback", "popular"],
            weights={"collaborative": 1.0, "content_fallback": 0.5, "popular": 0.3},
            content_fallback_enabled=False,
        )


def test_train_and_recommend_weighted_fusion_with_default_models(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    # weights alone still trigger fusion against DEFAULT_MODELS.
    weights = {"collaborative": 1.0, "item_based": 0.5, "popular": 0.3}
    from_default = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=5,
        weights=weights,
    )
    from_explicit = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=5,
        enabled_models=DEFAULT_MODELS,
        weights=weights,
    )

    fused_labels = {
        "personalized",
        "item_based",
        "popular_fallback",
        "personalized+item_based",
        "personalized+popular_fallback",
        "item_based+popular_fallback",
        "personalized+item_based+popular_fallback",
    }
    assert set(from_default["source"]) <= fused_labels
    pd.testing.assert_frame_equal(from_default.reset_index(drop=True), from_explicit.reset_index(drop=True))


def test_train_and_recommend_weighted_fusion_respects_top_k_and_ranks_by_score(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    recommendations = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=2,
        enabled_models=["collaborative", "item_based", "popular"],
        weights={"collaborative": 1.0, "item_based": 0.5, "popular": 0.2},
    )

    assert (recommendations.groupby(Columns.User).size() <= 2).all()
    for _, group in recommendations.groupby(Columns.User):
        assert list(group[Columns.Rank]) == list(range(1, len(group) + 1))
        assert list(group[Columns.Score]) == sorted(group[Columns.Score], reverse=True)


def test_train_and_recommend_weighted_fusion_merges_sources_for_shared_items(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    recommendations = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=5,
        enabled_models=["popular", "latest"],
        weights={"popular": 1.0, "latest": 1.0},
    )

    assert set(recommendations["source"]) == {"popular_fallback+latest"}


def test_train_and_recommend_weighted_fusion_joins_labels_in_enabled_models_order(
    sample_items, feature_config
):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    # Source label must follow enabled_models order, not alphabetical.
    popular_first = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=5,
        enabled_models=["popular", "latest"],
        weights={"popular": 1.0, "latest": 1.0},
    )
    latest_first = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=5,
        enabled_models=["latest", "popular"],
        weights={"popular": 1.0, "latest": 1.0},
    )

    assert set(popular_first["source"]) == {"popular_fallback+latest"}
    assert set(latest_first["source"]) == {"latest+popular_fallback"}


def test_train_and_recommend_reuses_strategy_cache_across_calls(sample_items, feature_config, monkeypatch):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    fit_calls = []
    original_factory = STRATEGIES["popular"].factory

    def counting_factory():
        model = original_factory()
        original_fit = model.fit

        def counting_fit(dataset):
            fit_calls.append(1)
            return original_fit(dataset)

        model.fit = counting_fit
        return model

    monkeypatch.setitem(
        STRATEGIES, "popular", Strategy(counting_factory, personalized=False, source_label="popular_fallback")
    )

    cache: dict[str, RecommenderModel] = {}
    first = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=2,
        enabled_models=["popular"],
        strategy_cache=cache,
    )
    second = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=2,
        enabled_models=["popular"],
        strategy_cache=cache,
    )

    assert len(fit_calls) == 1
    assert "popular" in cache
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_train_and_recommend_strategy_cache_reused_across_different_top_k_and_weights(
    sample_items, feature_config, monkeypatch
):
    # Cache fitted models, not recommend() output — different top_k/weights reuse the fit.
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    fit_calls = []
    original_factory = STRATEGIES["popular"].factory

    def counting_factory():
        model = original_factory()
        original_fit = model.fit

        def counting_fit(dataset):
            fit_calls.append(1)
            return original_fit(dataset)

        model.fit = counting_fit
        return model

    monkeypatch.setitem(
        STRATEGIES, "popular", Strategy(counting_factory, personalized=False, source_label="popular_fallback")
    )

    cache: dict[str, RecommenderModel] = {}
    small_top_k = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=1,
        enabled_models=["popular"],
        strategy_cache=cache,
    )
    large_top_k = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=5,
        enabled_models=["popular"],
        weights={"popular": 2.0},
        strategy_cache=cache,
    )

    assert len(fit_calls) == 1
    assert (small_top_k.groupby(Columns.User).size() <= 1).all()
    assert (large_top_k.groupby(Columns.User).size() <= 5).all()


def test_train_and_recommend_without_cache_refits_every_call(sample_items, feature_config, monkeypatch):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    fit_calls = []
    original_factory = STRATEGIES["popular"].factory

    def counting_factory():
        model = original_factory()
        original_fit = model.fit

        def counting_fit(dataset):
            fit_calls.append(1)
            return original_fit(dataset)

        model.fit = counting_fit
        return model

    monkeypatch.setitem(
        STRATEGIES, "popular", Strategy(counting_factory, personalized=False, source_label="popular_fallback")
    )

    train_and_recommend(
        built, target_users=["u1", "u2", "u3"], config=feature_config, top_k=2, enabled_models=["popular"]
    )
    train_and_recommend(
        built, target_users=["u1", "u2", "u3"], config=feature_config, top_k=2, enabled_models=["popular"]
    )

    assert len(fit_calls) == 2


def test_train_and_recommend_parallel_fit_matches_sequential(sample_items, feature_config):
    # Deterministic strategies: parallel fit must match sequential.
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)
    kwargs = dict(
        built=built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=2,
        enabled_models=["popular", "latest"],
    )

    sequential = train_and_recommend(**kwargs, max_workers=1)
    # max_workers > model count must be capped, not raise.
    parallel = train_and_recommend(**kwargs, max_workers=10)

    pd.testing.assert_frame_equal(sequential.reset_index(drop=True), parallel.reset_index(drop=True))


def test_train_and_recommend_parallel_fit_populates_strategy_cache(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)
    cache: dict[str, RecommenderModel] = {}

    train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=2,
        enabled_models=["popular", "latest"],
        strategy_cache=cache,
        max_workers=2,
    )

    assert set(cache) == {"popular", "latest"}


def test_train_and_recommend_empty_weights_dict_enables_fusion(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    # Empty {} still opts into fusion (implicit weight 1.0).
    recommendations = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=5,
        enabled_models=["popular", "latest"],
        weights={},
    )

    assert set(recommendations["source"]) == {"popular_fallback+latest"}


def test_train_and_recommend_weighted_fusion_defaults_missing_weight_to_one(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    # Omitted weight key defaults to 1.0.
    partial = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=5,
        enabled_models=["popular", "latest"],
        weights={"latest": 0.5},
    )
    explicit_default = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=5,
        enabled_models=["popular", "latest"],
        weights={"popular": 1.0, "latest": 0.5},
    )
    explicit_changed = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=5,
        enabled_models=["popular", "latest"],
        weights={"popular": 0.3, "latest": 0.5},
    )

    assert set(partial["source"]) == {"popular_fallback+latest"}

    merged_default = partial.merge(
        explicit_default, on=[Columns.User, Columns.Item], suffixes=("_partial", "_explicit")
    )
    assert not merged_default.empty
    assert (merged_default[f"{Columns.Score}_partial"] == merged_default[f"{Columns.Score}_explicit"]).all()

    merged_changed = partial.merge(
        explicit_changed, on=[Columns.User, Columns.Item], suffixes=("_partial", "_changed")
    )
    assert not merged_changed.empty
    assert (merged_changed[f"{Columns.Score}_partial"] != merged_changed[f"{Columns.Score}_changed"]).any()


def test_train_and_recommend_custom_rrf_k_changes_fused_scores(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    small_k = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=5,
        enabled_models=["popular", "latest"],
        weights={"popular": 1.0, "latest": 1.0},
        rrf_k=1,
    )
    large_k = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=5,
        enabled_models=["popular", "latest"],
        weights={"popular": 1.0, "latest": 1.0},
        rrf_k=1000,
    )

    # Larger rrf_k strictly lowers weight/(rrf_k+rank) for same pairs.
    merged = small_k.merge(large_k, on=[Columns.User, Columns.Item], suffixes=("_small_k", "_large_k"))
    assert not merged.empty
    assert (merged[Columns.Score + "_small_k"] > merged[Columns.Score + "_large_k"]).all()


def test_train_and_recommend_respects_per_user_eligibility(feature_config):
    from dataclasses import replace

    from cicerone.feature_config import EligibilityRule

    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    )
    users = pd.DataFrame(
        [
            {"user_id": "u1", "nationality": "IT", "favorite_styles": ["ipa"], "region_slug": "lazio"},
            {"user_id": "u2", "nationality": "US", "favorite_styles": ["lager"], "region_slug": "toscana"},
        ]
    )
    items = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "category": "beer",
                "producer_id": "p1",
                "published": True,
                "in_stock": True,
                "available_countries": ["IT", "FR"],
            },
            {
                "item_id": "i2",
                "category": "beer",
                "producer_id": "p2",
                "published": True,
                "in_stock": True,
                "available_countries": ["US"],
            },
        ]
    )
    config = replace(
        feature_config,
        eligibility=[
            EligibilityRule(
                name="ships_to_user",
                op="user_in_item_list",
                item_column="available_countries",
                user_column="nationality",
            )
        ],
    )
    built = build_dataset(events, users, items, config, half_life_days=90)
    recommendations = train_and_recommend(
        built,
        target_users=["u1", "u2"],
        config=config,
        top_k=2,
        enabled_models=["popular"],
    )

    u1_items = set(recommendations.loc[recommendations[Columns.User] == "u1", Columns.Item])
    u2_items = set(recommendations.loc[recommendations[Columns.User] == "u2", Columns.Item])
    assert u1_items <= {"i1"}
    assert u2_items <= {"i2"}
    assert "i2" not in u1_items
    assert "i1" not in u2_items


def test_train_and_recommend_empty_users_frame_skips_user_scoped_rules(feature_config, caplog):
    """Empty users.parquet must not enable broken cohort mode (treat like missing)."""
    from dataclasses import replace

    from cicerone.feature_config import EligibilityRule

    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "category": "beer",
                "producer_id": "p1",
                "published": True,
                "in_stock": True,
                "available_countries": ["IT"],
            },
            {
                "item_id": "i2",
                "category": "beer",
                "producer_id": "p2",
                "published": True,
                "in_stock": True,
                "available_countries": ["US"],
            },
        ]
    )
    config = replace(
        feature_config,
        eligibility=[
            EligibilityRule(
                name="ships_to_user",
                op="user_in_item_list",
                item_column="available_countries",
                user_column="nationality",
            )
        ],
    )
    built = build_dataset(events, None, items, config, half_life_days=90)
    # Empty users frame (e.g. empty users.parquet).
    built = replace(built, users=pd.DataFrame(columns=["user_id", "nationality"]))

    recommendations = train_and_recommend(
        built,
        target_users=["u1", "u2"],
        config=config,
        top_k=2,
        enabled_models=["popular"],
    )

    assert "no users frame is available" in caplog.text
    for uid in ("u1", "u2"):
        got = set(recommendations.loc[recommendations[Columns.User] == uid, Columns.Item])
        assert got == {"i1", "i2"}


def test_train_and_recommend_paying_producer_boost_reorders(feature_config):
    from dataclasses import replace

    from cicerone.feature_config import BoostRule

    now = pd.Timestamp.utcnow()
    # Equal popularity → boost alone decides order.
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u3", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u3", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "category": "beer",
                "producer_id": "p1",
                "published": True,
                "in_stock": True,
                "is_paying_producer": False,
            },
            {
                "item_id": "i2",
                "category": "beer",
                "producer_id": "p2",
                "published": True,
                "in_stock": True,
                "is_paying_producer": True,
            },
        ]
    )
    config = replace(
        feature_config,
        boosts=[
            BoostRule(
                name="paying_producer",
                kind="boolean",
                item_column="is_paying_producer",
                factor=10.0,
            )
        ],
    )
    built = build_dataset(events, None, items, config, half_life_days=90)
    recommendations = train_and_recommend(
        built,
        target_users=["ghost"],
        config=config,
        top_k=2,
        enabled_models=["popular"],
    )
    ghost = recommendations[recommendations[Columns.User] == "ghost"].sort_values(Columns.Rank)
    assert list(ghost[Columns.Item])[0] == "i2"
    assert not ghost[Columns.Item].isin(["i3", "i4"]).any()


def test_train_and_recommend_paying_producer_boost_overfetch(feature_config):
    """Boosted item ranked just below top_k by popularity must enter final top_k
    via boost over-fetch + apply_boosts, not only by reordering within top_k.
    """
    from dataclasses import replace

    from cicerone.feature_config import DEFAULT_BOOST_OVERFETCH_FACTOR, BoostRule
    from cicerone.model import _recommend_k

    assert DEFAULT_BOOST_OVERFETCH_FACTOR > 1
    assert _recommend_k(2, True, overfetch_factor=5) == 10
    assert feature_config.boost_overfetch_factor >= 1

    now = pd.Timestamp.utcnow()
    # Over-fetch + boost can promote i3 past popularity-only top_k.
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u3", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i3", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "category": "beer",
                "producer_id": "p1",
                "published": True,
                "in_stock": True,
                "is_paying_producer": False,
            },
            {
                "item_id": "i2",
                "category": "beer",
                "producer_id": "p2",
                "published": True,
                "in_stock": True,
                "is_paying_producer": False,
            },
            {
                "item_id": "i3",
                "category": "beer",
                "producer_id": "p3",
                "published": True,
                "in_stock": True,
                "is_paying_producer": True,
            },
        ]
    )
    baseline_config = replace(feature_config, boosts=[])
    boosted_config = replace(
        feature_config,
        boosts=[
            BoostRule(
                name="paying_producer",
                kind="boolean",
                item_column="is_paying_producer",
                factor=100.0,
            )
        ],
    )
    built = build_dataset(events, None, items, boosted_config, half_life_days=90)

    baseline = train_and_recommend(
        built,
        target_users=["ghost"],
        config=baseline_config,
        top_k=2,
        enabled_models=["popular"],
    )
    baseline_items = list(
        baseline.loc[baseline[Columns.User] == "ghost"].sort_values(Columns.Rank)[Columns.Item]
    )
    assert baseline_items == ["i1", "i2"]
    assert "i3" not in baseline_items

    boosted = train_and_recommend(
        built,
        target_users=["ghost"],
        config=boosted_config,
        top_k=2,
        enabled_models=["popular"],
    )
    boosted_items = list(
        boosted.loc[boosted[Columns.User] == "ghost"].sort_values(Columns.Rank)[Columns.Item]
    )
    assert len(boosted_items) == 2
    assert "i3" in boosted_items
    assert boosted_items[0] == "i3"


def test_topk_extraction_preserves_external_ids_no_duplicates_or_seen_items(feature_config):
    """Regression guard for internal↔external ID mapping bugs in top-K extraction.

    External user/item IDs are sparse integers that do not match rectools'
    dense 0..n-1 internal indices. If recommend() (or a hand-rolled top-K)
    leaked internal indices, they would not be in the external catalog.
    Personalized strategies must also exclude already-seen items and never
    emit duplicate item IDs for the same user.
    """
    now = pd.Timestamp.utcnow()
    # IDs far from 0..n-1 so reindex leaks can't look valid.
    external_users = [1000, 2000, 3000]
    external_items = [100, 200, 300, 400, 500, 600]
    # Distinct seen-blocks per warm user for unambiguous asserts.
    seen_by_user = {
        1000: [100, 200, 300, 400, 500],
        2000: [200, 300, 400, 500, 600],
        3000: [100, 300, 400, 500, 600],
    }
    rows = []
    for user, items in seen_by_user.items():
        for item in items:
            rows.append(
                {
                    "user_id": user,
                    "item_id": item,
                    "event_type": "purchase",
                    "quantity": 1,
                    "occurred_at": now,
                }
            )
    events = pd.DataFrame(rows)
    items = pd.DataFrame(
        [
            {
                "item_id": item_id,
                "category": "beer",
                "producer_id": "p1",
                "published": True,
                "in_stock": True,
            }
            for item_id in external_items
        ]
    )
    built = build_dataset(events, None, items, feature_config, half_life_days=90)

    assert list(built.dataset.item_id_map.external_ids) == external_items
    assert set(built.dataset.item_id_map.internal_ids) == set(range(len(external_items)))
    assert set(built.dataset.item_id_map.internal_ids).isdisjoint(external_items)

    recommendations = train_and_recommend(
        built,
        target_users=external_users,
        config=feature_config,
        top_k=3,
        enabled_models=["collaborative", "popular"],
    )

    assert not recommendations.empty
    assert set(recommendations[Columns.Item]).issubset(external_items)
    assert set(recommendations[Columns.User]).issubset(external_users)
    assert set(recommendations[Columns.Item]).isdisjoint(range(len(external_items)))
    assert set(recommendations[Columns.User]).isdisjoint(range(len(external_users)))
    assert not recommendations.duplicated(subset=[Columns.User, Columns.Item]).any()

    personalized = recommendations[recommendations["source"] == "personalized"]
    for user_id, group in personalized.groupby(Columns.User):
        seen = set(seen_by_user[int(user_id)])
        assert set(group[Columns.Item]).isdisjoint(seen)


def test_should_log_epoch_first_last_and_interval():
    from cicerone.model import _should_log_epoch

    assert _should_log_epoch(1, 30, 5)
    assert _should_log_epoch(5, 30, 5)
    assert _should_log_epoch(30, 30, 5)
    assert not _should_log_epoch(2, 30, 5)
    assert not _should_log_epoch(29, 30, 5)


def test_warn_on_epoch_metric_trajectory_regression_and_plateau(caplog):
    from cicerone.config import EpochMetricsSettings
    from cicerone.model import _warn_on_epoch_metric_trajectory

    settings = EpochMetricsSettings(every=5)

    with caplog.at_level("WARNING"):
        _warn_on_epoch_metric_trajectory([(1, {"Precision@2": 0.5})], settings)
    assert caplog.text == ""

    with caplog.at_level("WARNING"):
        _warn_on_epoch_metric_trajectory(
            [
                (1, {"Precision@2": 0.8}),
                (5, {"Precision@2": 0.7}),
                (10, {"Precision@2": 0.4}),
            ],
            settings,
        )
    assert "regressed" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        _warn_on_epoch_metric_trajectory(
            [
                (1, {"Recall@2": 0.50}),
                (5, {"Recall@2": 0.501}),
                (10, {"Recall@2": 0.502}),
            ],
            settings,
        )
    assert "plateaued" in caplog.text


def test_warn_on_epoch_metric_trajectory_skips_metrics_missing_from_some_snapshots(caplog):
    from cicerone.config import EpochMetricsSettings
    from cicerone.model import _warn_on_epoch_metric_trajectory

    # Heterogeneous keys: no KeyError; singleton values skip regression WARN.
    with caplog.at_level("WARNING"):
        _warn_on_epoch_metric_trajectory(
            [
                (1, {"Precision@2": 0.9, "Recall@2": 0.5}),
                (5, {"Precision@2": 0.4}),
            ],
            EpochMetricsSettings(every=5),
        )
    assert "Precision@2" in caplog.text
    assert "regressed" in caplog.text
    assert "Recall@2" not in caplog.text


def test_fit_lightfm_with_epoch_metrics_rejects_non_lightfm_wrapper():
    from cicerone.config import EpochMetricsSettings
    from cicerone.model import _fit_lightfm_with_epoch_metrics

    class NoPartial:
        def fit(self, dataset):
            return self

        def recommend(self, **kwargs):
            return pd.DataFrame()

    with pytest.raises(TypeError, match="fit_partial"):
        _fit_lightfm_with_epoch_metrics(
            NoPartial(), None, pd.DataFrame(), settings=EpochMetricsSettings(every=1), top_k=2
        )


def test_epoch_metric_total_epochs_prefers_n_epochs_then_epochs():
    from cicerone.model import _epoch_metric_total_epochs

    class WithNEpochs:
        n_epochs = 7

    class WithEpochs:
        epochs = 3

    class WithNeither:
        pass

    assert _epoch_metric_total_epochs(WithNEpochs()) == 7
    assert _epoch_metric_total_epochs(WithEpochs()) == 3
    with pytest.raises(TypeError, match="n_epochs/epochs"):
        _epoch_metric_total_epochs(WithNeither())


def test_warn_on_epoch_metric_trajectory_plateau_scale_handles_negative_values(caplog):
    from cicerone.config import EpochMetricsSettings
    from cicerone.model import _warn_on_epoch_metric_trajectory

    # All-negative window: scale by max(|v|), not abs(max(v)).
    with caplog.at_level("WARNING"):
        _warn_on_epoch_metric_trajectory(
            [
                (1, {"Score": -10.0}),
                (5, {"Score": -10.01}),
                (10, {"Score": -10.02}),
            ],
            EpochMetricsSettings(every=5),
        )
    assert "plateaued" in caplog.text


def test_sample_epoch_metric_users_is_seeded_not_prefix():
    from cicerone.model import RANDOM_STATE, _sample_epoch_metric_users

    ordered = list(range(1000))
    sampled = _sample_epoch_metric_users(ordered, max_users=10)
    assert len(sampled) == 10
    assert sampled != ordered[:10]
    assert _sample_epoch_metric_users(ordered, max_users=10) == sampled
    assert set(sampled).issubset(ordered)
    assert _sample_epoch_metric_users([3, 1, 2], max_users=10) == [3, 1, 2]
    assert RANDOM_STATE == 42


def test_train_and_recommend_logs_epoch_metrics_when_configured(
    sample_items, feature_config, monkeypatch, caplog
):
    import cicerone.model as model_module
    from cicerone.config import EpochMetricsSettings

    # Short epoch loop so this stays a unit test.
    monkeypatch.setattr(model_module, "COLLABORATIVE_EPOCHS", 4)
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    with caplog.at_level("INFO"):
        recommendations = train_and_recommend(
            built,
            target_users=["u1", "u2", "u3"],
            config=feature_config,
            top_k=2,
            enabled_models=["collaborative"],
            epoch_metrics=EpochMetricsSettings(every=2),
        )

    assert not recommendations.empty
    assert "Collaborative epoch 1/4 metrics:" in caplog.text
    assert "Collaborative epoch 2/4 metrics:" in caplog.text
    assert "Collaborative epoch 4/4 metrics:" in caplog.text
    assert "Collaborative epoch 3/4 metrics:" not in caplog.text
    assert "Precision@2" in caplog.text
    assert "Recall@2" in caplog.text


def test_train_and_recommend_skips_epoch_metrics_by_default(sample_items, feature_config, caplog):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    with caplog.at_level("INFO"):
        train_and_recommend(
            built,
            target_users=["u1", "u2", "u3"],
            config=feature_config,
            top_k=2,
            enabled_models=["collaborative"],
        )

    assert "Collaborative epoch" not in caplog.text


def test_default_models_three_tier_chain():
    assert DEFAULT_MODELS == ["collaborative", "item_based", "popular"]


def test_resolve_recommend_models_inserts_content_fallback_before_popular():
    from cicerone.model import resolve_recommend_models

    assert resolve_recommend_models(
        ["collaborative", "item_based", "popular"],
        blending_enabled=False,
        content_fallback_enabled=True,
    ) == ["collaborative", "item_based", "content_fallback", "popular"]


def test_resolve_recommend_models_skips_content_fallback_when_disabled(caplog):
    from cicerone.model import resolve_recommend_models

    with caplog.at_level("INFO"):
        resolved = resolve_recommend_models(
            ["collaborative", "content_fallback", "popular"],
            blending_enabled=False,
            content_fallback_enabled=False,
        )
    assert resolved == ["collaborative", "popular"]
    assert "content_fallback is listed" in caplog.text


def test_resolve_run_models_centralizes_content_fallback_and_blending():
    from cicerone.model import content_fallback_enabled_from_models, plan_model_run, resolve_run_models

    resolved, recommend = resolve_run_models(
        ["collaborative", "popular"],
        blending_enabled=False,
        content_fallback_enabled=True,
    )
    assert resolved == ["collaborative", "popular"]
    assert recommend == ["collaborative", "content_fallback", "popular"]
    assert content_fallback_enabled_from_models(recommend) is True
    assert content_fallback_enabled_from_models(["collaborative", "popular"]) is False

    # Artifact/AutoML path: omit the flag and derive from the stored list.
    plan = plan_model_run(
        ["collaborative", "content_fallback", "popular"],
        blending_enabled=False,
        content_fallback_enabled=None,
    )
    assert plan.content_fallback_active is True
    assert list(plan.recommend_models) == ["collaborative", "content_fallback", "popular"]


def test_item_based_k_neighbors_reaches_tfidf_recommender(sample_items, feature_config, monkeypatch):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)
    seen: list[int] = []
    real_tfidf = TFIDFRecommender

    def tracking_tfidf(*args, **kwargs):
        seen.append(kwargs["K"])
        return real_tfidf(*args, **kwargs)

    monkeypatch.setattr("cicerone.model.TFIDFRecommender", tracking_tfidf)

    train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=2,
        enabled_models=["item_based"],
        item_based_k_neighbors=7,
    )
    assert seen == [7]


def test_content_fallback_surfaces_cold_item_for_matching_user(feature_config):
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            # u1 is a beer buyer — should match cold beer i_new.
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            # u2 is a wine buyer — must not receive cold beer i_new.
            {"user_id": "u2", "item_id": "i3", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
            {"item_id": "i3", "category": "wine", "producer_id": "p1", "published": True, "in_stock": True},
            # Brand-new beer with zero events — similar to u1 history, not u2.
            {
                "item_id": "i_new",
                "category": "beer",
                "producer_id": "p9",
                "published": True,
                "in_stock": True,
            },
        ]
    )
    built = build_dataset(events, None, items, feature_config, half_life_days=90)

    recommendations = train_and_recommend(
        built,
        target_users=["u1", "u2"],
        config=feature_config,
        top_k=5,
        enabled_models=["content_fallback"],
        content_fallback_enabled=True,
    )

    u1 = recommendations[recommendations[Columns.User] == "u1"]
    assert not u1.empty
    assert "i_new" in set(u1[Columns.Item])
    assert (u1[u1[Columns.Item] == "i_new"]["source"] == "content_fallback").all()

    u2 = recommendations[recommendations[Columns.User] == "u2"]
    assert "i_new" not in set(u2[Columns.Item])


def test_content_fallback_disabled_emits_no_content_source(feature_config):
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {
                "item_id": "i_new",
                "category": "beer",
                "producer_id": "p9",
                "published": True,
                "in_stock": True,
            },
        ]
    )
    built = build_dataset(events, None, items, feature_config, half_life_days=90)

    recommendations = train_and_recommend(
        built,
        target_users=["u1"],
        config=feature_config,
        top_k=5,
        enabled_models=["content_fallback", "popular"],
        content_fallback_enabled=False,
    )

    assert "content_fallback" not in set(recommendations["source"])
    assert not recommendations.empty


def test_content_fallback_respects_availability_filters(feature_config):
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {
                "item_id": "i_new_ok",
                "category": "beer",
                "producer_id": "p9",
                "published": True,
                "in_stock": True,
            },
            {
                "item_id": "i_new_blocked",
                "category": "beer",
                "producer_id": "p9",
                "published": False,
                "in_stock": True,
            },
        ]
    )
    built = build_dataset(events, None, items, feature_config, half_life_days=90)

    recommendations = train_and_recommend(
        built,
        target_users=["u1"],
        config=feature_config,
        top_k=5,
        enabled_models=["content_fallback"],
        content_fallback_enabled=True,
    )

    recommended_items = set(recommendations[Columns.Item])
    assert "i_new_ok" in recommended_items
    assert "i_new_blocked" not in recommended_items


def test_item_based_and_content_fallback_require_interactions(feature_config, sample_users):
    """Feature-only warm users must not get item_based / content_fallback rows."""
    now = pd.Timestamp.utcnow()
    # u4 is feature-only in sample_users; give u1 beer interactions + a cold beer.
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
            {
                "item_id": "i_new",
                "category": "beer",
                "producer_id": "p9",
                "published": True,
                "in_stock": True,
            },
        ]
    )
    built = build_dataset(events, sample_users, items, feature_config, half_life_days=90)
    gated_models = ["item_based", "content_fallback"]
    gated_sources = {"item_based", "content_fallback"}

    # Full chain: feature-only u4 may get collaborative/popular, never gated sources.
    mixed = train_and_recommend(
        built,
        target_users=["u1", "u4"],
        config=feature_config,
        top_k=5,
        enabled_models=["collaborative", *gated_models, "popular"],
        content_fallback_enabled=True,
    )
    assert "u4" in set(mixed[Columns.User])
    feature_only = mixed[mixed[Columns.User] == "u4"]
    assert not feature_only["source"].isin(gated_sources).any()

    # Gated strategies alone: interacting user gets rows; feature-only gets none.
    interacting_only = train_and_recommend(
        built,
        target_users=["u1"],
        config=feature_config,
        top_k=5,
        enabled_models=gated_models,
        content_fallback_enabled=True,
    )
    assert not interacting_only.empty
    assert interacting_only["source"].isin(gated_sources).all()

    feature_only_gated = train_and_recommend(
        built,
        target_users=["u4"],
        config=feature_config,
        top_k=5,
        enabled_models=gated_models,
        content_fallback_enabled=True,
    )
    assert feature_only_gated.empty
