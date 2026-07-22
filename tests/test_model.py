from __future__ import annotations

import pandas as pd
import pytest
from rectools import Columns

from cicerone.config import STRATEGY_NAMES
from cicerone.dataset import build_dataset
from cicerone.model import (
    STRATEGIES,
    Strategy,
    _recommendable_item_ids,
    _validate_strategy_names,
    train_and_recommend,
    validate_model_weights,
)


def _synthetic_events() -> pd.DataFrame:
    now = pd.Timestamp.utcnow()
    rows = []
    # Give each of u1/u2/u3 a handful of purchases so LightFM has signal.
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


def test_recommendable_item_ids_filters_on_all_configured_columns(sample_items):
    all_ids = pd.Index(["i1", "i2", "i3", "i4"])
    allowed = _recommendable_item_ids(sample_items, ["published", "in_stock"], all_ids)
    assert set(allowed) == {"i1", "i2"}


def test_recommendable_item_ids_skips_missing_column_and_warns(sample_items, caplog):
    all_ids = pd.Index(["i1", "i2", "i3", "i4"])
    allowed = _recommendable_item_ids(sample_items, ["published", "not_a_real_column"], all_ids)
    assert "not_a_real_column" in caplog.text
    assert set(allowed) == {"i1", "i2", "i3"}


def test_recommendable_item_ids_no_items_returns_all():
    all_ids = pd.Index(["i1", "i2"])
    assert _recommendable_item_ids(None, ["published"], all_ids) == ["i1", "i2"]


def test_recommendable_item_ids_no_filters_configured_returns_all(sample_items):
    all_ids = pd.Index(["i1", "i2", "i3", "i4"])
    assert _recommendable_item_ids(sample_items, [], all_ids) == ["i1", "i2", "i3", "i4"]


def test_train_and_recommend_respects_top_k_and_availability_filter(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    recommendations = train_and_recommend(
        built, target_users=["u1", "u2", "u3"], config=feature_config, top_k=2
    )

    assert set(recommendations[Columns.User]) == {"u1", "u2", "u3"}
    assert (recommendations.groupby(Columns.User).size() <= 2).all()
    # i3 is out of stock, i4 is unpublished — neither should ever be recommended.
    assert not recommendations[Columns.Item].isin(["i3", "i4"]).any()
    assert set(recommendations["source"]) <= {"personalized", "popular_fallback"}


def test_train_and_recommend_falls_back_to_popularity_for_cold_users(
    sample_items, feature_config, sample_users
):
    events = _synthetic_events()
    # u4 has features but never interacts -> rectools still knows it via
    # features (hybrid cold-start) and can produce personalized recs for it.
    built = build_dataset(events, sample_users, sample_items, feature_config, half_life_days=90)

    recommendations = train_and_recommend(built, target_users=["u1", "u4"], config=feature_config, top_k=2)

    warm_via_features = recommendations[recommendations[Columns.User] == "u4"]
    assert not warm_via_features.empty


def test_train_and_recommend_falls_back_to_popularity_for_fully_unknown_users(sample_items, feature_config):
    events = _synthetic_events()
    # "ghost" has no interactions and no features at all -> truly cold,
    # unknown to the dataset entirely -> must get the popularity fallback.
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

    # An explicit empty list is a configuration error, not "no strategies" --
    # it must not silently fall through to an empty-but-"successful" result.
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
    # top_k is enforced per user even after combining multiple strategies...
    assert (recommendations.groupby(Columns.User).size() <= 3).all()
    # ...and there are no duplicate (user, item) pairs across the combined strategies.
    assert not recommendations.duplicated(subset=[Columns.User, Columns.Item]).any()


def test_strategies_keys_match_config_strategy_names():
    # cicerone.config.STRATEGY_NAMES is the canonical list of valid model
    # identifiers (validated against at config-load time); it must stay in
    # sync with the strategies actually implemented here.
    assert set(STRATEGIES) == set(STRATEGY_NAMES)


def test_validate_strategy_names_raises_on_mismatch():
    with pytest.raises(RuntimeError, match="must match"):
        _validate_strategy_names({"popular": STRATEGIES["popular"]}, ("popular", "latest"))


def test_validate_model_weights_no_op_when_none():
    # No weights configured -> fusion mode isn't in play, nothing to validate.
    validate_model_weights(None)


def test_train_and_recommend_no_warm_users_and_only_personalized_strategies_returns_empty(
    sample_items, feature_config
):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

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

    # Both non-personalized strategies see every target user & all allowed
    # items, so every recommended pair should be backed by both sources.
    # Joined in enabled_models order ("popular" before "latest"), not
    # alphabetically.
    assert set(recommendations["source"]) == {"popular_fallback+latest"}


def test_train_and_recommend_weighted_fusion_joins_labels_in_enabled_models_order(
    sample_items, feature_config
):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    # Same two strategies, opposite enabled_models order -> the joined
    # source label should flip too, since it's meant to reflect the
    # configured priority order, not an alphabetical sort of source labels
    # ("latest" would otherwise always sort before "popular_fallback").
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

    cache: dict[str, pd.DataFrame] = {}
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


def test_train_and_recommend_empty_weights_dict_enables_fusion(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    # An explicitly empty weights dict is not the same as omitting weights:
    # it still opts into fusion mode (every strategy defaults to weight 1.0),
    # so the merged "+"-joined source label should appear, same as when
    # weights are given explicitly.
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

    # "popular" is omitted from weights -> should default to weight 1.0,
    # same as passing it explicitly.
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

    # Both models still contribute recommendations even though "popular"'s
    # weight is implicit.
    assert set(partial["source"]) == {"popular_fallback+latest"}

    # Omitting "popular" defaults it to weight 1.0, so fused scores should
    # match explicitly passing popular=1.0...
    merged_default = partial.merge(
        explicit_default, on=[Columns.User, Columns.Item], suffixes=("_partial", "_explicit")
    )
    assert not merged_default.empty
    assert (merged_default[f"{Columns.Score}_partial"] == merged_default[f"{Columns.Score}_explicit"]).all()

    # ...but changing popular's explicit weight away from the implicit
    # default of 1.0 should change the fused scores.
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

    # RRF fused score is weight / (rrf_k + rank): for a fixed (positive) rank
    # and weight, a larger rrf_k strictly lowers the score. Both runs recommend
    # the same (user, item) pairs here (only 2 allowed items per user), so
    # every pair should show this exact monotonic relationship.
    merged = small_k.merge(large_k, on=[Columns.User, Columns.Item], suffixes=("_small_k", "_large_k"))
    assert not merged.empty
    assert (merged[Columns.Score + "_small_k"] > merged[Columns.Score + "_large_k"]).all()
