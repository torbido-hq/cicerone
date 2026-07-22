from __future__ import annotations

import pandas as pd
import pytest
from rectools import Columns

from cicerone.dataset import build_dataset
from cicerone.model import _recommendable_item_ids, train_and_recommend


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
    assert set(recommendations["source"]) == {"latest+popular_fallback"}


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

    assert set(recommendations["source"]) == {"latest+popular_fallback"}


def test_train_and_recommend_custom_rrf_k_changes_fused_scores(sample_items, feature_config):
    events = _synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    default_k = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=5,
        enabled_models=["popular", "latest"],
        weights={"popular": 1.0, "latest": 1.0},
    )
    custom_k = train_and_recommend(
        built,
        target_users=["u1", "u2", "u3"],
        config=feature_config,
        top_k=5,
        enabled_models=["popular", "latest"],
        weights={"popular": 1.0, "latest": 1.0},
        rrf_k=1,
    )

    merged = default_k.merge(custom_k, on=[Columns.User, Columns.Item], suffixes=("_default", "_custom"))
    assert not merged[Columns.Score + "_default"].equals(merged[Columns.Score + "_custom"])
