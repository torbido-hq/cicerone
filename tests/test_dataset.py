from __future__ import annotations

import pandas as pd
import pytest
from rectools import Columns

from cicerone.dataset import _explode_features, _time_decay_multiplier, build_dataset, build_interactions
from cicerone.feature_config import FeatureColumn


def test_time_decay_multiplier_no_age_is_full_weight():
    now = pd.Series([pd.Timestamp.utcnow()])
    decay = _time_decay_multiplier(now, half_life_days=90)
    assert decay.iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_time_decay_multiplier_half_life_is_half_weight():
    occurred = pd.Series([pd.Timestamp.utcnow() - pd.Timedelta(days=90)])
    decay = _time_decay_multiplier(occurred, half_life_days=90)
    assert decay.iloc[0] == pytest.approx(0.5, abs=1e-3)


def test_time_decay_multiplier_future_timestamp_clips_to_full_weight():
    occurred = pd.Series([pd.Timestamp.utcnow() + pd.Timedelta(days=10)])
    decay = _time_decay_multiplier(occurred, half_life_days=90)
    assert decay.iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_weighted_interactions_scales_purchase_by_quantity(feature_config):
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i1", "event_type": "purchase", "quantity": 5, "occurred_at": now},
        ]
    )
    result = build_interactions(events, feature_config, half_life_days=90)

    row_u1 = result[result[Columns.User] == "u1"].iloc[0]
    row_u2 = result[result[Columns.User] == "u2"].iloc[0]
    assert row_u2[Columns.Weight] > row_u1[Columns.Weight]


def test_weighted_interactions_aggregates_multiple_events_for_same_pair(feature_config):
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "view", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i1", "event_type": "saved", "quantity": 1, "occurred_at": now},
        ]
    )
    result = build_interactions(events, feature_config, half_life_days=90)

    assert len(result) == 1
    expected = feature_config.event_weights["view"] + feature_config.event_weights["saved"]
    assert result.iloc[0][Columns.Weight] == pytest.approx(expected, rel=1e-3)


def test_weighted_interactions_drops_unknown_event_types(feature_config):
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "teleportation",
                "quantity": 1,
                "occurred_at": now,
            },
        ]
    )
    result = build_interactions(events, feature_config, half_life_days=90)
    assert result.empty


def test_weighted_interactions_caps_repeated_events(feature_config):
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "view", "quantity": 1, "occurred_at": now}
            for _ in range(20)
        ]
    )
    result = build_interactions(events, feature_config, half_life_days=90)

    cap = feature_config.event_caps["view"]
    max_weight = cap * feature_config.event_weights["view"]
    assert result.iloc[0][Columns.Weight] == pytest.approx(max_weight, rel=1e-3)


def test_weighted_interactions_negative_reviews_floor_at_epsilon(feature_config):
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "review_negative",
                "quantity": 1,
                "occurred_at": now,
            },
        ]
    )
    result = build_interactions(events, feature_config, half_life_days=90)

    assert result.iloc[0][Columns.Weight] > 0
    assert result.iloc[0][Columns.Weight] == pytest.approx(1e-3, rel=1e-6)


def test_explode_features_categorical_column():
    df = pd.DataFrame([{"item_id": "i1", "category": "beer"}, {"item_id": "i2", "category": None}])
    result = _explode_features(
        df, "item_id", Columns.Item, [FeatureColumn(column="category", type="categorical")]
    )

    assert list(result[Columns.Item]) == ["i1"]
    assert list(result["value"]) == ["beer"]
    assert list(result["feature"]) == ["category"]


def test_explode_features_list_column_produces_one_row_per_value():
    df = pd.DataFrame([{"user_id": "u1", "favorite_styles": ["ipa", "stout"]}])
    result = _explode_features(
        df, "user_id", Columns.User, [FeatureColumn(column="favorite_styles", type="list")]
    )

    assert sorted(result["value"]) == ["ipa", "stout"]


def test_explode_features_missing_column_is_skipped(caplog):
    df = pd.DataFrame([{"item_id": "i1"}])
    result = _explode_features(
        df, "item_id", Columns.Item, [FeatureColumn(column="does_not_exist", type="categorical")]
    )
    assert result.empty


def test_build_dataset_end_to_end(sample_events, sample_users, sample_items, feature_config):
    built = build_dataset(sample_events, sample_users, sample_items, feature_config, half_life_days=90)

    assert not built.interactions.empty
    assert built.items is sample_items
    assert set(built.dataset.user_id_map.external_ids) >= {"u1", "u2", "u3"}
    assert set(built.dataset.item_id_map.external_ids) >= {"i1", "i2", "i3"}


def test_build_dataset_without_users_or_items(sample_events, feature_config):
    built = build_dataset(sample_events, None, None, feature_config, half_life_days=90)

    assert built.items is None
    assert not built.interactions.empty
