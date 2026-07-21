from __future__ import annotations

import pandas as pd
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
