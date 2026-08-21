from __future__ import annotations

import math

import pandas as pd
from rectools import Columns

from cicerone.blending import (
    BLENDED_SOURCE,
    COLD_START_USER_ID,
    LATEST_SOURCE,
    PERSONALIZED_SOURCE,
    POPULAR_SOURCE,
    append_cold_start_rows,
    blend_for_users,
    build_latest_ranking,
    expand_latest_ranking,
    interaction_counts,
    personalized_weight,
    rank_latest_items,
    resolve_latest_date_column,
    source_weights,
)
from cicerone.dataset import build_dataset
from cicerone.feature_config import BlendingConfig, FeatureConfig
from cicerone.model import train_and_recommend


def _blending(**overrides) -> BlendingConfig:
    base = dict(
        enabled=True,
        curve="sigmoid",
        midpoint=5.0,
        steepness=1.0,
        saturate_at=10.0,
        popular_share=0.7,
        latest_date_columns=("published_at", "created_at", "occurred_at"),
        rrf_k=60.0,
    )
    base.update(overrides)
    return BlendingConfig(**base)


def test_personalized_weight_sigmoid_cold_vs_rich():
    config = _blending(curve="sigmoid", midpoint=5.0, steepness=1.0)
    cold = personalized_weight(0, config)
    rich = personalized_weight(50, config)
    mid = personalized_weight(5, config)
    assert cold == 0.0
    assert rich > 0.9
    assert abs(mid - 0.5) < 1e-9


def test_personalized_weight_linear_saturates():
    config = _blending(curve="linear", saturate_at=10.0)
    assert personalized_weight(0, config) == 0.0
    assert personalized_weight(5, config) == 0.5
    assert personalized_weight(10, config) == 1.0
    assert personalized_weight(99, config) == 1.0


def test_personalized_weight_sigmoid_clamps_extreme_exponent():
    config = _blending(curve="sigmoid", midpoint=5.0, steepness=1e9)
    assert personalized_weight(10**12, config) == 1.0
    assert personalized_weight(0, config) == 0.0
    assert 0.0 <= personalized_weight(1, config) <= 1.0


def test_source_weights_redistribute_when_latest_unavailable():
    config = _blending(curve="linear", saturate_at=10.0, popular_share=0.7)
    with_latest = source_weights(0, config, latest_available=True)
    without = source_weights(0, config, latest_available=False)
    assert with_latest[PERSONALIZED_SOURCE] == 0.0
    assert math.isclose(with_latest[POPULAR_SOURCE], 0.7)
    assert math.isclose(with_latest[LATEST_SOURCE], 0.3)
    assert without[LATEST_SOURCE] == 0.0
    assert math.isclose(without[POPULAR_SOURCE], 1.0)


def test_resolve_latest_date_column_picks_first_usable():
    items = pd.DataFrame(
        [
            {"item_id": "i1", "created_at": "2024-01-01", "published_at": None},
            {"item_id": "i2", "created_at": "2024-02-01", "published_at": "not-a-date"},
        ]
    )
    assert resolve_latest_date_column(items, ["published_at", "created_at"]) == "created_at"
    assert resolve_latest_date_column(items, ["missing"]) is None
    assert resolve_latest_date_column(None, ["created_at"]) is None


def test_build_latest_ranking_orders_by_date_and_respects_allowlist():
    items = pd.DataFrame(
        [
            {"item_id": "old", "published_at": "2020-01-01"},
            {"item_id": "new", "published_at": "2024-06-01"},
            {"item_id": "blocked", "published_at": "2025-01-01"},
        ]
    )
    ranking = build_latest_ranking(
        items,
        "published_at",
        allowed_item_ids=["old", "new"],
        top_k=2,
        target_users=["u1", "u2"],
    )
    assert list(ranking[ranking[Columns.User] == "u1"][Columns.Item]) == ["new", "old"]
    assert set(ranking[Columns.User]) == {"u1", "u2"}
    assert (ranking["source"] == LATEST_SOURCE).all()
    assert "blocked" not in set(ranking[Columns.Item])


def test_blend_cold_start_prefers_popular_and_latest():
    config = _blending(curve="linear", saturate_at=10.0, popular_share=0.5)
    personalized = pd.DataFrame(
        [
            {
                Columns.User: "cold",
                Columns.Item: "p1",
                Columns.Rank: 1,
                Columns.Score: 9.0,
                "source": PERSONALIZED_SOURCE,
            }
        ]
    )
    popular = pd.DataFrame(
        [
            {
                Columns.User: "cold",
                Columns.Item: "pop1",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": POPULAR_SOURCE,
            },
            {
                Columns.User: "cold",
                Columns.Item: "pop2",
                Columns.Rank: 2,
                Columns.Score: 0.5,
                "source": POPULAR_SOURCE,
            },
        ]
    )
    latest = pd.DataFrame(
        [
            {
                Columns.User: "cold",
                Columns.Item: "lat1",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": LATEST_SOURCE,
            }
        ]
    )
    out = blend_for_users(
        personalized=personalized,
        popular=popular,
        latest=latest,
        counts={"cold": 0},
        target_users=["cold"],
        config=config,
        top_k=3,
        latest_available=True,
    )
    assert "p1" not in set(out[Columns.Item])
    assert set(out[Columns.Item]) <= {"pop1", "pop2", "lat1"}
    assert set(out["source"]) <= {POPULAR_SOURCE, LATEST_SOURCE, BLENDED_SOURCE}


def test_blend_rich_history_prefers_personalized():
    config = _blending(curve="linear", saturate_at=10.0, popular_share=0.5)
    personalized = pd.DataFrame(
        [
            {
                Columns.User: "rich",
                Columns.Item: "p1",
                Columns.Rank: 1,
                Columns.Score: 9.0,
                "source": PERSONALIZED_SOURCE,
            },
            {
                Columns.User: "rich",
                Columns.Item: "p2",
                Columns.Rank: 2,
                Columns.Score: 8.0,
                "source": PERSONALIZED_SOURCE,
            },
        ]
    )
    popular = pd.DataFrame(
        [
            {
                Columns.User: "rich",
                Columns.Item: "pop1",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": POPULAR_SOURCE,
            }
        ]
    )
    out = blend_for_users(
        personalized=personalized,
        popular=popular,
        latest=None,
        counts={"rich": 100},
        target_users=["rich"],
        config=config,
        top_k=2,
        latest_available=False,
    )
    assert list(out[Columns.Item]) == ["p1", "p2"]
    assert (out["source"] == PERSONALIZED_SOURCE).all()


def test_blend_mid_curve_mixes_sources():
    config = _blending(curve="linear", saturate_at=10.0, popular_share=0.5, rrf_k=1.0)
    personalized = pd.DataFrame(
        [
            {
                Columns.User: "mid",
                Columns.Item: "p1",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": PERSONALIZED_SOURCE,
            }
        ]
    )
    popular = pd.DataFrame(
        [
            {
                Columns.User: "mid",
                Columns.Item: "pop1",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": POPULAR_SOURCE,
            }
        ]
    )
    latest = pd.DataFrame(
        [
            {
                Columns.User: "mid",
                Columns.Item: "lat1",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": LATEST_SOURCE,
            }
        ]
    )
    weights = source_weights(5, config, latest_available=True)
    assert math.isclose(weights[PERSONALIZED_SOURCE], 0.5)
    assert math.isclose(weights[POPULAR_SOURCE], 0.25)
    assert math.isclose(weights[LATEST_SOURCE], 0.25)

    out = blend_for_users(
        personalized=personalized,
        popular=popular,
        latest=latest,
        counts={"mid": 5},
        target_users=["mid"],
        config=config,
        top_k=3,
        latest_available=True,
    )
    assert set(out[Columns.Item]) == {"p1", "pop1", "lat1"}
    # Distinct single-source rows keep their labels; no shared items → not blended.
    assert set(out["source"]) == {PERSONALIZED_SOURCE, POPULAR_SOURCE, LATEST_SOURCE}


def test_blend_shared_item_gets_blended_source():
    config = _blending(curve="linear", saturate_at=10.0, popular_share=0.5, rrf_k=1.0)
    personalized = pd.DataFrame(
        [
            {
                Columns.User: "u",
                Columns.Item: "shared",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": PERSONALIZED_SOURCE,
            }
        ]
    )
    popular = pd.DataFrame(
        [
            {
                Columns.User: "u",
                Columns.Item: "shared",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": POPULAR_SOURCE,
            }
        ]
    )
    out = blend_for_users(
        personalized=personalized,
        popular=popular,
        latest=None,
        counts={"u": 5},
        target_users=["u"],
        config=config,
        top_k=1,
        latest_available=False,
    )
    assert list(out[Columns.Item]) == ["shared"]
    assert list(out["source"]) == [BLENDED_SOURCE]


def test_append_cold_start_rows_adds_sentinel():
    config = _blending(curve="linear", saturate_at=10.0)
    popular = pd.DataFrame(
        [
            {
                Columns.User: COLD_START_USER_ID,
                Columns.Item: "i1",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": POPULAR_SOURCE,
            }
        ]
    )
    base = pd.DataFrame(
        [
            {
                Columns.User: "u1",
                Columns.Item: "i1",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": POPULAR_SOURCE,
            }
        ]
    )
    out = append_cold_start_rows(
        base,
        popular=popular,
        latest=None,
        config=config,
        top_k=1,
        latest_available=False,
    )
    assert COLD_START_USER_ID in set(out[Columns.User].astype(str))


def test_collapse_best_rank_keeps_lowest_rank():
    from cicerone.blending import collapse_best_rank

    frame = pd.DataFrame(
        [
            {Columns.User: "u1", Columns.Item: "a", Columns.Rank: 3, Columns.Score: 1.0, "source": "x"},
            {Columns.User: "u1", Columns.Item: "a", Columns.Rank: 1, Columns.Score: 0.5, "source": "y"},
            {Columns.User: "u1", Columns.Item: "b", Columns.Rank: 2, Columns.Score: 0.9, "source": "x"},
        ]
    )
    out = collapse_best_rank(frame)
    assert list(out[Columns.Item]) == ["a", "b"]
    assert list(out[Columns.Rank]) == [1, 2]


def test_interaction_counts_empty_frame():
    assert interaction_counts(pd.DataFrame()) == {}
    assert interaction_counts(pd.DataFrame({"item_id": [1]})) == {}


def test_build_latest_ranking_edge_cases():
    items = pd.DataFrame([{"item_id": "i1", "published_at": "not-a-date"}])
    empty = build_latest_ranking(items, "published_at", ["i1"], top_k=1, target_users=["u1"])
    assert empty.empty
    assert build_latest_ranking(items, "published_at", [], top_k=1, target_users=["u1"]).empty
    assert build_latest_ranking(items, "published_at", ["missing"], top_k=1, target_users=["u1"]).empty


def test_blend_for_users_empty_when_no_contributions():
    config = _blending(curve="linear", saturate_at=10.0)
    out = blend_for_users(
        personalized=pd.DataFrame(
            columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, "source"]
        ),
        popular=pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, "source"]),
        latest=None,
        counts={},
        target_users=["u1"],
        config=config,
        top_k=3,
        latest_available=False,
    )
    assert out.empty


def test_append_cold_start_rows_noop_when_no_popular():
    config = _blending()
    base = pd.DataFrame(
        [
            {
                Columns.User: "u1",
                Columns.Item: "i1",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": PERSONALIZED_SOURCE,
            }
        ]
    )
    out = append_cold_start_rows(
        base,
        popular=pd.DataFrame(),
        latest=None,
        config=config,
        top_k=1,
        latest_available=False,
    )
    assert list(out[Columns.User]) == ["u1"]


def test_train_and_recommend_with_blending_end_to_end(feature_config: FeatureConfig, sample_items):
    now = pd.Timestamp.utcnow()
    items = sample_items.copy()
    items["published_at"] = [
        now - pd.Timedelta(days=30),
        now - pd.Timedelta(days=1),
        now - pd.Timedelta(days=10),
        now - pd.Timedelta(days=5),
    ]
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i1", "event_type": "view", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "view", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i1", "event_type": "saved", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "saved", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u3", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u3", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    )
    config = FeatureConfig(
        event_weights=feature_config.event_weights,
        quantity_scaled_events=feature_config.quantity_scaled_events,
        event_caps=feature_config.event_caps,
        user_features=feature_config.user_features,
        item_features=feature_config.item_features,
        item_availability_filters=feature_config.item_availability_filters,
        blending=_blending(curve="linear", saturate_at=5.0, popular_share=0.6),
    )
    built = build_dataset(events, None, items, config, half_life_days=90)
    recs = train_and_recommend(
        built,
        target_users=["u1", "u4"],
        config=config,
        top_k=3,
        enabled_models=["collaborative", "popular"],
    )
    assert not recs.empty
    assert COLD_START_USER_ID in set(recs["user_id"].astype(str))
    cold = recs[recs["user_id"] == "u4"]
    assert not cold.empty
    # Cold user should not be dominated by personalized-only rows.
    assert set(cold["source"]) <= {POPULAR_SOURCE, LATEST_SOURCE, BLENDED_SOURCE}


def test_train_and_recommend_blending_without_date_column(feature_config: FeatureConfig, sample_items):
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i1", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i2", "event_type": "purchase", "quantity": 1, "occurred_at": now},
        ]
    )
    config = FeatureConfig(
        event_weights=feature_config.event_weights,
        quantity_scaled_events=feature_config.quantity_scaled_events,
        event_caps=feature_config.event_caps,
        user_features=[],
        item_features=feature_config.item_features,
        item_availability_filters=feature_config.item_availability_filters,
        blending=_blending(curve="linear", saturate_at=4.0),
    )
    built = build_dataset(events, None, sample_items, config, half_life_days=90)
    recs = train_and_recommend(
        built,
        target_users=["u1"],
        config=config,
        top_k=2,
        enabled_models=["collaborative"],
    )
    assert not recs.empty
    assert LATEST_SOURCE not in set(recs["source"])


def test_rank_latest_items_and_expand_latest_ranking():
    items = pd.DataFrame(
        [
            {"item_id": "old", "published_at": "2020-01-01"},
            {"item_id": "new", "published_at": "2024-06-01"},
        ]
    )
    ranked = rank_latest_items(items, "published_at", ["old", "new"], top_k=2)
    assert [item for item, _rank, _score in ranked] == ["new", "old"]
    expanded = expand_latest_ranking(ranked, ["u1", "u2"])
    assert list(expanded[Columns.User]) == ["u1", "u1", "u2", "u2"]
    assert list(expanded[Columns.Item]) == ["new", "old", "new", "old"]
    assert expand_latest_ranking(ranked, []).empty
    assert expand_latest_ranking([], ["u1"]).empty
    dup = expand_latest_ranking(ranked, ["u1", "u1"])
    assert list(dup[Columns.User]) == ["u1", "u1", "u1", "u1"]


def test_blend_for_users_shared_latest_avoids_cartesian_frame():
    config = _blending(curve="linear", saturate_at=1.0, popular_share=0.5)
    popular = pd.DataFrame(
        [
            {
                Columns.User: "u1",
                Columns.Item: "pop1",
                Columns.Rank: 1,
                Columns.Score: 1.0,
                "source": POPULAR_SOURCE,
            }
        ]
    )
    shared = [("lat1", 1, 2.0), ("lat2", 2, 1.0)]
    out = blend_for_users(
        personalized=pd.DataFrame(
            columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, "source"]
        ),
        popular=popular,
        latest=None,
        counts={"u1": 0},
        target_users=["u1"],
        config=config,
        top_k=3,
        latest_available=True,
        shared_latest=shared,
    )
    assert "lat1" in set(out[Columns.Item])


def test_blend_for_users_latest_by_user_avoids_cartesian_frame():
    config = _blending(curve="linear", saturate_at=1.0, popular_share=0.5)
    empty = pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, "source"])
    out = blend_for_users(
        personalized=empty,
        popular=empty,
        latest=None,
        counts={"u1": 0, "u2": 0},
        target_users=["u1", "u2"],
        config=config,
        top_k=2,
        latest_available=True,
        latest_by_user={
            "u1": [("a", 1, 2.0)],
            "u2": [("b", 1, 2.0)],
        },
    )
    by_user = {user: list(group[Columns.Item]) for user, group in out.groupby(Columns.User)}
    assert by_user["u1"] == ["a"]
    assert by_user["u2"] == ["b"]


def test_blend_for_users_latest_by_user_normalizes_non_string_keys():
    config = _blending(curve="linear", saturate_at=1.0, popular_share=0.5)
    empty = pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, "source"])
    out = blend_for_users(
        personalized=empty,
        popular=empty,
        latest=None,
        counts={"1": 0},
        target_users=[1],
        config=config,
        top_k=1,
        latest_available=True,
        latest_by_user={1: [("a", 1, 2.0)]},  # type: ignore[dict-item]
    )
    assert list(out[Columns.Item]) == ["a"]

    # Lookup also works when the index already uses str keys but targets are ints.
    out_str_keys = blend_for_users(
        personalized=empty,
        popular=empty,
        latest=None,
        counts={"1": 0},
        target_users=[1],
        config=config,
        top_k=1,
        latest_available=True,
        latest_by_user={"1": [("b", 1, 2.0)]},
    )
    assert list(out_str_keys[Columns.Item]) == ["b"]
