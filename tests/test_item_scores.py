from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from rectools import Columns

from cicerone.dataset import build_interactions
from cicerone.item_scores import (
    ITEM_COLUMN,
    LATEST_SCORE_COLUMN,
    N_USERS_COLUMN,
    POPULAR_SCORE_COLUMN,
    build_item_scores,
    empty_item_scores,
    normalize_item_scores,
    page_item_scores,
)
from cicerone.model_config import LATEST_WINDOW_DAYS


def test_build_item_scores_matches_interaction_sums(feature_config, sample_events, sample_items):
    interactions = build_interactions(sample_events, feature_config, half_life_days=90.0)
    scores = build_item_scores(
        sample_events,
        sample_items,
        feature_config,
        90.0,
        interactions=interactions,
    )
    by_item = interactions.groupby(Columns.Item)[Columns.Weight].sum()
    n_users = interactions.groupby(Columns.Item)[Columns.User].nunique()
    for item_id, weight in by_item.items():
        row = scores.loc[scores[ITEM_COLUMN] == item_id].iloc[0]
        assert row[POPULAR_SCORE_COLUMN] == pytest.approx(float(weight))
        assert int(row[N_USERS_COLUMN]) == int(n_users[item_id])
    assert "i4" in set(scores[ITEM_COLUMN])
    cold = scores.loc[scores[ITEM_COLUMN] == "i4"].iloc[0]
    assert float(cold[POPULAR_SCORE_COLUMN]) == 0.0
    assert float(cold[LATEST_SCORE_COLUMN]) == 0.0
    assert int(cold[N_USERS_COLUMN]) == 0


def test_build_item_scores_windowed_latest_zeros_old_events(feature_config, sample_items):
    now = datetime.now(UTC)
    old = now - timedelta(days=LATEST_WINDOW_DAYS + 2)
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": old,
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
    scores = build_item_scores(events, sample_items, feature_config, 90.0, now=now)
    latest = scores.set_index(ITEM_COLUMN)[LATEST_SCORE_COLUMN]
    assert float(latest["i1"]) == 0.0
    assert float(latest["i2"]) > 0.0
    assert float(scores.set_index(ITEM_COLUMN)[POPULAR_SCORE_COLUMN]["i1"]) > 0.0


def test_build_item_scores_empty_catalog_and_naive_now(feature_config):
    empty_interactions = pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Weight])
    assert build_item_scores(
        pd.DataFrame(), None, feature_config, 90.0, interactions=empty_interactions
    ).empty
    interactions = pd.DataFrame({Columns.Item: ["i1"], Columns.Weight: [2.0]})
    scores = build_item_scores(
        pd.DataFrame([{"user_id": "u1"}]),
        None,
        feature_config,
        90.0,
        interactions=interactions,
        now=datetime(2026, 9, 1, 12, 0, 0),
    )
    assert list(scores[ITEM_COLUMN]) == ["i1"]
    assert float(scores.iloc[0][POPULAR_SCORE_COLUMN]) == 2.0
    assert int(scores.iloc[0][N_USERS_COLUMN]) == 0


def test_item_weight_helpers_missing_columns():
    from cicerone.item_scores import _item_n_users, _item_weight_sum

    assert _item_weight_sum(pd.DataFrame()).empty
    assert _item_n_users(pd.DataFrame()).empty
    no_weight = pd.DataFrame({Columns.Item: ["i1"], Columns.User: ["u1"]})
    assert float(_item_weight_sum(no_weight)["i1"]) == 0.0
    no_user = pd.DataFrame({Columns.Item: ["i1"], Columns.Weight: [1.0]})
    assert int(_item_n_users(no_user)["i1"]) == 0


def test_normalize_item_scores_validates_and_sorts():
    frame = pd.DataFrame(
        [
            {"item_id": "b", "popular_score": "2.0", "latest_score": 1.0, "n_users": 2},
            {"item_id": "a", "popular_score": 1.0, "latest_score": 0.0, "n_users": 1},
        ]
    )
    out = normalize_item_scores(frame)
    assert list(out[ITEM_COLUMN]) == ["a", "b"]
    assert float(out.iloc[0][POPULAR_SCORE_COLUMN]) == 1.0
    assert normalize_item_scores(empty_item_scores()).empty
    with pytest.raises(ValueError, match="missing columns"):
        normalize_item_scores(pd.DataFrame())
    with pytest.raises(ValueError, match="missing columns"):
        normalize_item_scores(pd.DataFrame([{"item_id": "a", "popular_score": 1.0}]))
    with pytest.raises(ValueError, match="non-finite"):
        normalize_item_scores(
            pd.DataFrame([{"item_id": "a", "popular_score": "x", "latest_score": 0.0, "n_users": 1}])
        )
    with pytest.raises(ValueError, match="non-finite"):
        normalize_item_scores(
            pd.DataFrame([{"item_id": "a", "popular_score": float("nan"), "latest_score": 0.0, "n_users": 1}])
        )
    with pytest.raises(ValueError, match="negative"):
        normalize_item_scores(
            pd.DataFrame([{"item_id": "a", "popular_score": 1.0, "latest_score": 0.0, "n_users": -1}])
        )


def test_page_item_scores_cursor_and_single_id():
    frame = pd.DataFrame(
        [
            {"item_id": "a", "popular_score": 1.0, "latest_score": 0.0, "n_users": 1},
            {"item_id": "b", "popular_score": 2.0, "latest_score": 1.0, "n_users": 2},
            {"item_id": "c", "popular_score": 3.0, "latest_score": 0.0, "n_users": 1},
        ]
    )
    first, cursor = page_item_scores(frame, limit=2)
    assert list(first[ITEM_COLUMN]) == ["a", "b"]
    assert cursor == "b"
    second, done = page_item_scores(frame, limit=2, cursor=cursor)
    assert list(second[ITEM_COLUMN]) == ["c"]
    assert done is None
    one, none = page_item_scores(frame, limit=10, item_id="b")
    assert list(one[ITEM_COLUMN]) == ["b"]
    assert none is None
    missing, _ = page_item_scores(frame, limit=10, item_id="z")
    assert missing.empty
    empty, empty_cursor = page_item_scores(empty_item_scores(), limit=10)
    assert empty.empty
    assert empty_cursor is None
