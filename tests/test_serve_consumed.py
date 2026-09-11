from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from test_serve import _FakeReader, _feature_config, _items_df, _recs_df, _settings

from cicerone.events.consumed import ConsumedOverlay
from cicerone.serve import create_app
from cicerone.serve.consumed import consumed_item_ids, drop_consumed, merge_fill


class _History:
    def __init__(self, events: pd.DataFrame):
        self._events = events

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
        rows = self._events[self._events["user_id"] == user_id].head(limit)
        return rows.reset_index(drop=True)

    def get_user(self, user_id: str):
        del user_id
        return None


def test_drop_consumed_removes_matching_ids():
    frame = pd.DataFrame([{"item_id": "i1", "rank": 1}, {"item_id": "i2", "rank": 2}])
    out = drop_consumed(frame, {"i1"})
    assert list(out["item_id"]) == ["i2"]


def test_merge_fill_appends_unseen_items():
    primary = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"}]
    )
    filler = pd.DataFrame(
        [
            {
                "user_id": "__cold_start__",
                "item_id": "i1",
                "rank": 1,
                "score": 0.4,
                "source": "popular_fallback",
            },
            {
                "user_id": "__cold_start__",
                "item_id": "i9",
                "rank": 2,
                "score": 0.3,
                "source": "popular_fallback",
            },
        ]
    )
    merged = merge_fill(primary, filler, k=3, exclude=set())
    assert list(merged["item_id"]) == ["i1", "i9"]


def test_consumed_item_ids_unions_history_and_overlay():
    history = _History(pd.DataFrame([{"user_id": "u1", "item_id": "i1"}]))
    overlay = ConsumedOverlay()
    overlay.add("u1", "i2")
    assert consumed_item_ids("u1", history=history, overlay=overlay, lookback=10) == {"i1", "i2"}


def test_recommendations_hide_consumed_from_history():
    history = _History(pd.DataFrame([{"user_id": "u1", "item_id": "i1"}]))
    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        feature_config=_feature_config(),
        history_reader=history,
    )
    body = TestClient(app).get("/recommendations/u1", headers={"Authorization": "Bearer secret"}).json()
    assert [row["item_id"] for row in body["items"]] == ["i2"]


def test_recommendations_hide_consumed_from_overlay():
    overlay = ConsumedOverlay()
    overlay.add("u1", "i2")
    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        feature_config=_feature_config(),
        consumed=overlay,
    )
    body = TestClient(app).get("/recommendations/u1", headers={"Authorization": "Bearer secret"}).json()
    assert [row["item_id"] for row in body["items"]] == ["i1"]


def test_recommendations_exclude_consumed_query_false_keeps_history_items():
    history = _History(pd.DataFrame([{"user_id": "u1", "item_id": "i1"}]))
    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        feature_config=_feature_config(),
        history_reader=history,
    )
    body = (
        TestClient(app)
        .get(
            "/recommendations/u1?exclude_consumed=false",
            headers={"Authorization": "Bearer secret"},
        )
        .json()
    )
    assert [row["item_id"] for row in body["items"]] == ["i1", "i2"]


def test_consumed_item_ids_history_error_keeps_overlay():
    class _Boom:
        def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
            del user_id, limit
            raise RuntimeError("unavailable")

        def get_user(self, user_id: str):
            del user_id
            return None

    overlay = ConsumedOverlay()
    overlay.add("u1", "i2")
    assert consumed_item_ids("u1", history=_Boom(), overlay=overlay, lookback=10) == {"i2"}


def test_recommendations_fill_after_hide():
    recs = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"},
            {
                "user_id": "__cold_start__",
                "item_id": "i2",
                "rank": 1,
                "score": 0.4,
                "source": "popular_fallback",
            },
        ]
    )
    history = _History(pd.DataFrame([{"user_id": "u1", "item_id": "i1"}]))
    app = create_app(
        _settings(),
        _FakeReader(recs, _items_df()),
        feature_config=_feature_config(),
        history_reader=history,
    )
    body = TestClient(app).get("/recommendations/u1", headers={"Authorization": "Bearer secret"}).json()
    assert body["fallback"] is False
    assert [row["item_id"] for row in body["items"]] == ["i2"]


def test_consumed_overlay_rejects_non_positive_bounds():
    with pytest.raises(ValueError, match="max_items_per_user"):
        ConsumedOverlay(max_items_per_user=0)
    with pytest.raises(ValueError, match="max_users"):
        ConsumedOverlay(max_users=0)


def test_consumed_overlay_evicts_old_items_and_users():
    overlay = ConsumedOverlay(max_items_per_user=2, max_users=2)
    overlay.add("u1", "i1")
    overlay.add("u1", "i2")
    overlay.add("u1", "i3")
    assert overlay.item_ids("u1") == {"i2", "i3"}
    overlay.add("u2", "a")
    overlay.add("u3", "b")
    assert overlay.item_ids("u1") == set()
    assert overlay.item_ids("u2") == {"a"}
    assert overlay.item_ids("u3") == {"b"}


def test_consumed_item_ids_missing_history_is_quiet():
    class _Missing:
        def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
            del user_id, limit
            raise FileNotFoundError("events.parquet")

        def get_user(self, user_id: str):
            del user_id
            return None

    overlay = ConsumedOverlay()
    overlay.add("u1", "i2")
    assert consumed_item_ids("u1", history=_Missing(), overlay=overlay, lookback=10) == {"i2"}
