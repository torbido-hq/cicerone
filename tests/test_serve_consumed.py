from __future__ import annotations

import threading

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from test_serve import _FakeReader, _feature_config, _items_df, _recs_df, _settings

from cicerone.events.consumed import ConsumedOverlay
from cicerone.io.db_store import DatabaseInputSource
from cicerone.serve import create_app
from cicerone.serve.consumed import consumed_item_ids, consumed_item_ids_for_users, drop_consumed, merge_fill


class _History:
    def __init__(self, events: pd.DataFrame):
        self._events = events

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
        rows = self._events[self._events["user_id"] == user_id].head(limit)
        return rows.reset_index(drop=True)

    def get_user(self, user_id: str):
        del user_id
        return None


class _CountingHistory(_History):
    def __init__(self, events: pd.DataFrame):
        super().__init__(events)
        self.single = 0
        self.bulk = 0

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
        self.single += 1
        return super().get_events_for_user(user_id, limit)

    def get_events_for_users(self, user_ids, limit: int) -> dict[str, pd.DataFrame]:
        self.bulk += 1
        return {str(user_id): _History.get_events_for_user(self, str(user_id), limit) for user_id in user_ids}


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


def test_consumed_item_ids_for_users_unions_history_and_overlay():
    history = _CountingHistory(
        pd.DataFrame([{"user_id": "u1", "item_id": "i1"}, {"user_id": "u2", "item_id": "i3"}])
    )
    overlay = ConsumedOverlay()
    overlay.add("u1", "i2")
    assert consumed_item_ids_for_users(["u1", "u2"], history=history, overlay=overlay, lookback=10) == {
        "u1": {"i1", "i2"},
        "u2": {"i3"},
    }
    assert history.bulk == 1
    assert history.single == 0


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


def test_recommendations_batch_uses_bulk_history():
    history = _CountingHistory(pd.DataFrame([{"user_id": "u1", "item_id": "i1"}]))
    recs = pd.concat(
        [
            _recs_df(),
            pd.DataFrame(
                [{"user_id": "u2", "item_id": "i2", "rank": 1, "score": 0.8, "source": "personalized"}]
            ),
        ],
        ignore_index=True,
    )
    app = create_app(
        _settings(),
        _FakeReader(recs, _items_df()),
        feature_config=_feature_config(),
        history_reader=history,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    single = client.get("/recommendations/u1", headers=headers)
    assert single.status_code == 200
    assert history.single == 1
    assert history.bulk == 0
    batch = client.post("/recommendations/batch", json={"user_ids": ["u1", "u2"]}, headers=headers)
    assert batch.status_code == 200
    assert [row["item_id"] for row in batch.json()["users"][0]["items"]] == ["i2"]
    assert history.bulk == 1
    assert history.single == 1


def test_recommendations_hide_consumed_from_memory_sqlite_history():
    source = DatabaseInputSource({"database_url": "sqlite+pysqlite://"})
    with source._engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE events ("
                "user_id TEXT, item_id TEXT, event_type TEXT, quantity INTEGER, occurred_at TEXT)"
            )
        )
        conn.execute(text("INSERT INTO events VALUES ('u1', 'i1', 'view', 1, '2026-08-21')"))
    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        feature_config=_feature_config(),
        history_reader=source,
    )
    body = TestClient(app).get("/recommendations/u1", headers={"Authorization": "Bearer secret"}).json()
    assert [row["item_id"] for row in body["items"]] == ["i2"]


def test_sqlite_get_events_for_users_one_query(monkeypatch):
    source = DatabaseInputSource({"database_url": "sqlite+pysqlite://"})
    with source._engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE events ("
                "user_id TEXT, item_id TEXT, event_type TEXT, quantity INTEGER, occurred_at TEXT)"
            )
        )
        conn.execute(text("INSERT INTO events VALUES ('u1', 'i1', 'view', 1, '2026-08-21')"))
        conn.execute(text("INSERT INTO events VALUES ('u2', 'i3', 'view', 1, '2026-08-22')"))
        conn.execute(text("INSERT INTO events VALUES ('u1', 'i9', 'view', 1, '2026-08-20')"))
    real_read = pd.read_sql
    loaded: list[int] = []

    def counting_read(sql, *args, **kwargs):
        frame = real_read(sql, *args, **kwargs)
        if "IN" in str(sql).upper():
            loaded.append(len(frame))
        return frame

    monkeypatch.setattr(pd, "read_sql", counting_read)
    frames = source.get_events_for_users(["u1", "u2"], limit=1)
    assert list(frames["u1"]["item_id"]) == ["i1"]
    assert list(frames["u2"]["item_id"]) == ["i3"]
    assert len(loaded) == 1


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


def test_consumed_item_ids_for_users_isolates_single_user_errors():
    class _Partial(_History):
        def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
            if user_id == "u2":
                raise RuntimeError("unavailable")
            return super().get_events_for_user(user_id, limit)

    history = _Partial(pd.DataFrame([{"user_id": "u1", "item_id": "i1"}, {"user_id": "u2", "item_id": "i3"}]))
    overlay = ConsumedOverlay()
    overlay.add("u2", "i9")
    assert consumed_item_ids_for_users(["u1", "u2"], history=history, overlay=overlay, lookback=10) == {
        "u1": {"i1"},
        "u2": {"i9"},
    }


def test_consumed_item_ids_for_users_missing_bulk_does_not_retry_per_user():
    class _MissingBulk(_CountingHistory):
        def get_events_for_users(self, user_ids, limit: int) -> dict[str, pd.DataFrame]:
            self.bulk += 1
            del user_ids, limit
            raise FileNotFoundError("events.parquet")

    history = _MissingBulk(pd.DataFrame([{"user_id": "u1", "item_id": "i1"}]))
    overlay = ConsumedOverlay()
    overlay.add("u1", "i2")
    assert consumed_item_ids_for_users(["u1", "u2"], history=history, overlay=overlay, lookback=10) == {
        "u1": {"i2"},
        "u2": set(),
    }
    assert history.bulk == 1
    assert history.single == 0


def test_consumed_item_ids_for_users_missing_single_stops_remaining_reads():
    class _MissingAfterFirst(_History):
        def __init__(self, events: pd.DataFrame):
            super().__init__(events)
            self.single = 0

        def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
            self.single += 1
            if user_id == "u2":
                raise FileNotFoundError("events.parquet")
            return super().get_events_for_user(user_id, limit)

    history = _MissingAfterFirst(
        pd.DataFrame([{"user_id": "u1", "item_id": "i1"}, {"user_id": "u2", "item_id": "i3"}])
    )
    overlay = ConsumedOverlay()
    overlay.add("u3", "i9")
    assert consumed_item_ids_for_users(["u1", "u2", "u3"], history=history, overlay=overlay, lookback=10) == {
        "u1": {"i1"},
        "u2": set(),
        "u3": {"i9"},
    }
    assert history.single == 2


def test_consumed_item_ids_for_users_bulk_error_falls_back_per_user():
    class _BoomBulk(_CountingHistory):
        def get_events_for_users(self, user_ids, limit: int) -> dict[str, pd.DataFrame]:
            self.bulk += 1
            del user_ids, limit
            raise RuntimeError("bulk down")

    history = _BoomBulk(
        pd.DataFrame([{"user_id": "u1", "item_id": "i1"}, {"user_id": "u2", "item_id": "i3"}])
    )
    overlay = ConsumedOverlay()
    overlay.add("u1", "i2")
    assert consumed_item_ids_for_users(["u1", "u2"], history=history, overlay=overlay, lookback=10) == {
        "u1": {"i1", "i2"},
        "u2": {"i3"},
    }
    assert history.bulk == 1
    assert history.single == 2


def test_sqlite_get_events_for_users_bounds_fallback_without_occurred_at(monkeypatch):
    source = DatabaseInputSource({"database_url": "sqlite+pysqlite://"})
    with source._engine.begin() as conn:
        conn.execute(text("CREATE TABLE events (user_id TEXT, item_id TEXT)"))
        for index in range(20):
            conn.execute(text("INSERT INTO events VALUES ('u1', :item)"), {"item": f"i{index}"})
            conn.execute(text("INSERT INTO events VALUES ('u2', :item)"), {"item": f"j{index}"})
    real_read = pd.read_sql
    loaded: list[int] = []

    def counting_read(sql, *args, **kwargs):
        frame = real_read(sql, *args, **kwargs)
        loaded.append(len(frame))
        return frame

    monkeypatch.setattr(pd, "read_sql", counting_read)
    frames = source.get_events_for_users(["u1", "u2"], limit=1)
    assert len(frames["u1"]) == 1
    assert len(frames["u2"]) == 1
    assert loaded
    assert max(loaded) <= 16


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


def test_recommendations_category_only_does_not_fill():
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
    app = create_app(
        _settings(),
        _FakeReader(recs, _items_df()),
        feature_config=_feature_config(),
    )
    body = (
        TestClient(app)
        .get(
            "/recommendations/u1?category=wine",
            headers={"Authorization": "Bearer secret"},
        )
        .json()
    )
    assert body["items"] == []


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


def test_consumed_overlay_discard_pair_and_user():
    overlay = ConsumedOverlay()
    overlay.add("u1", "i1")
    overlay.add("u1", "i2")
    overlay.discard("u1", "i1")
    assert overlay.item_ids("u1") == {"i2"}
    overlay.discard("u1")
    assert overlay.item_ids("u1") == set()


def test_consumed_overlay_discard_item_all_users():
    overlay = ConsumedOverlay()
    overlay.add("u1", "i1")
    overlay.add("u1", "i2")
    overlay.add("u2", "i1")
    overlay.discard_item("i1")
    assert overlay.item_ids("u1") == {"i2"}
    assert overlay.item_ids("u2") == set()


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


def test_consumed_overlay_replace_pairs_is_atomic():
    overlay = ConsumedOverlay()
    overlay.add("u1", "i1")
    overlay.replace_pairs([("u1", "i1")], [("u1", "i2")])
    assert overlay.item_ids("u1") == {"i2"}


def test_consumed_overlay_mutation_serializes_store_then_reconcile():
    overlay = ConsumedOverlay()
    order: list[str] = []
    started = threading.Event()
    release = threading.Event()

    def first() -> None:
        with overlay.mutation():
            order.append("a-store")
            started.set()
            assert release.wait(2)
            overlay.replace_pairs([], [("u1", "i1")])
            order.append("a-overlay")

    def second() -> None:
        assert started.wait(2)
        with overlay.mutation():
            order.append("b-store")
            overlay.replace_pairs([("u1", "i1")], [("u1", "i2")])
            order.append("b-overlay")

    workers = [threading.Thread(target=first), threading.Thread(target=second)]
    for worker in workers:
        worker.start()
    release.set()
    for worker in workers:
        worker.join(2)
    assert order == ["a-store", "a-overlay", "b-store", "b-overlay"]
    assert overlay.item_ids("u1") == {"i2"}
