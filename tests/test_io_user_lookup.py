from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

from cicerone.io.db_store import DatabaseInputSource
from cicerone.io.user_lookup import filter_rows_for_user, newest_events


def test_filter_rows_for_user_stringifies_ids():
    frame = pd.DataFrame([{"user_id": 1, "item_id": "i1"}, {"user_id": 2, "item_id": "i2"}])
    matched = filter_rows_for_user(frame, "1")
    assert list(matched["item_id"]) == ["i1"]


def test_filter_rows_for_user_empty_without_column():
    matched = filter_rows_for_user(pd.DataFrame({"item_id": ["i1"]}), "u1")
    assert matched.empty
    assert list(matched.columns) == ["item_id"]


def test_filter_rows_for_user_preserves_columns_when_empty():
    frame = pd.DataFrame({"user_id": ["u1"], "item_id": ["i1"]}).iloc[0:0]
    matched = filter_rows_for_user(frame, "ghost")
    assert list(matched.columns) == ["user_id", "item_id"]
    assert matched.empty


def test_newest_events_empty_preserves_columns():
    frame = pd.DataFrame({"item_id": pd.Series(dtype=str), "occurred_at": pd.Series(dtype=str)})
    out = newest_events(frame, 5)
    assert list(out.columns) == ["item_id", "occurred_at"]
    assert out.empty


def test_newest_events_sorts_and_limits():
    frame = pd.DataFrame(
        [
            {"item_id": "old", "occurred_at": "2026-08-01T00:00:00Z"},
            {"item_id": "new", "occurred_at": "2026-08-21T00:00:00Z"},
            {"item_id": "mid", "occurred_at": "2026-08-10T00:00:00Z"},
        ]
    )
    assert list(newest_events(frame, 2)["item_id"]) == ["new", "mid"]


def test_newest_events_without_timestamp_keeps_order():
    frame = pd.DataFrame([{"item_id": "a"}, {"item_id": "b"}, {"item_id": "c"}])
    assert list(newest_events(frame, 2)["item_id"]) == ["a", "b"]


def test_newest_events_does_not_drop_existing_sort_column():
    frame = pd.DataFrame(
        [
            {"item_id": "old", "occurred_at": "2026-08-01T00:00:00Z", "_sort": "keep"},
            {"item_id": "new", "occurred_at": "2026-08-21T00:00:00Z", "_sort": "keep"},
        ]
    )
    out = newest_events(frame, 1)
    assert list(out["item_id"]) == ["new"]
    assert list(out["_sort"]) == ["keep"]


def test_sqlite_get_events_for_user_and_get_user(tmp_path):
    url = f"sqlite:///{tmp_path / 'input.db'}"
    engine = create_engine(url)
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "old", "event_type": "view", "occurred_at": "2026-08-01T00:00:00Z"},
            {
                "user_id": "u1",
                "item_id": "new",
                "event_type": "purchase",
                "occurred_at": "2026-08-21T00:00:00Z",
            },
            {
                "user_id": "u2",
                "item_id": "other",
                "event_type": "view",
                "occurred_at": "2026-08-21T00:00:00Z",
            },
        ]
    ).to_sql("events", engine, index=False)
    pd.DataFrame([{"user_id": "u1", "region_slug": "lazio"}]).to_sql("users", engine, index=False)

    source = DatabaseInputSource({"database_url": url})
    rows = source.get_events_for_user("u1", limit=1)
    assert list(rows["item_id"]) == ["new"]
    assert source.get_user("u1")["region_slug"] == "lazio"
    assert source.get_user("ghost") is None
    engine.dispose()


def test_sqlite_get_events_for_user_utc_order_beats_lexical_limit(tmp_path):
    url = f"sqlite:///{tmp_path / 'input.db'}"
    engine = create_engine(url)
    pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "lexical_new",
                "event_type": "view",
                "occurred_at": "2026-08-21T00:00:00+02:00",
            },
            {
                "user_id": "u1",
                "item_id": "utc_new",
                "event_type": "view",
                "occurred_at": "2026-08-20T23:30:00+00:00",
            },
        ]
    ).to_sql("events", engine, index=False)

    source = DatabaseInputSource({"database_url": url})
    rows = source.get_events_for_user("u1", limit=1)
    assert list(rows["item_id"]) == ["utc_new"]
    engine.dispose()


def test_sqlite_custom_query_trailing_semicolon(tmp_path):
    url = f"sqlite:///{tmp_path / 'input.db'}"
    engine = create_engine(url)
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "view", "occurred_at": "2026-08-21T00:00:00Z"}]
    ).to_sql("events", engine, index=False)
    pd.DataFrame([{"user_id": "u1", "region_slug": "lazio"}]).to_sql("users", engine, index=False)

    source = DatabaseInputSource(
        {
            "database_url": url,
            "events_query": 'SELECT * FROM "events";',
            "users_query": 'SELECT * FROM "users" ; ',
        }
    )

    assert list(source.get_events_for_user("u1", limit=10)["item_id"]) == ["i1"]
    assert source.get_user("u1")["region_slug"] == "lazio"
    engine.dispose()
