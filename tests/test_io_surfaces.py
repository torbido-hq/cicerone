from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy.exc import OperationalError

from cicerone.blending import LATEST_SOURCE, POPULAR_SOURCE
from cicerone.io.dataset_store import DatasetOutputSink
from cicerone.io.surfaces import (
    SURFACE_FILES,
    latest_from_items,
    neighbors_from_events,
    popular_from_events,
    surfaces_stamp_payload,
)
from cicerone.io.surfaces_reader import DatasetSurfacesReader, DbSurfacesReader, similar_as_surface


def test_popular_from_events_breaks_user_ties_by_event_count():
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "a"},
            {"user_id": "u2", "item_id": "a"},
            {"user_id": "u1", "item_id": "a"},
            {"user_id": "u1", "item_id": "b"},
            {"user_id": "u2", "item_id": "b"},
        ]
    )
    frame = popular_from_events(events, 2)
    assert list(frame["item_id"]) == ["a", "b"]
    assert list(frame["score"]) == [2, 2]


def test_popular_from_events_ranks_by_distinct_users():
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "a"},
            {"user_id": "u2", "item_id": "a"},
            {"user_id": "u1", "item_id": "b"},
            {"user_id": "u1", "item_id": "a"},
        ]
    )
    frame = popular_from_events(events, 2)
    assert list(frame["item_id"]) == ["a", "b"]
    assert frame.iloc[0]["source"] == POPULAR_SOURCE
    assert frame.iloc[0]["score"] == 2


def test_latest_from_items_uses_published_at():
    items = pd.DataFrame(
        [
            {"item_id": "old", "published_at": "2026-01-01T00:00:00Z"},
            {"item_id": "new", "published_at": "2026-09-01T00:00:00Z"},
        ]
    )
    frame = latest_from_items(items, 2)
    assert list(frame["item_id"]) == ["new", "old"]
    assert frame.iloc[0]["source"] == LATEST_SOURCE


def test_neighbors_from_events_caps_per_user_history():
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": f"i{n}", "occurred_at": f"2026-09-01T00:{n:02d}:00Z"}
            for n in range(30)
        ]
        + [{"user_id": "u2", "item_id": "i0"}, {"user_id": "u2", "item_id": "i1"}]
    )
    frame = neighbors_from_events(events, 3, history_cap=4)
    assert set(frame["item_id"]) <= {"i0", "i1", "i26", "i27", "i28", "i29"}
    assert frame.groupby("item_id").size().max() <= 3


def test_neighbors_from_events_scores_shared_users():
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "a"},
            {"user_id": "u1", "item_id": "b"},
            {"user_id": "u2", "item_id": "a"},
            {"user_id": "u2", "item_id": "b"},
            {"user_id": "u2", "item_id": "c"},
            {"user_id": "u3", "item_id": "c"},
        ]
    )
    frame = neighbors_from_events(events, 2)
    a_neighbors = frame.loc[frame["item_id"] == "a"].sort_values("rank")
    assert list(a_neighbors["neighbor_id"])[0] == "b"


def test_surface_builders_drop_blank_ids():
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "a"},
            {"user_id": None, "item_id": "b"},
            {"user_id": "u2", "item_id": float("nan")},
            {"user_id": "  ", "item_id": "c"},
            {"user_id": "u3", "item_id": "  "},
        ]
    )
    popular = popular_from_events(events, 5)
    assert list(popular["item_id"]) == ["a"]
    neighbors = neighbors_from_events(events, 3)
    assert neighbors.empty
    latest = latest_from_items(
        pd.DataFrame(
            [
                {"item_id": None, "published_at": "2026-09-01T00:00:00Z"},
                {"item_id": "  ", "published_at": "2026-09-02T00:00:00Z"},
                {"item_id": "keep", "published_at": "2026-09-03T00:00:00Z"},
            ]
        ),
        5,
    )
    assert list(latest["item_id"]) == ["keep"]


def test_surface_builders_empty_inputs():
    empty = pd.DataFrame()
    assert popular_from_events(empty, 5).empty
    assert latest_from_items(empty, 5).empty
    assert neighbors_from_events(empty, 5).empty
    assert popular_from_events(pd.DataFrame([{"item_id": "a"}]), 0).empty
    assert latest_from_items(pd.DataFrame([{"item_id": "a"}]), 5).empty
    assert neighbors_from_events(pd.DataFrame([{"item_id": "a"}]), 5).empty


def test_dataset_surfaces_reader_round_trip(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    sink = DatasetOutputSink(options)
    sink.write_surfaces(
        popular=pd.DataFrame([{"item_id": "i1", "rank": 1, "score": 3.0, "source": "popular_fallback"}]),
        latest=pd.DataFrame([{"item_id": "i2", "rank": 1, "score": 2.0, "source": "latest"}]),
        neighbors=pd.DataFrame([{"item_id": "i1", "neighbor_id": "i2", "rank": 1, "score": 0.9}]),
    )
    reader = DatasetSurfacesReader(options)
    assert list(reader.get_popular(1)["item_id"]) == ["i1"]
    assert list(reader.get_latest(1)["item_id"]) == ["i2"]
    similar = similar_as_surface(reader.get_similar("i1", 5))
    assert list(similar["item_id"]) == ["i2"]
    assert reader.get_similar("missing", 5).empty
    reader.refresh()
    assert list(reader.get_popular(1)["item_id"]) == ["i1"]
    assert (tmp_path / "surfaces_stamp.json").is_file()
    many = reader.get_similar_many(["i1", "missing"], 5)
    assert list(similar_as_surface(many["i1"])["item_id"]) == ["i2"]
    assert many["missing"].empty


def test_db_surfaces_reader_loads_session_neighbors_in_one_query(monkeypatch):
    reader = DbSurfacesReader({"database_url": "sqlite://"})
    calls: list[object] = []

    def one_shot(*_args, **_kwargs):
        calls.append(_kwargs.get("params"))
        return pd.DataFrame(
            [
                {"item_id": "i1", "neighbor_id": "a", "rank": 1, "score": 0.9},
                {"item_id": "i2", "neighbor_id": "b", "rank": 1, "score": 0.8},
                {"item_id": "i1", "neighbor_id": "c", "rank": 2, "score": 0.1},
            ]
        )

    monkeypatch.setattr(pd, "read_sql", one_shot)
    many = reader.get_similar_many(["i1", "i2"], 1)
    assert len(calls) == 1
    assert list(many["i1"]["neighbor_id"]) == ["a"]
    assert list(many["i2"]["neighbor_id"]) == ["b"]


def test_dataset_surfaces_reader_keeps_cache_when_stamp_mismatches(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    sink = DatasetOutputSink(options)
    sink.write_surfaces(
        popular=pd.DataFrame([{"item_id": "i1", "rank": 1, "score": 3.0, "source": "popular_fallback"}]),
        latest=pd.DataFrame([{"item_id": "i2", "rank": 1, "score": 2.0, "source": "latest"}]),
        neighbors=pd.DataFrame([{"item_id": "i1", "neighbor_id": "i2", "rank": 1, "score": 0.9}]),
    )
    reader = DatasetSurfacesReader(options)
    sink.write_surfaces(
        popular=pd.DataFrame([{"item_id": "i9", "rank": 1, "score": 9.0, "source": "popular_fallback"}]),
        latest=pd.DataFrame([{"item_id": "i8", "rank": 1, "score": 1.0, "source": "latest"}]),
        neighbors=pd.DataFrame([{"item_id": "i9", "neighbor_id": "i8", "rank": 1, "score": 0.2}]),
    )
    (tmp_path / "popular.parquet").write_bytes((tmp_path / "latest.parquet").read_bytes())
    reader.refresh()
    assert list(reader.get_popular(1)["item_id"]) == ["i1"]
    assert list(reader.get_latest(1)["item_id"]) == ["i2"]


def test_dataset_surfaces_reader_keeps_cache_when_parquet_is_unreadable(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    sink = DatasetOutputSink(options)
    sink.write_surfaces(
        popular=pd.DataFrame([{"item_id": "i1", "rank": 1, "score": 3.0, "source": "popular_fallback"}]),
        latest=pd.DataFrame([{"item_id": "i2", "rank": 1, "score": 2.0, "source": "latest"}]),
        neighbors=pd.DataFrame([{"item_id": "i1", "neighbor_id": "i2", "rank": 1, "score": 0.9}]),
    )
    reader = DatasetSurfacesReader(options)
    (tmp_path / "popular.parquet").write_bytes(b"not-parquet")
    files = [(name, (tmp_path / name).read_bytes()) for name in SURFACE_FILES]
    (tmp_path / "surfaces_stamp.json").write_bytes(surfaces_stamp_payload(files))
    reader.refresh()
    assert list(reader.get_popular(1)["item_id"]) == ["i1"]
    assert list(reader.get_latest(1)["item_id"]) == ["i2"]


def test_dataset_surfaces_reader_keeps_legacy_trio_when_later_files_diverge(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    pd.DataFrame([{"item_id": "i1", "rank": 1, "score": 1.0, "source": "popular_fallback"}]).to_parquet(
        tmp_path / "popular.parquet", index=False
    )
    pd.DataFrame([{"item_id": "i2", "rank": 1, "score": 2.0, "source": "latest"}]).to_parquet(
        tmp_path / "latest.parquet", index=False
    )
    pd.DataFrame([{"item_id": "i1", "neighbor_id": "i2", "rank": 1, "score": 0.9}]).to_parquet(
        tmp_path / "item_neighbors.parquet", index=False
    )
    reader = DatasetSurfacesReader(options)
    assert list(reader.get_popular(1)["item_id"]) == ["i1"]
    pd.DataFrame([{"item_id": "i9", "rank": 1, "score": 9.0, "source": "popular_fallback"}]).to_parquet(
        tmp_path / "popular.parquet", index=False
    )
    reader.refresh()
    assert list(reader.get_popular(1)["item_id"]) == ["i1"]
    assert list(reader.get_latest(1)["item_id"]) == ["i2"]


def test_dataset_surfaces_reader_ignores_partial_legacy_files(tmp_path):
    options = {"storage_backend": "local", "path": str(tmp_path)}
    pd.DataFrame([{"item_id": "i1", "rank": 1, "score": 1.0, "source": "popular_fallback"}]).to_parquet(
        tmp_path / "popular.parquet", index=False
    )
    reader = DatasetSurfacesReader(options)
    assert reader.get_popular(1).empty


def test_db_surfaces_reader_keeps_missing_table_empty(monkeypatch):
    reader = DbSurfacesReader({"database_url": "sqlite://"})

    def missing(*_args, **_kwargs):
        raise OperationalError("SELECT", {}, Exception("no such table: recommendation_popular"))

    monkeypatch.setattr(pd, "read_sql", missing)
    assert reader.get_popular(5).empty
    assert reader.get_similar("i1", 5).empty


def test_db_surfaces_reader_reraises_connectivity_errors(monkeypatch):
    reader = DbSurfacesReader({"database_url": "sqlite://"})

    def refused(*_args, **_kwargs):
        raise OperationalError("SELECT", {}, Exception("connection refused"))

    monkeypatch.setattr(pd, "read_sql", refused)
    with pytest.raises(OperationalError):
        reader.get_popular(5)
    with pytest.raises(OperationalError):
        reader.get_similar("i1", 5)
