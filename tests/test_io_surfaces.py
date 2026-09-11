from __future__ import annotations

import pandas as pd

from cicerone.blending import LATEST_SOURCE, POPULAR_SOURCE
from cicerone.io.dataset_store import DatasetOutputSink
from cicerone.io.surfaces import latest_from_items, neighbors_from_events, popular_from_events
from cicerone.io.surfaces_reader import DatasetSurfacesReader, similar_as_surface


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
