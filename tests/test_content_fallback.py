"""Unit tests for content-based cold-item fallback."""

from __future__ import annotations

import pandas as pd
import pytest
from rectools import Columns

from cicerone.content_fallback import ContentFallbackModel, build_content_fallback_model
from cicerone.feature_config import FeatureColumn


class _DummyDataset:
    pass


def test_feature_dict_handles_list_features_and_nans():
    items = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "category": "beer",
                "tags": ["ipa", "hoppy", float("nan"), pd.NA],
                "region_slug": None,
            },
            {
                "item_id": "i2",
                "category": float("nan"),
                "tags": None,
                "region_slug": "lazio",
            },
            {
                "item_id": "i3",
                "category": pd.NA,
                "tags": pd.NA,
                "region_slug": pd.NA,
            },
        ]
    )
    interactions = pd.DataFrame(
        {
            Columns.User: ["u1"],
            Columns.Item: ["i1"],
            Columns.Weight: [1.0],
            Columns.Datetime: [pd.Timestamp.utcnow()],
        }
    )
    model = ContentFallbackModel(
        feature_columns=[
            FeatureColumn(column="category", type="categorical"),
            FeatureColumn(column="tags", type="list"),
            FeatureColumn(column="region_slug", type="categorical"),
            FeatureColumn(column="missing_col", type="categorical"),
        ],
        items=items,
        interactions=interactions,
    )
    model.fit(_DummyDataset())
    assert "i1" in model._item_ids
    assert "i2" in model._item_ids
    # Null-only features → no tokens → skipped from matrix.
    assert "i3" not in model._item_ids
    # No "<NA>" / "nan" token pollution in the fitted vocabulary.
    assert model._vectorizer is not None
    feature_names = set(model._vectorizer.get_feature_names_out())
    assert not any("NA" in name or "nan" in name.lower() for name in feature_names)


def test_fit_with_no_items_or_features_is_noop():
    model = ContentFallbackModel(feature_columns=[], items=None, interactions=None)
    model.fit(_DummyDataset())
    recs = model.recommend(
        users=["u1"], dataset=_DummyDataset(), k=5, filter_viewed=True, items_to_recommend=None
    )
    assert recs.empty


def test_fit_skips_items_without_any_feature_tokens():
    items = pd.DataFrame([{"item_id": "i1", "category": None}])
    model = ContentFallbackModel(
        feature_columns=[FeatureColumn(column="category", type="categorical")],
        items=items,
        interactions=pd.DataFrame(),
    )
    model.fit(_DummyDataset())
    assert model._item_ids == []
    assert model.recommend(users=["u1"], dataset=_DummyDataset(), k=3, filter_viewed=True).empty


def test_recommend_skips_users_without_history_or_unmapped_history():
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer"},
            {"item_id": "i_new", "category": "beer"},
        ]
    )
    interactions = pd.DataFrame(
        {
            Columns.User: ["u1"],
            Columns.Item: ["i1"],
            Columns.Weight: [1.0],
            Columns.Datetime: [pd.Timestamp.utcnow()],
        }
    )
    model = ContentFallbackModel(
        feature_columns=[FeatureColumn(column="category", type="categorical")],
        items=items,
        interactions=interactions,
    )
    model.fit(_DummyDataset())
    # u2: no history; u3: history item unknown to the feature matrix.
    model._user_history["u3"] = ["ghost_item"]
    recs = model.recommend(
        users=["u2", "u3"],
        dataset=_DummyDataset(),
        k=5,
        filter_viewed=True,
        items_to_recommend=["i_new"],
    )
    assert recs.empty


def test_recommend_empty_when_allowlist_excludes_all_cold_items():
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer"},
            {"item_id": "i_new", "category": "beer"},
        ]
    )
    interactions = pd.DataFrame(
        {
            Columns.User: ["u1"],
            Columns.Item: ["i1"],
            Columns.Weight: [1.0],
            Columns.Datetime: [pd.Timestamp.utcnow()],
        }
    )
    model = build_content_fallback_model(
        feature_columns=[FeatureColumn(column="category", type="categorical")],
        max_neighbors=10,
        items=items,
        interactions=interactions,
    )
    model.fit(_DummyDataset())
    recs = model.recommend(
        users=["u1"],
        dataset=_DummyDataset(),
        k=5,
        filter_viewed=True,
        items_to_recommend=["i1"],  # warm only — no cold items allowed
    )
    assert recs.empty


def test_fit_releases_source_frames_after_building_indexes():
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer"},
            {"item_id": "i_new", "category": "beer"},
        ]
    )
    interactions = pd.DataFrame(
        {
            Columns.User: ["u1"],
            Columns.Item: ["i1"],
            Columns.Weight: [1.0],
            Columns.Datetime: [pd.Timestamp.utcnow()],
        }
    )
    model = build_content_fallback_model(
        feature_columns=[FeatureColumn(column="category", type="categorical")],
        max_neighbors=10,
        items=items,
        interactions=interactions,
    )
    model.fit(_DummyDataset())
    assert model.items is None
    assert model.interactions is None
    assert model._item_index
    assert model._cold_ids
    recs = model.recommend(users=["u1"], dataset=_DummyDataset(), k=5, filter_viewed=True)
    assert not recs.empty


def test_recommend_respects_max_neighbors_cap():
    """max_neighbors bounds recommendations even when a larger k is requested."""
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer"},
            {"item_id": "i2", "category": "beer"},
            {"item_id": "i3", "category": "beer"},
        ]
    )
    interactions = pd.DataFrame(
        {
            Columns.User: ["u1"],
            Columns.Item: ["i1"],
            Columns.Weight: [1.0],
            Columns.Datetime: [pd.Timestamp.utcnow()],
        }
    )
    model = build_content_fallback_model(
        feature_columns=[FeatureColumn(column="category", type="categorical")],
        max_neighbors=1,
        items=items,
        interactions=interactions,
    )
    model.fit(_DummyDataset())
    recs = model.recommend(users=["u1"], dataset=_DummyDataset(), k=5, filter_viewed=True)
    assert len(recs) <= 1
    assert not recs.empty
    assert set(recs[Columns.Item]) <= {"i2", "i3"}


def test_fit_raises_clear_error_when_interactions_missing_id_columns():
    items = pd.DataFrame([{"item_id": "i1", "category": "beer"}])
    interactions = pd.DataFrame({"user": ["u1"], "product": ["i1"]})
    model = ContentFallbackModel(
        feature_columns=[FeatureColumn(column="category", type="categorical")],
        items=items,
        interactions=interactions,
    )
    with pytest.raises(ValueError, match="interactions is missing a required id column"):
        model.fit(_DummyDataset())


def test_fit_raises_clear_error_when_items_missing_id_column():
    items = pd.DataFrame([{"sku": "i1", "category": "beer"}])
    interactions = pd.DataFrame(
        {
            Columns.User: ["u1"],
            Columns.Item: ["i1"],
            Columns.Weight: [1.0],
            Columns.Datetime: [pd.Timestamp.utcnow()],
        }
    )
    model = ContentFallbackModel(
        feature_columns=[FeatureColumn(column="category", type="categorical")],
        items=items,
        interactions=interactions,
    )
    with pytest.raises(ValueError, match="items is missing a required id column"):
        model.fit(_DummyDataset())


def test_feature_dict_parses_list_like_strings():
    from cicerone.content_fallback import _feature_dict

    row = pd.Series(
        {
            "item_id": "i1",
            "tags": "ipa, stout",
            "styles": '["lager", "pils"]',
            "solo": "single",
        }
    )
    tokens = _feature_dict(
        row,
        [
            FeatureColumn(column="tags", type="list"),
            FeatureColumn(column="styles", type="list"),
            FeatureColumn(column="solo", type="list"),
        ],
    )
    assert tokens["tags=ipa"] == 1.0
    assert tokens["tags=stout"] == 1.0
    assert tokens["styles=lager"] == 1.0
    assert tokens["styles=pils"] == 1.0
    assert tokens["solo=single"] == 1.0


def test_recommend_thread_pool_path_returns_rows_for_each_user(monkeypatch):
    monkeypatch.setattr("cicerone.content_fallback._RECOMMEND_THREAD_MIN_USERS", 1)
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer"},
            {"item_id": "i_new", "category": "beer"},
        ]
    )
    interactions = pd.DataFrame(
        {
            Columns.User: ["u1", "u2"],
            Columns.Item: ["i1", "i1"],
            Columns.Weight: [1.0, 1.0],
            Columns.Datetime: [pd.Timestamp.utcnow(), pd.Timestamp.utcnow()],
        }
    )
    model = ContentFallbackModel(
        feature_columns=[FeatureColumn(column="category", type="categorical")],
        items=items,
        interactions=interactions,
    )
    model.fit(_DummyDataset())
    recs = model.recommend(
        users=["u1", "u2"],
        dataset=_DummyDataset(),
        k=5,
        filter_viewed=True,
    )
    assert set(recs[Columns.User].astype(str)) == {"u1", "u2"}


def test_recommend_thread_pool_matches_serial_user_rank_order(monkeypatch):
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer"},
            {"item_id": "i_new", "category": "beer"},
        ]
    )
    interactions = pd.DataFrame(
        {
            Columns.User: ["u1", "u2"],
            Columns.Item: ["i1", "i1"],
            Columns.Weight: [1.0, 1.0],
            Columns.Datetime: [pd.Timestamp.utcnow(), pd.Timestamp.utcnow()],
        }
    )
    model = ContentFallbackModel(
        feature_columns=[FeatureColumn(column="category", type="categorical")],
        items=items,
        interactions=interactions,
    )
    model.fit(_DummyDataset())
    monkeypatch.setattr("cicerone.content_fallback._RECOMMEND_THREAD_MIN_USERS", 100)
    serial = model.recommend(users=["u2", "u1"], dataset=_DummyDataset(), k=5, filter_viewed=True)
    monkeypatch.setattr("cicerone.content_fallback._RECOMMEND_THREAD_MIN_USERS", 1)
    threaded = model.recommend(users=["u2", "u1"], dataset=_DummyDataset(), k=5, filter_viewed=True)
    pd.testing.assert_frame_equal(serial.reset_index(drop=True), threaded.reset_index(drop=True))
    assert list(serial[Columns.User].astype(str).drop_duplicates()) == ["u1", "u2"]


def test_recommend_handles_mixed_non_string_ids(monkeypatch):
    monkeypatch.setattr("cicerone.content_fallback._RECOMMEND_THREAD_MIN_USERS", 1)
    items = pd.DataFrame(
        [
            {"item_id": 1, "category": "beer"},
            {"item_id": "2", "category": "beer"},
            {"item_id": 3, "category": "beer"},
        ]
    )
    interactions = pd.DataFrame(
        {
            Columns.User: ["u1", "u1", "u2"],
            Columns.Item: [1, "2", 1],
            Columns.Weight: [1.0, 1.0, 1.0],
            Columns.Datetime: [pd.Timestamp.utcnow()] * 3,
        }
    )
    model = ContentFallbackModel(
        feature_columns=[FeatureColumn(column="category", type="categorical")],
        items=items,
        interactions=interactions,
    )
    model.fit(_DummyDataset())
    recs = model.recommend(
        users=["u1", "u2"],
        dataset=_DummyDataset(),
        k=5,
        filter_viewed=True,
        items_to_recommend=[1, "2", 3],
    )
    assert set(recs[Columns.User].astype(str)) == {"u1", "u2"}
    assert set(map(str, recs[Columns.Item])) <= {"1", "2", "3"}


def test_recommend_single_user_path_does_not_use_thread_pool(monkeypatch):
    class _ThreadPoolSpy:
        def __init__(self, *_, **__):
            raise AssertionError("ThreadPoolExecutor should not be used for single-user path")

    monkeypatch.setattr("cicerone.content_fallback.ThreadPoolExecutor", _ThreadPoolSpy)
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer"},
            {"item_id": "i_new", "category": "beer"},
        ]
    )
    interactions = pd.DataFrame(
        {
            Columns.User: ["u1"],
            Columns.Item: ["i1"],
            Columns.Weight: [1.0],
            Columns.Datetime: [pd.Timestamp.utcnow()],
        }
    )
    model = ContentFallbackModel(
        feature_columns=[FeatureColumn(column="category", type="categorical")],
        items=items,
        interactions=interactions,
    )
    model.fit(_DummyDataset())
    recs = model.recommend(users=["u1"], dataset=_DummyDataset(), k=5, filter_viewed=True)
    assert set(recs[Columns.User].astype(str)) == {"u1"}
    assert (recs[Columns.Item] == "i_new").all()


def test_recommend_skips_historyless_user_without_affecting_others(monkeypatch):
    monkeypatch.setattr("cicerone.content_fallback._RECOMMEND_THREAD_MIN_USERS", 1)
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer"},
            {"item_id": "i_new", "category": "beer"},
        ]
    )
    interactions = pd.DataFrame(
        {
            Columns.User: ["u1"],
            Columns.Item: ["i1"],
            Columns.Weight: [1.0],
            Columns.Datetime: [pd.Timestamp.utcnow()],
        }
    )
    model = ContentFallbackModel(
        feature_columns=[FeatureColumn(column="category", type="categorical")],
        items=items,
        interactions=interactions,
    )
    model.fit(_DummyDataset())
    recs = model.recommend(users=["u1", "u2"], dataset=_DummyDataset(), k=5, filter_viewed=True)
    assert set(recs[Columns.User].astype(str)) == {"u1"}
    assert "u2" not in set(recs[Columns.User].astype(str))
