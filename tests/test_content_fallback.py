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
    # i3 has only null feature values → no tokens → skipped from matrix
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
    # u2 has no history; u3's history item is unknown to the feature matrix.
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
