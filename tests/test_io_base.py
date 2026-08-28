from __future__ import annotations

import pandas as pd

from cicerone.io.base import BaseRecommendationReader
from cicerone.io.recommendation_reader import _ItemFilterMixin


def test_base_recommendation_reader_defaults():
    class Minimal(BaseRecommendationReader):
        def get_recommendations(self, user_id: str, k: int, *, variant: str | None = None) -> pd.DataFrame:
            del user_id, k, variant
            return pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.5}])

    reader = Minimal()
    assert list(reader.get_recommendations("u1", 1)["item_id"]) == ["i1"]
    assert reader.get_items() is None
    assert reader.items_version() == 0
    assert reader.get_cold_start_fallback(5).empty
    reader.refresh()  # no-op default
    reader.configure_item_filters(category_column="category", availability_filters=["published"])


def test_item_filter_mixin_lazy_inits_if_subclass_skips_init():
    class Bare(_ItemFilterMixin):
        pass

    bare = Bare()
    assert bare.items_version() == 0
    assert bare.get_items() is None
    bare.configure_item_filters(category_column="category")
    assert bare.items_version() == 1
