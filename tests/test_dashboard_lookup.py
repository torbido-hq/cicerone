from __future__ import annotations

import pandas as pd
from conftest import make_settings

from cicerone.dashboard_lookup import (
    MISSING,
    format_recommendation_rows,
    lookup_k,
    lookup_recommendations,
)


def test_lookup_k_caps_at_20():
    assert lookup_k(50) == 20
    assert lookup_k(10) == 10


def test_lookup_recommendations_empty_user_id_is_not_queried():
    result = lookup_recommendations(make_settings(dashboard_enabled=True), None, "  ")

    assert result["queried"] is False
    assert result["items"] == []


def test_format_recommendation_rows_uses_placeholders():
    recs = pd.DataFrame([{"item_id": "i1", "rank": None, "score": None, "source": None, "category": None}])
    rows = format_recommendation_rows(recs, category_column="category")

    assert rows == [
        {
            "rank": MISSING,
            "item_id": "i1",
            "score": MISSING,
            "source": MISSING,
            "category": MISSING,
        }
    ]


def test_format_recommendation_rows_formats_score():
    recs = pd.DataFrame([{"item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}])
    rows = format_recommendation_rows(recs, category_column=None)

    assert rows == [{"rank": "1", "item_id": "i1", "score": "0.9000", "source": "personalized"}]
