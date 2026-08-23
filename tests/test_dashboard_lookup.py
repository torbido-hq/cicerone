from __future__ import annotations

import pandas as pd
from conftest import make_settings

from cicerone.dashboard_lookup import (
    LOOKUP_FAILED,
    MISSING,
    format_recommendation_rows,
    lookup_k,
    lookup_recommendations,
)


class _BoomReader:
    def refresh(self) -> None:
        return

    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame:
        raise RuntimeError("dsn=postgres://secret@host/db")

    def get_items(self) -> pd.DataFrame | None:
        return None

    def get_cold_start_fallback(self, k: int) -> pd.DataFrame:
        return pd.DataFrame()


class _KReader:
    def refresh(self) -> None:
        return

    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "user_id": user_id,
                    "item_id": f"i{i}",
                    "rank": i,
                    "score": 1.0,
                    "source": "personalized",
                }
                for i in range(1, k + 1)
            ]
        )

    def get_items(self) -> pd.DataFrame | None:
        return None

    def get_cold_start_fallback(self, k: int) -> pd.DataFrame:
        return pd.DataFrame()


def test_lookup_k_is_min_of_top_k_and_cap():
    assert lookup_k(50, 20) == 20
    assert lookup_k(10, 20) == 10
    assert lookup_k(50, 5) == 5


def test_lookup_recommendations_empty_user_id_is_not_queried():
    result = lookup_recommendations(make_settings(dashboard_enabled=True), None, "  ")

    assert result["queried"] is False
    assert result["items"] == []


def test_lookup_recommendations_hides_exception_details():
    result = lookup_recommendations(make_settings(dashboard_enabled=True), _BoomReader(), "u1")

    assert result["error"] == LOOKUP_FAILED
    assert "postgres" not in result["error"]
    assert "secret" not in str(result)


def test_lookup_recommendations_uses_dashboard_lookup_k():
    settings = make_settings(dashboard_enabled=True, top_k=50, dashboard_lookup_k=5)
    result = lookup_recommendations(settings, _KReader(), "u1")

    assert [row["item_id"] for row in result["items"]] == ["i1", "i2", "i3", "i4", "i5"]


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


def test_format_recommendation_rows_empty_text_is_placeholder():
    recs = pd.DataFrame([{"item_id": "i1", "rank": 1, "score": 0.1, "source": "", "category": ""}])
    rows = format_recommendation_rows(recs, category_column="category")
    assert rows[0]["source"] == MISSING
    assert rows[0]["category"] == MISSING


def test_format_recommendation_rows_pd_na_text_is_placeholder():
    recs = pd.DataFrame([{"item_id": "i1", "rank": 1, "score": 0.1, "source": pd.NA, "category": pd.NA}])
    rows = format_recommendation_rows(recs, category_column="category")
    assert rows[0]["source"] == MISSING
    assert rows[0]["category"] == MISSING
