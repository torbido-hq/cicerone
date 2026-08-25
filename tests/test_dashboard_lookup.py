from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import make_settings

from cicerone.dashboard_lookup import (
    HISTORY_FAILED,
    HISTORY_UNAVAILABLE,
    LOOKUP_FAILED,
    MISSING,
    format_event_rows,
    format_recommendation_rows,
    format_user_attrs,
    lookup_inspector,
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


class _History:
    def __init__(
        self,
        events: pd.DataFrame,
        user: dict | None = None,
        *,
        events_error: Exception | None = None,
        user_error: Exception | None = None,
    ):
        self._events = events
        self._user = user
        self._events_error = events_error
        self._user_error = user_error

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
        if self._events_error is not None:
            raise self._events_error
        rows = self._events[self._events["user_id"].astype(str) == user_id]
        return rows.head(limit).reset_index(drop=True)

    def get_user(self, user_id: str) -> dict | None:
        if self._user_error is not None:
            raise self._user_error
        if self._user is None or str(self._user.get("user_id")) != user_id:
            return None
        return self._user


def test_lookup_k_is_min_of_top_k_and_cap():
    assert lookup_k(50, 20) == 20
    assert lookup_k(10, 20) == 10
    assert lookup_k(50, 5) == 5


def test_lookup_recommendations_empty_user_id_is_not_queried():
    result = lookup_recommendations(make_settings(dashboard_enabled=True), None, "  ")

    assert result["queried"] is False
    assert result["items"] == []
    assert result["events"] == []


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
            "reasons": MISSING,
            "category": MISSING,
        }
    ]


def test_format_recommendation_rows_formats_score():
    recs = pd.DataFrame([{"item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}])
    rows = format_recommendation_rows(recs, category_column=None)

    assert rows == [
        {
            "rank": "1",
            "item_id": "i1",
            "score": "0.9000",
            "source": "personalized",
            "reasons": MISSING,
        }
    ]


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


def test_format_recommendation_rows_summarizes_reasons():
    recs = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "rank": 1,
                "score": 0.9,
                "source": "blended",
                "reasons": (
                    '{"sources":[{"label":"personalized"},{"label":"popular_fallback"}],'
                    '"similar_items":[{"item_id":"i9","score":0.5}]}'
                ),
            }
        ]
    )
    rows = format_recommendation_rows(recs, category_column=None)
    assert rows[0]["reasons"] == "personalized+popular_fallback · like i9"


def test_format_event_rows_uses_placeholders():
    events = pd.DataFrame([{"item_id": None, "event_type": None, "quantity": None, "occurred_at": None}])
    rows = format_event_rows(events)

    assert rows == [
        {
            "occurred_at": MISSING,
            "item_id": MISSING,
            "event_type": MISSING,
            "quantity": MISSING,
        }
    ]


def test_format_event_rows_formats_quantity_and_timestamp():
    occurred = pd.Timestamp("2026-08-21T12:00:00Z")
    events = pd.DataFrame(
        [{"item_id": "i1", "event_type": "purchase", "quantity": 3.0, "occurred_at": occurred}]
    )
    rows = format_event_rows(events)

    assert rows[0]["item_id"] == "i1"
    assert rows[0]["event_type"] == "purchase"
    assert rows[0]["quantity"] == "3"
    assert rows[0]["occurred_at"].startswith("2026-08-21")


def test_format_user_attrs_skips_user_id_and_missing():
    rows = format_user_attrs(
        {"user_id": "u1", "region_slug": "lazio", "favorite_styles": ["ipa", "stout"], "empty": None}
    )

    assert rows == [
        {"name": "region_slug", "value": "lazio"},
        {"name": "favorite_styles", "value": "ipa, stout"},
    ]


def test_format_user_attrs_accepts_series_and_ndarray():
    rows = format_user_attrs(
        {
            "user_id": "u1",
            "tags": pd.Series(["ipa", "stout"]),
            "codes": np.array(["a", "b"]),
            "blank": "",
        }
    )

    assert rows == [
        {"name": "tags", "value": "ipa, stout"},
        {"name": "codes", "value": "a, b"},
    ]


def test_lookup_inspector_keeps_recommendations_when_user_attrs_are_sequences():
    events = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "view", "quantity": 1, "occurred_at": None}]
    )
    result = lookup_inspector(
        make_settings(dashboard_enabled=True),
        _KReader(),
        _History(
            events,
            {
                "user_id": "u1",
                "favorite_styles": pd.Series(["ipa", "stout"]),
                "tag_ids": np.array([1, 2]),
            },
        ),
        "u1",
    )

    assert result["error"] is None
    assert result["events"]
    assert result["user_attrs"] == [
        {"name": "favorite_styles", "value": "ipa, stout"},
        {"name": "tag_ids", "value": "1, 2"},
    ]


def test_lookup_inspector_empty_user_id_skips_history():
    result = lookup_inspector(
        make_settings(dashboard_enabled=True), _KReader(), _History(pd.DataFrame()), "  "
    )

    assert result["queried"] is False
    assert result["events"] == []


def test_lookup_inspector_marks_overlap_and_source_mix():
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "view",
                "quantity": 1,
                "occurred_at": "2026-08-21",
            },
            {
                "user_id": "u1",
                "item_id": "i9",
                "event_type": "purchase",
                "quantity": 2,
                "occurred_at": "2026-08-20",
            },
        ]
    )
    result = lookup_inspector(
        make_settings(dashboard_enabled=True, top_k=50, dashboard_lookup_k=2),
        _KReader(),
        _History(events, {"user_id": "u1", "region_slug": "lazio"}),
        "u1",
    )

    assert result["overlap_item_ids"] == ["i1"]
    assert result["source_mix"] == [{"source": "personalized", "count": 2}]
    assert result["warm"] is True
    assert result["user_attrs"] == [{"name": "region_slug", "value": "lazio"}]
    assert [row["item_id"] for row in result["events"]] == ["i1", "i9"]
    assert result["show_quantity"] is True


def test_lookup_inspector_history_unavailable_keeps_recommendations():
    result = lookup_inspector(make_settings(dashboard_enabled=True), _KReader(), None, "u1")

    assert [row["item_id"] for row in result["items"]] == [
        "i1",
        "i2",
        "i3",
        "i4",
        "i5",
        "i6",
        "i7",
        "i8",
        "i9",
        "i10",
    ]
    assert result["events_error"] == HISTORY_UNAVAILABLE
    assert result["error"] is None


def test_lookup_inspector_hides_history_exception_details():
    result = lookup_inspector(
        make_settings(dashboard_enabled=True),
        _KReader(),
        _History(pd.DataFrame(), events_error=RuntimeError("dsn=postgres://secret@host/db")),
        "u1",
    )

    assert result["events_error"] == HISTORY_FAILED
    assert "secret" not in str(result)
    assert result["items"]


def test_lookup_inspector_missing_events_file_is_unavailable():
    result = lookup_inspector(
        make_settings(dashboard_enabled=True),
        _KReader(),
        _History(pd.DataFrame(), events_error=FileNotFoundError("events.parquet")),
        "u1",
    )

    assert result["events_error"] == HISTORY_UNAVAILABLE


def test_lookup_inspector_caps_events():
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": f"i{i}",
                "event_type": "view",
                "quantity": 1,
                "occurred_at": f"2026-08-{i:02d}",
            }
            for i in range(1, 10)
        ]
    )
    result = lookup_inspector(
        make_settings(dashboard_enabled=True, dashboard_lookup_events=3),
        _KReader(),
        _History(events),
        "u1",
    )

    assert [row["item_id"] for row in result["events"]] == ["i1", "i2", "i3"]


def test_lookup_inspector_user_attr_error_still_returns_events():
    events = pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "event_type": "view", "quantity": 1, "occurred_at": None}]
    )
    result = lookup_inspector(
        make_settings(dashboard_enabled=True),
        _KReader(),
        _History(events, user_error=RuntimeError("users boom")),
        "u1",
    )

    assert result["events"]
    assert result["user_attrs"] == []
