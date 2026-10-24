from __future__ import annotations

import pandas as pd

from cicerone.io.recommendation_reader import select_cold_start_fallback
from cicerone.io.recommendation_schema import filter_variant_rows


class FakeReader:
    def __init__(self, recs: pd.DataFrame, items: pd.DataFrame | None = None):
        self._recs = recs
        self._items = items
        self._items_version = 0
        self.refresh_calls = 0
        self.get_items_calls = 0

    def refresh(self) -> None:
        self.refresh_calls += 1
        self._items_version += 1

    def get_recommendations(self, user_id: str, k: int, *, variant: str | None = None) -> pd.DataFrame:
        rows = self._recs[self._recs["user_id"] == user_id].sort_values("rank")
        rows = filter_variant_rows(rows, variant)
        return rows.head(k).reset_index(drop=True)

    def get_items(self) -> pd.DataFrame | None:
        self.get_items_calls += 1
        return self._items

    def items_version(self) -> int:
        return self._items_version

    def get_cold_start_fallback(self, k: int, *, variant: str | None = None) -> pd.DataFrame:
        rows = filter_variant_rows(self._recs, variant)
        return select_cold_start_fallback(rows, k)

    def configure_item_filters(
        self,
        *,
        category_column: str | None = None,
        availability_filters=(),
    ) -> None:
        del category_column, availability_filters
        self._items_version += 1


class FakeManifest:
    def read_latest(self):
        return {"generated_at": "2026-08-04T12:00:00+00:00"}

    def read_recent(self, limit: int):
        del limit
        latest = self.read_latest()
        return [latest] if latest else []
