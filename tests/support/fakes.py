"""Shared test fakes."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cicerone.io.recommendation_reader import select_cold_start_fallback
from cicerone.io.recommendation_schema import filter_variant_rows


class FakeS3Body:
    def __init__(self, payload: bytes, *, fail: bool = False) -> None:
        self._payload = payload
        self._fail = fail
        self.closed = False
        self.reads: list[int | None] = []

    def read(self, size: int | None = None) -> bytes:
        self.reads.append(size)
        if self._fail:
            raise RuntimeError("read failed")
        if size is None:
            return self._payload
        return self._payload[:size]

    def close(self) -> None:
        self.closed = True


class FakeS3Client:
    def __init__(self, body: FakeS3Body) -> None:
        self.body = body

    def get_object(self, **_kwargs: Any) -> dict[str, Any]:
        return {"Body": self.body}


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


class FakeGuard:
    def __init__(self, result: bool = True):
        self.result = result
        self.calls: list[str] = []

    def trigger(self, triggered_by: str) -> bool:
        self.calls.append(triggered_by)
        return self.result


class FakeLock:
    """LockBackend that records ``acquire`` / ``owned`` / ``release`` calls in ``events``."""

    def __init__(self, *, acquires: bool = True, owned: bool = True) -> None:
        self.acquires = acquires
        self.is_owned = owned
        self.events: list[str] = []

    def acquire(self) -> bool:
        self.events.append("acquire")
        return self.acquires

    def release(self) -> None:
        self.events.append("release")

    def owned(self) -> bool:
        self.events.append("owned")
        return self.is_owned

    def is_locked(self) -> bool:
        return True
