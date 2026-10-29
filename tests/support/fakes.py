"""Shared test fakes."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pandas as pd

from cicerone.io.recommendation_reader import select_cold_start_fallback
from cicerone.io.recommendation_schema import filter_variant_rows


class RecordingConnection:
    """Records executed SQL; ``begin`` and ``begin_nested`` are no-op transactions."""

    def __init__(self, dialect: Any) -> None:
        self.dialect = dialect
        self.sql: list[str] = []

    @contextmanager
    def begin(self) -> Iterator[RecordingConnection]:
        yield self

    def begin_nested(self) -> RecordingConnection:
        return self

    def commit(self) -> None:
        pass

    def execute(self, statement: Any) -> None:
        self.sql.append(str(statement))


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


def patch_parquet_reads(
    monkeypatch: Any, *, pushdown_error: BaseException | None = None
) -> list[dict[str, Any]]:
    """Record ``read_parquet`` calls; raise ``pushdown_error`` on reads with columns or filters."""
    from cicerone.io import options as io_options

    real = io_options.read_parquet
    calls: list[dict[str, Any]] = []

    def read(options: dict[str, Any], filename: str, *, columns: Any = None, filters: Any = None) -> Any:
        calls.append({"columns": columns, "filters": filters})
        if pushdown_error is not None and (columns is not None or filters is not None):
            raise pushdown_error
        return real(options, filename, columns=columns, filters=filters)

    monkeypatch.setattr(io_options, "read_parquet", read)
    return calls


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
    """Writer lock whose ``acquire`` / ``owned`` answers are fixed values or callables."""

    def __init__(
        self, *, acquire: bool | Callable[[], bool] = True, owned: bool | Callable[[], bool] = True
    ) -> None:
        self._acquire = acquire
        self._owned = owned
        self.acquires = 0

    def acquire(self) -> bool:
        self.acquires += 1
        return self._acquire() if callable(self._acquire) else self._acquire

    def release(self) -> None:
        return None

    def owned(self) -> bool:
        return self._owned() if callable(self._owned) else self._owned

    def is_locked(self) -> bool:
        return True


class FenceCheck:
    """Fence check that passes for the first ``fail_at - 1`` calls, then fails."""

    def __init__(self, fail_at: int) -> None:
        self.fail_at = fail_at
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls < self.fail_at


def raising(exc: BaseException) -> Callable[..., Any]:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise exc

    return _raise
