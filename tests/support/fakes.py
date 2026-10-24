"""Shared fakes for storage tests."""

from __future__ import annotations

from typing import Any


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
