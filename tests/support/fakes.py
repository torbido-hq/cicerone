"""Shared test fakes."""

from __future__ import annotations

import sys
from typing import Any

import pytest


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


def hide_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make ``import name`` fail as if its optional extra were not installed."""
    monkeypatch.setitem(sys.modules, name, None)
