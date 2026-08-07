"""Protocols for pluggable input/output backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol

import pandas as pd


class InputSource(Protocol):
    def read_events(self) -> pd.DataFrame: ...

    def read_users(self) -> pd.DataFrame | None: ...

    def read_items(self) -> pd.DataFrame | None: ...


class OutputSink(Protocol):
    def write_recommendations(self, df: pd.DataFrame) -> None: ...

    def write_manifest(self, manifest: dict) -> None: ...

    def write_model_artifact(self, payload: bytes) -> None: ...

    def write_items_snapshot(self, df: pd.DataFrame) -> None:
        """Persist items for serve-time category/availability filters."""
        ...


class RecommendationReader(Protocol):
    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame: ...

    def refresh(self) -> None:
        """Reload caches. No-op for live backends."""
        ...

    def get_items(self) -> pd.DataFrame | None:
        """Items snapshot for serve-time filters, if available."""
        ...

    def items_version(self) -> int:
        """Monotonic token bumped when the items snapshot changes."""
        ...

    def get_cold_start_fallback(self, k: int) -> pd.DataFrame:
        """``__cold_start__`` rows, else a popular/latest heuristic for unknown users."""
        ...

    def configure_item_filters(
        self,
        *,
        category_column: str | None = None,
        availability_filters: Sequence[str] = (),
    ) -> None:
        """Configure serve-time category / availability columns on the items snapshot."""
        ...


class BaseRecommendationReader(ABC):
    """Optional base with empty defaults for serve filter / cold-start hooks.

    Custom readers can subclass this and only implement ``get_recommendations``
    (and override the hooks they need) instead of satisfying every Protocol
    method from scratch.
    """

    def refresh(self) -> None:
        return

    def get_items(self) -> pd.DataFrame | None:
        return None

    def items_version(self) -> int:
        return 0

    def get_cold_start_fallback(self, k: int) -> pd.DataFrame:
        del k
        return pd.DataFrame()

    def configure_item_filters(
        self,
        *,
        category_column: str | None = None,
        availability_filters: Sequence[str] = (),
    ) -> None:
        del category_column, availability_filters

    @abstractmethod
    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame: ...


class ManifestReader(Protocol):
    def read_latest(self) -> dict | None: ...

    def read_recent(self, limit: int) -> list[dict]:
        """Up to `limit` newest manifests. Dataset backend always returns 0–1."""
        ...
