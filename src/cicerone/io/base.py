"""Protocols for pluggable input/output backends."""

from __future__ import annotations

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
        """Optional: persist items for serve-time category/availability filters."""
        ...


class RecommendationReader(Protocol):
    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame: ...

    def refresh(self) -> None:
        """Reload caches. No-op for live backends."""
        ...

    def get_items(self) -> pd.DataFrame | None:
        """Items snapshot for serve-time filters, if available."""
        ...

    def get_cold_start_fallback(self, k: int) -> pd.DataFrame:
        """Precomputed popular/latest rows for unknown users."""
        ...


class ManifestReader(Protocol):
    def read_latest(self) -> dict | None: ...

    def read_recent(self, limit: int) -> list[dict]:
        """Up to `limit` newest manifests. Dataset backend always returns 0–1."""
        ...
