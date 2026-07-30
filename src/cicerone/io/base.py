"""Protocols for pluggable input/output backends.

Both input and output are abstracted behind these two interfaces so the job
doesn't care whether it's reading/writing static files (S3-compatible or
local disk) or a database.
"""

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

    def write_model_artifact(self, payload: bytes) -> None:
        """Persists a serialized ModelArtifact (see cicerone.artifact)."""
        ...


class RecommendationReader(Protocol):
    """Read-only counterpart of OutputSink, used by serve mode to read
    precomputed recommendations back out of the output store.
    """

    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame: ...

    def refresh(self) -> None:
        """Reloads any cached data. A no-op for backends that read live on
        every call (e.g. a database).
        """
        ...


class ManifestReader(Protocol):
    """Read-only access to job run manifests, used by the dashboard to read
    back whatever OutputSink.write_manifest() already wrote.
    """

    def read_latest(self) -> dict | None:
        """Returns the most recently written manifest, or None if the job
        has never run against this output store.
        """
        ...

    def read_recent(self, limit: int) -> list[dict]:
        """Returns up to `limit` most recent manifests, newest first. The
        dataset backend only ever has one entry regardless of `limit`.
        """
        ...
