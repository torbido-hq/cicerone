"""Protocols for pluggable input/output backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from cicerone.capability import CapabilityMeta


def manifest_newer(manifest: Mapping[str, Any], started_at: str) -> bool:
    """Whether a stored run stamped ``manifest`` after ``started_at``; only string stamps count."""
    return any(
        isinstance(stamp, str) and stamp > started_at
        for stamp in (manifest.get("generated_at"), manifest.get("last_incremental_at"))
    )


class InputSource(Protocol):
    def read_events(self) -> pd.DataFrame: ...

    def read_users(self) -> pd.DataFrame | None: ...

    def read_items(self) -> pd.DataFrame | None: ...

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame: ...

    def get_events_for_users(self, user_ids: Sequence[str], limit: int) -> dict[str, pd.DataFrame]: ...

    def get_user(self, user_id: str) -> dict[str, Any] | None: ...


class UserHistoryReader(Protocol):
    """Per-user slices of input events/users for the dashboard inspector."""

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame: ...

    def get_events_for_users(self, user_ids: Sequence[str], limit: int) -> dict[str, pd.DataFrame]: ...

    def get_user(self, user_id: str) -> dict[str, Any] | None: ...


class OutputSink(Protocol):
    def write_recommendations(self, df: pd.DataFrame) -> None: ...

    def replace_recommendations_for_users(self, df: pd.DataFrame, *, user_ids: Sequence[str]) -> int:
        """Replace all rows for ``user_ids`` with ``df``; return distinct user count after write."""
        ...

    def write_manifest(self, manifest: dict, *, skip_if_newer_than: str | None = None) -> bool: ...

    def write_model_artifact(self, payload: bytes) -> None: ...

    def read_model_artifact(self) -> bytes | None:
        """Latest fitted-model blob, or ``None`` if the sink has not written one."""
        ...

    def write_items_snapshot(self, df: pd.DataFrame) -> None:
        """Persist items for serve-time category/availability filters."""
        ...

    def write_surfaces(
        self,
        *,
        popular: pd.DataFrame,
        latest: pd.DataFrame,
        neighbors: pd.DataFrame,
    ) -> None:
        """Persist popular / latest / item-neighbor tables for serve surfaces."""
        ...

    def write_item_scores(self, df: pd.DataFrame) -> None:
        """Persist catalog-wide popular/latest scores for search indexers."""
        ...


# Optional sink capabilities; callers check them with isinstance.
@runtime_checkable
class AppliedEventLog(Protocol, metaclass=CapabilityMeta):
    """Records incremental event keys in the same write as the recommendations they produced."""

    def applied_event_keys(self, keys: Sequence[str]) -> set[str]: ...

    def replace_recommendations_for_users(
        self, df: pd.DataFrame, *, user_ids: Sequence[str], applied_keys: Sequence[str] = ()
    ) -> int: ...


@runtime_checkable
class GuardedRecommendationsSink(Protocol, metaclass=CapabilityMeta):
    def recommendations_write(self) -> AbstractContextManager[None]: ...


@runtime_checkable
class WriterHeldSink(Protocol, metaclass=CapabilityMeta):
    def ensure_writer_held(self) -> None: ...


@runtime_checkable
class ArtifactFingerprintSink(Protocol, metaclass=CapabilityMeta):
    def model_artifact_fingerprint(self) -> str | None:
        """Cheap identity of the latest artifact (mtime/ETag/written_at), or ``None``."""
        ...


@runtime_checkable
class ConditionalArtifactSink(Protocol, metaclass=CapabilityMeta):
    def replace_model_artifact_if(self, payload: bytes, expected_fingerprint: str) -> bool:
        """Write only when the current fingerprint still matches ``expected_fingerprint``."""
        ...


def recommendations_write_guard(sink: object) -> AbstractContextManager[None]:
    return sink.recommendations_write() if isinstance(sink, GuardedRecommendationsSink) else nullcontext()


def ensure_sink_writer_held(sink: object) -> None:
    if isinstance(sink, WriterHeldSink):
        sink.ensure_writer_held()


class RecommendationReader(Protocol):
    def get_recommendations(self, user_id: str, k: int, *, variant: str | None = None) -> pd.DataFrame: ...

    def get_recommendations_for_users(
        self, user_ids: Sequence[str], k: int, *, variant: str | None = None
    ) -> dict[str, pd.DataFrame]: ...

    def refresh(self) -> bool | None:
        """Reload caches. No-op for live backends. False means the previous recommendations were kept."""
        ...

    def get_items(self) -> pd.DataFrame | None:
        """Items snapshot for serve-time filters, if available."""
        ...

    def items_version(self) -> int:
        """Monotonic token bumped when the items snapshot changes."""
        ...

    def get_item_scores_snapshot(self) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Id-sorted catalog ``item_scores`` rows and their item ids, or empty when none written."""
        ...

    def get_cold_start_fallback(self, k: int, *, variant: str | None = None) -> pd.DataFrame:
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


@runtime_checkable
class VariantSnapshotReader(Protocol, metaclass=CapabilityMeta):
    def present_variant_names(self) -> tuple[str, ...] | None: ...


class BaseRecommendationReader(ABC):
    """Optional base with empty defaults for serve filter / cold-start hooks.

    Custom readers can subclass this and only implement ``get_recommendations``
    (and override the hooks they need) instead of satisfying every Protocol
    method from scratch.
    """

    def refresh(self) -> bool | None:
        return None

    def get_items(self) -> pd.DataFrame | None:
        return None

    def items_version(self) -> int:
        return 0

    def get_item_scores_snapshot(self) -> tuple[pd.DataFrame, np.ndarray | None]:
        return pd.DataFrame(), None

    def get_cold_start_fallback(self, k: int, *, variant: str | None = None) -> pd.DataFrame:
        del k, variant
        return pd.DataFrame()

    def present_variant_names(self) -> tuple[str, ...] | None:
        return None

    def configure_item_filters(
        self,
        *,
        category_column: str | None = None,
        availability_filters: Sequence[str] = (),
    ) -> None:
        del category_column, availability_filters

    @abstractmethod
    def get_recommendations(self, user_id: str, k: int, *, variant: str | None = None) -> pd.DataFrame: ...

    def get_recommendations_for_users(
        self, user_ids: Sequence[str], k: int, *, variant: str | None = None
    ) -> dict[str, pd.DataFrame]:
        return {
            str(user_id): self.get_recommendations(str(user_id), k, variant=variant) for user_id in user_ids
        }


def recommendations_for_users(
    reader: RecommendationReader,
    user_ids: Sequence[str],
    k: int,
    *,
    variant: str | None = None,
) -> dict[str, pd.DataFrame]:
    ids = [str(user_id) for user_id in user_ids]
    bulk = getattr(reader, "get_recommendations_for_users", None)
    if callable(bulk):
        loaded = bulk(ids, k, variant=variant)
        empty = pd.DataFrame()
        return {user_id: loaded.get(user_id, empty) for user_id in ids}
    return {user_id: reader.get_recommendations(user_id, k, variant=variant) for user_id in ids}


class ManifestReader(Protocol):
    def read_latest(self) -> dict | None: ...

    def read_recent(self, limit: int) -> list[dict]:
        """Up to `limit` newest manifests. Dataset backend always returns 0–1."""
        ...
