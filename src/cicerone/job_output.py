"""Job publication fence and manifest write helpers."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from cicerone.locks import LockLostError

logger = logging.getLogger(__name__)


@dataclass
class JobManifest:
    triggered_by: str | None = None
    lock_backend: str | None = None
    status: str = "failed"
    error: str | None = None
    n_events: int | None = None
    n_target_users: int | None = None
    n_users_with_recommendations: int | None = None
    n_items: int | None = None
    top_k: int | None = None
    models: str = ""
    model_weights: str = ""
    rrf_k: float | None = None
    artifact_written: bool = False
    artifact_schema_version: int | None = None
    partial_outputs: bool = False
    automl_enabled: bool = False
    automl_metrics: str = ""
    experiment_id: str = ""
    experiment_variants: str = ""
    track_eval: str = ""
    served_eval: str = ""
    generated_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def keys(self) -> Any:
        return self.__dataclass_fields__.keys()

    def __getitem__(self, key: str) -> Any:
        _require_manifest_field(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        _require_manifest_field(key)
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self.__dataclass_fields__:
            return default
        return getattr(self, key)

    def update(self, other: Mapping[str, Any]) -> None:
        items = tuple(other.items())
        for key, _ in items:
            _require_manifest_field(key)
        for key, value in items:
            setattr(self, key, value)


_JOB_MANIFEST_FIELDS = frozenset(item.name for item in fields(JobManifest))


def _require_manifest_field(key: str) -> None:
    if key not in _JOB_MANIFEST_FIELDS:
        raise KeyError(key)


_MAX_ERROR_LENGTH = 500


def truncate_job_error(exc: BaseException) -> str:
    error_message = str(exc)
    if len(error_message) > _MAX_ERROR_LENGTH:
        return error_message[:_MAX_ERROR_LENGTH] + "... (truncated)"
    return error_message


def skip_stale_job_manifest(
    *,
    fence_check: Callable[[], bool] | None = None,
    exc: BaseException | None = None,
) -> bool:
    if isinstance(exc, LockLostError):
        logger.error("Skipping job manifest: %s", exc)
        return True
    if fence_check is not None and not fence_check():
        logger.error("Skipping job manifest: retrain lock lost before write")
        return True
    return False


def ensure_fence(fence_check: Callable[[], bool] | None) -> None:
    if fence_check is not None and not fence_check():
        raise LockLostError("retrain lock lost before write", kind="retrain")


def ensure_publication_fence(sink: Any, fence_check: Callable[[], bool] | None) -> None:
    ensure_fence(fence_check)
    ensure = getattr(sink, "ensure_writer_held", None)
    if callable(ensure):
        ensure()


def write_manifest_accepts_skip(write: Any) -> bool:
    try:
        return "skip_if_newer_than" in inspect.signature(write).parameters
    except (TypeError, ValueError):
        return False


def write_job_manifest(
    sink: Any, manifest: JobManifest | Mapping[str, Any], *, skip_if_newer_than: str | None = None
) -> bool:
    payload = manifest.as_dict() if isinstance(manifest, JobManifest) else dict(manifest)
    write = sink.write_manifest
    if write_manifest_accepts_skip(write):
        result = write(payload, skip_if_newer_than=skip_if_newer_than)
        return result is not False
    write(payload)
    return True
