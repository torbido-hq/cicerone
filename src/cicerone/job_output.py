"""Job publication fence and manifest write helpers."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from cicerone.io.base import OutputSink, ensure_sink_writer_held
from cicerone.locks import LockLostError, WriteFence

logger = logging.getLogger(__name__)


@dataclass(slots=True)
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
    # Some output may be on storage, but the run failed before its success manifest.
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


_MAX_ERROR_LENGTH = 500


def truncate_job_error(exc: BaseException) -> str:
    error_message = str(exc)
    if len(error_message) > _MAX_ERROR_LENGTH:
        return error_message[:_MAX_ERROR_LENGTH] + "... (truncated)"
    return error_message


def skip_stale_job_manifest(fence: WriteFence, *, exc: BaseException | None = None) -> bool:
    if isinstance(exc, LockLostError):
        logger.error("Skipping job manifest: %s", exc)
        return True
    if fence.check is not None and not fence.check():
        logger.error("Skipping job manifest: %s", fence.lost)
        return True
    return False


def ensure_publication_fence(sink: object, retrain: WriteFence) -> None:
    retrain.ensure()
    ensure_sink_writer_held(sink)


def write_job_manifest(
    sink: OutputSink, manifest: JobManifest, *, skip_if_newer_than: str | None = None
) -> bool:
    return sink.write_manifest(manifest.as_dict(), skip_if_newer_than=skip_if_newer_than)
