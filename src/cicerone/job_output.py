"""Job publication fence and manifest write helpers."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

from cicerone.locks import LockLostError

logger = logging.getLogger(__name__)

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


def write_job_manifest(sink: Any, manifest: dict[str, Any], *, skip_if_newer_than: str | None = None) -> bool:
    write = sink.write_manifest
    if write_manifest_accepts_skip(write):
        result = write(manifest, skip_if_newer_than=skip_if_newer_than)
        return result is not False
    write(manifest)
    return True
