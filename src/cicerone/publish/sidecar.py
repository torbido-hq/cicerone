"""Sidecar publish helpers shared by the job and incremental updater."""

from __future__ import annotations

import logging

from cicerone.config.settings import IOSettings
from cicerone.io.factory import build_manifest_reader
from cicerone.job_eval import OPTIONAL_IO_ERRORS, log_caught
from cicerone.locks import LockLostError, WriterLockBusyError

logger = logging.getLogger(__name__)


def sidecar_generation_current(output: IOSettings, generated_at: str) -> bool | None:
    try:
        latest = build_manifest_reader(output).read_latest()
    except (LockLostError, WriterLockBusyError):
        raise
    except OPTIONAL_IO_ERRORS as exc:
        log_caught("Failed to read manifest generation before sidecar publish", exc)
        return None
    if latest is None:
        return None
    current = latest.get("generated_at")
    if current is None:
        return False
    return str(current) == generated_at


def log_sidecar_generation_skip(current: bool | None, *, incremental: bool = False) -> None:
    prefix = "Skipping incremental publish" if incremental else "Skipping publish"
    if current is False:
        logger.info("%s: recommendations were superseded", prefix)
        return
    logger.info("%s: could not confirm sidecar generation", prefix)
