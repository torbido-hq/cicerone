"""Sidecar publish helpers shared by the job and incremental updater."""

from __future__ import annotations

import logging

from cicerone.config.settings import IOSettings
from cicerone.io.factory import build_manifest_reader

logger = logging.getLogger(__name__)


def sidecar_generation_current(output: IOSettings, generated_at: str) -> bool:
    try:
        latest = build_manifest_reader(output).read_latest()
    except Exception:
        logger.exception("Failed to read manifest generation before sidecar publish")
        return False
    current = latest.get("generated_at") if latest else None
    return current is not None and str(current) == generated_at
