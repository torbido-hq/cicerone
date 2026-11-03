"""Cleanup that runs while another error propagates: it logs any error, so the original error wins.

A leaf module (no cicerone imports) so publish/ and events/ can use it without an import cycle.
"""

from __future__ import annotations

import logging
from collections.abc import Callable


def cleanup_after_failure(cleanup: Callable[[], object], message: str, log: logging.Logger) -> None:
    try:
        cleanup()
    except Exception:
        log.exception(message)
