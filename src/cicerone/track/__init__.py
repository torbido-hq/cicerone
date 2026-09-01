"""Impression/click tracking (off the training event path)."""

from __future__ import annotations

from cicerone.track.normalize import (
    TRACK_KIND_CLICK,
    TRACK_KIND_IMPRESSION,
    TrackNormalizeError,
    normalize_track,
)
from cicerone.track.store import TrackStore, require_appendable_track_log

__all__ = [
    "TRACK_KIND_CLICK",
    "TRACK_KIND_IMPRESSION",
    "TrackNormalizeError",
    "TrackStore",
    "normalize_track",
    "require_appendable_track_log",
]
