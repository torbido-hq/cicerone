"""Incremental event keys recorded with the recommendations write that applied them.

A redelivery after a restart is then recognised and acked instead of applied twice.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta

APPLIED_EVENTS_RETENTION = timedelta(days=7)
# Dataset output keeps the record in the parquet metadata; cap it so the file stays small.
APPLIED_EVENTS_MAX_KEYS = 100_000
DEFAULT_APPLIED_EVENTS_TABLE = "incremental_applied_events"
APPLIED_EVENTS_ATTR = "cicerone_applied_events"


def replay_key(kind: str, value: str) -> str:
    """Stored key: a hash of ``kind:value``, so ids and fingerprints never collide and raw ids stay out."""
    return hashlib.sha256(f"{kind}:{value}".encode()).hexdigest()


def applied_cutoff(now: datetime) -> str:
    return (now - APPLIED_EVENTS_RETENTION).isoformat()


def unexpired_keys(recorded: Mapping[str, str], now: datetime) -> set[str]:
    cutoff = applied_cutoff(now)
    return {str(key) for key, at in recorded.items() if str(at) >= cutoff}


def merge_applied_keys(existing: Mapping[str, str], keys: Iterable[str], now: datetime) -> dict[str, str]:
    """``{key: applied_at}`` with ``keys`` added, older than the retention dropped, newest capped."""
    stamp = now.astimezone(UTC).isoformat()
    merged = {str(key): str(at) for key, at in existing.items() if str(at) >= applied_cutoff(now)}
    merged.update(dict.fromkeys((str(key) for key in keys), stamp))
    if len(merged) <= APPLIED_EVENTS_MAX_KEYS:
        return merged
    newest = sorted(merged.items(), key=lambda entry: entry[1], reverse=True)[:APPLIED_EVENTS_MAX_KEYS]
    return dict(newest)
