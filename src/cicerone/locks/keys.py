"""Lock key helpers shared by Postgres and Redis backends."""

from __future__ import annotations

import hashlib

from cicerone.config.constants import DEFAULT_LOCK_KEY, DEFAULT_LOCK_TTL_SECONDS

REDIS_LOCK_KEY = DEFAULT_LOCK_KEY
REDIS_LOCK_TTL_MS = DEFAULT_LOCK_TTL_SECONDS * 1000


def events_apply_lock_key(run_guard_key: str) -> str:
    """Lease key for incremental apply; distinct from the full-retrain lock."""
    return f"{run_guard_key}:events:apply"


def advisory_keys_from_lock_key(lock_key: str) -> tuple[int, int]:
    digest = hashlib.sha256(lock_key.encode()).digest()
    return (
        int.from_bytes(digest[:4], "big") & 0x7FFFFFFF,
        int.from_bytes(digest[4:8], "big") & 0x7FFFFFFF,
    )


PG_ADVISORY_KEY1, PG_ADVISORY_KEY2 = advisory_keys_from_lock_key(DEFAULT_LOCK_KEY)
