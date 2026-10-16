"""Copy a JSONL log without one user's rows."""

from __future__ import annotations

import json
from typing import BinaryIO

from cicerone.io.recommendation_schema import USER_COLUMN


def filter_user_lines(source: BinaryIO, sink: BinaryIO, user_id: str) -> int:
    """Stream kept lines to ``sink`` and return the match count. A line that is not a user row raises."""
    removed = 0
    for line_number, line in enumerate(source, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"output JSONL line {line_number} is not JSON") from exc
        stored = parsed.get(USER_COLUMN) if isinstance(parsed, dict) else None
        if stored is None or not str(stored).strip():
            raise ValueError(f"output JSONL line {line_number} is missing {USER_COLUMN}")
        if str(stored).strip() == user_id:
            removed += 1
            continue
        sink.write(stripped + b"\n")
    return removed
