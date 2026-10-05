"""Rewrite a JSONL blob without one user's rows."""

from __future__ import annotations

import json

from cicerone.io.recommendation_schema import USER_COLUMN


def drop_user_lines(raw: bytes | None, user_id: str) -> tuple[bytes, int]:
    """Kept JSONL and the match count. A line that is not a user row raises first."""
    if not raw:
        return b"", 0
    kept: list[str] = []
    removed = 0
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"output JSONL line {line_number} is not JSON") from exc
        if (
            not isinstance(parsed, dict)
            or USER_COLUMN not in parsed
            or not str(parsed[USER_COLUMN] or "").strip()
        ):
            raise ValueError(f"output JSONL line {line_number} is missing {USER_COLUMN}")
        if str(parsed[USER_COLUMN]) == user_id:
            removed += 1
            continue
        kept.append(stripped)
    payload = "".join(f"{line}\n" for line in kept).encode("utf-8")
    return payload, removed
