"""Normalize track payloads into rows (impressions/clicks)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from cicerone.config.constants import TRACK_KIND_CLICK, TRACK_KIND_IMPRESSION, TRACK_KINDS
from cicerone.events.normalize import EventNormalizeError, parse_occurred_at

_REQUIRED = ("kind", "user_id", "item_id", "occurred_at")


class TrackNormalizeError(ValueError):
    """Invalid track payload."""


@dataclass(frozen=True)
class NormalizedTrack:
    kind: str
    user_id: str
    item_id: str
    rank: int | None
    occurred_at: datetime
    event_id: str
    variant: str | None
    experiment_id: str | None
    generated_at: str | None

    def as_row(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "user_id": self.user_id,
            "item_id": self.item_id,
            "rank": self.rank,
            "occurred_at": self.occurred_at.isoformat(),
            "event_id": self.event_id,
            "variant": self.variant,
            "experiment_id": self.experiment_id,
            "generated_at": self.generated_at,
        }


def _as_mapping(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    raise TrackNormalizeError("track event must be a JSON object")


def _optional_str(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value in (None, ""):
        return None
    return str(value).strip() or None


def _parse_rank(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        rank = int(value)
    except (TypeError, ValueError) as exc:
        raise TrackNormalizeError("rank must be an integer") from exc
    if rank < 1:
        raise TrackNormalizeError("rank must be >= 1")
    return rank


def normalize_track(payload: Any) -> NormalizedTrack:
    raw = _as_mapping(payload)
    missing = [key for key in _REQUIRED if raw.get(key) in (None, "")]
    if missing:
        raise TrackNormalizeError(f"missing required field(s): {', '.join(missing)}")
    kind = str(raw["kind"]).strip().lower()
    if kind not in TRACK_KINDS:
        raise TrackNormalizeError(f"kind must be one of {list(TRACK_KINDS)}, got {kind!r}")
    user_id = str(raw["user_id"]).strip()
    item_id = str(raw["item_id"]).strip()
    if not user_id or not item_id:
        raise TrackNormalizeError("user_id and item_id must be non-empty")
    try:
        occurred_at = parse_occurred_at(raw["occurred_at"])
    except EventNormalizeError as exc:
        raise TrackNormalizeError(str(exc)) from exc
    rank = _parse_rank(raw.get("rank"))
    if kind == TRACK_KIND_IMPRESSION and rank is None:
        raise TrackNormalizeError("impression requires rank >= 1")
    event_id = _optional_str(raw, "event_id") or _optional_str(raw, "idempotency_key") or str(uuid4())
    return NormalizedTrack(
        kind=kind,
        user_id=user_id,
        item_id=item_id,
        rank=rank,
        occurred_at=occurred_at,
        event_id=event_id,
        variant=_optional_str(raw, "variant"),
        experiment_id=_optional_str(raw, "experiment_id"),
        generated_at=_optional_str(raw, "generated_at"),
    )


__all__ = [
    "TRACK_KIND_CLICK",
    "TRACK_KIND_IMPRESSION",
    "NormalizedTrack",
    "TrackNormalizeError",
    "normalize_track",
]
