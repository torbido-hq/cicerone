"""Normalize backend payloads into NormalizedEvent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pandas as pd

from cicerone.events.base import NormalizedEvent

_REQUIRED = ("user_id", "item_id", "event_type", "occurred_at")


class EventNormalizeError(ValueError):
    """Invalid event payload."""


def _as_mapping(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    raise EventNormalizeError("event must be a JSON object")


def parse_occurred_at(value: Any) -> datetime:
    # stdlib only — keep webhook ingest off the pandas path
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        # Unix epoch seconds are always interpreted as UTC.
        return datetime.fromtimestamp(float(value), tz=UTC)
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise EventNormalizeError("occurred_at is invalid") from exc
    else:
        raise EventNormalizeError("occurred_at is invalid")
    if dt.tzinfo is None:
        raise EventNormalizeError("occurred_at timezone is required; use an explicit offset or 'Z'")
    return dt.astimezone(UTC)


def _parse_quantity(value: Any) -> int:
    if value is None:
        return 1
    try:
        quantity = int(value)
    except (TypeError, ValueError) as exc:
        raise EventNormalizeError("quantity must be an integer") from exc
    if quantity < 1:
        raise EventNormalizeError("quantity must be >= 1")
    return quantity


def event_fingerprint(event: NormalizedEvent) -> str:
    return "|".join(
        (
            event.user_id,
            event.item_id,
            event.event_type,
            str(event.quantity),
            event.occurred_at.isoformat(),
        )
    )


def normalize_event(payload: Any) -> NormalizedEvent:
    raw = _as_mapping(payload)
    missing = [key for key in _REQUIRED if raw.get(key) in (None, "")]
    if missing:
        raise EventNormalizeError(f"missing required field(s): {', '.join(missing)}")

    user_id = str(raw["user_id"]).strip()
    item_id = str(raw["item_id"]).strip()
    event_type = str(raw["event_type"]).strip()
    if not user_id or not item_id or not event_type:
        raise EventNormalizeError("user_id, item_id, and event_type must be non-empty")

    event_id = raw.get("event_id") or raw.get("idempotency_key") or str(uuid4())
    return NormalizedEvent(
        user_id=user_id,
        item_id=item_id,
        event_type=event_type,
        quantity=_parse_quantity(raw.get("quantity")),
        occurred_at=parse_occurred_at(raw["occurred_at"]),
        event_id=str(event_id),
    )


def normalize_events(payloads: Sequence[Any]) -> list[NormalizedEvent]:
    return [normalize_event(item) for item in payloads]


def events_to_dataframe(events: Sequence[NormalizedEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=["user_id", "item_id", "event_type", "quantity", "occurred_at"])
    frame = pd.DataFrame(
        [
            {
                "user_id": event.user_id,
                "item_id": event.item_id,
                "event_type": event.event_type,
                "quantity": event.quantity,
                "occurred_at": event.occurred_at,
            }
            for event in events
        ]
    )
    frame["occurred_at"] = pd.to_datetime(frame["occurred_at"], utc=True)
    return frame
