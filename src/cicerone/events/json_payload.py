"""Decode broker payloads and turn deliveries into normalized events."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from cicerone.events.base import NormalizedEvent
from cicerone.events.normalize import (
    EventNormalizeError,
    normalize_event,
    reject_object_fields,
    supplied_event_id,
)


def delivery_event(raw: Any, fallback_id: str) -> NormalizedEvent:
    payload = decode_json_object(raw)
    reject_object_fields(payload)
    if supplied_event_id(payload) is None:
        payload["event_id"] = fallback_id
    return normalize_event(payload)


def decode_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, (bytes, bytearray)):
        if not raw:
            raise EventNormalizeError("event must be a JSON object")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EventNormalizeError("event must be a JSON object") from exc
    elif isinstance(raw, str):
        text = raw
    else:
        raise EventNormalizeError("event must be a JSON object")
    try:
        parsed = json.loads(text)
    # ValueError covers JSONDecodeError and the int digit limit.
    except (ValueError, RecursionError) as exc:
        raise EventNormalizeError("event must be a JSON object") from exc
    if not isinstance(parsed, Mapping):
        raise EventNormalizeError("event must be a JSON object")
    return dict(parsed)
