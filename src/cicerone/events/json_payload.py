"""Decode broker payloads into JSON objects for event ingest."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from cicerone.events.normalize import EventNormalizeError


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
    except json.JSONDecodeError as exc:
        raise EventNormalizeError("event must be a JSON object") from exc
    if not isinstance(parsed, Mapping):
        raise EventNormalizeError("event must be a JSON object")
    return dict(parsed)
