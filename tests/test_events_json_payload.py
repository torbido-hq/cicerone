from __future__ import annotations

import json

import pytest
from support.events import UNPARSABLE_JSON_BODIES, event_payload

from cicerone.events.json_payload import decode_json_object, delivery_event
from cicerone.events.normalize import EventNormalizeError


def test_decode_mapping_passthrough():
    assert decode_json_object({"user_id": "u1"}) == {"user_id": "u1"}


def test_decode_bytes_and_str():
    assert decode_json_object(b'{"a": 1}') == {"a": 1}
    assert decode_json_object(bytearray(b'{"a": 1}')) == {"a": 1}
    assert decode_json_object('{"a": 1}') == {"a": 1}


def test_decode_rejects_invalid():
    with pytest.raises(EventNormalizeError):
        decode_json_object(b"")
    with pytest.raises(EventNormalizeError):
        decode_json_object("not-json")
    with pytest.raises(EventNormalizeError):
        decode_json_object("[1]")
    with pytest.raises(EventNormalizeError):
        decode_json_object(1)
    with pytest.raises(EventNormalizeError, match="JSON object"):
        decode_json_object(b"\xff")


@pytest.mark.usefixtures("default_int_digit_limit")
@pytest.mark.parametrize("body", UNPARSABLE_JSON_BODIES, ids=("huge_int", "deep_nesting"))
def test_decode_rejects_bodies_the_parser_cannot_read(body):
    with pytest.raises(EventNormalizeError, match="JSON object"):
        decode_json_object(body)
    with pytest.raises(EventNormalizeError, match="JSON object"):
        decode_json_object(body.decode())


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        ({}, "fallback"),
        ({"event_id": None}, "fallback"),
        ({"event_id": ""}, "fallback"),
        ({"event_id": "  "}, "fallback"),
        ({"event_id": 0}, "fallback"),
        ({"event_id": " e1 "}, "e1"),
        ({"event_id": 12}, "12"),
        ({"idempotency_key": "ik"}, "ik"),
        ({"event_id": 0, "idempotency_key": "ik"}, "ik"),
        ({"event_id": "e1", "idempotency_key": "ik"}, "e1"),
    ],
)
def test_delivery_event_id(ids, expected):
    payload = event_payload()
    payload.pop("event_id")
    payload.update(ids)
    original = dict(payload)
    event = delivery_event(payload, "fallback")
    assert event.event_id == expected
    assert event.generated_event_id is False
    assert payload == original
    assert delivery_event(json.dumps(payload).encode(), "fallback").event_id == expected


@pytest.mark.parametrize("raw", [b"not-json", b"[1]", {"user_id": "u1"}])
def test_delivery_event_rejects_invalid(raw):
    with pytest.raises(EventNormalizeError):
        delivery_event(raw, "fallback")
