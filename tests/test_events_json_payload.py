from __future__ import annotations

import pytest

from cicerone.events.json_payload import decode_json_object
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
