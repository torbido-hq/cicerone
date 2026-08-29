from __future__ import annotations

import pytest
from pydantic import ValidationError

from cicerone.reasons import dump_reasons, parse_reasons, source_reasons_payload
from cicerone.serve_schemas import RecommendationReasons


def test_parse_reasons_none_and_blank():
    assert parse_reasons(None) is None
    assert parse_reasons("") is None
    assert parse_reasons("not-json") is None
    assert parse_reasons([]) is None


def test_parse_reasons_round_trip():
    payload = source_reasons_payload("personalized", rank=2, weight=0.8, contribution=0.01)
    payload["similar_items"] = [{"item_id": "i9", "score": 0.5}]
    payload["matched_attributes"] = [{"column": "style", "value": "lager"}]
    parsed = parse_reasons(dump_reasons(payload))
    assert isinstance(parsed, RecommendationReasons)
    assert parsed.sources[0].label == "personalized"
    assert parsed.sources[0].rank == 2
    assert parsed.similar_items[0].item_id == "i9"
    assert parsed.matched_attributes[0].value == "lager"


def test_parse_reasons_bytes_and_invalid_payload():
    payload = source_reasons_payload("latest")
    parsed = parse_reasons(dump_reasons(payload).encode("utf-8"))
    assert parsed is not None
    assert parsed.sources[0].label == "latest"
    assert parse_reasons(b"{") is None
    assert parse_reasons({"sources": "personalized"}) is None


def test_parse_reasons_equality_raises():
    class _BoomEq:
        def __eq__(self, other: object) -> bool:
            raise TypeError("nope")

    assert parse_reasons(_BoomEq()) is None


def test_dump_reasons_rejects_invalid_payload():
    with pytest.raises(ValidationError):
        dump_reasons({"boosts": [{"name": "category"}]})


def test_parse_reasons_dict():
    parsed = parse_reasons({"sources": [{"label": "latest"}], "boosts": []})
    assert parsed is not None
    assert parsed.sources[0].label == "latest"
    assert parsed.boosts == []
