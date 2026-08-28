"""Serialize and parse the optional recommendation ``reasons`` payload.

Serve-safe: no rectools / sklearn / model imports.
"""

from __future__ import annotations

import json
from typing import Any

from cicerone.serve_schemas import RecommendationReasons
from cicerone.values import is_missing

SOURCE_CONTRIBS_COLUMN = "_source_contribs"
BOOST_HITS_COLUMN = "_boost_hits"


def dump_reasons(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def source_reasons_payload(
    label: str,
    *,
    rank: int | None = None,
    weight: float | None = 1.0,
    contribution: float | None = None,
) -> dict[str, Any]:
    return {
        "sources": [
            {
                "label": str(label),
                "rank": rank,
                "weight": weight,
                "contribution": contribution,
            }
        ],
        "boosts": [],
        "similar_items": [],
        "matched_attributes": [],
    }


def dump_source_reasons(
    label: str,
    *,
    rank: int | None = None,
    weight: float | None = 1.0,
    contribution: float | None = None,
) -> str:
    return dump_reasons(source_reasons_payload(label, rank=rank, weight=weight, contribution=contribution))


def parse_reasons(value: object) -> RecommendationReasons | None:
    if is_missing(value):
        return None
    try:
        if value == "":
            return None
    except (ValueError, TypeError):
        return None
    if isinstance(value, RecommendationReasons):
        return value
    payload: object = value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    try:
        return RecommendationReasons.model_validate(payload)
    except (TypeError, ValueError):
        return None
