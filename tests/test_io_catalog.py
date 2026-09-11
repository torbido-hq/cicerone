from __future__ import annotations

import math

import pandas as pd
import pytest

from cicerone.io.catalog import (
    jsonable_row,
    normalize_event_row,
    require_id,
    user_row_or_none,
)


def test_require_id_rejects_blank():
    with pytest.raises(ValueError, match="user_id"):
        require_id({"user_id": "  "}, "user_id")


def test_normalize_event_row_defaults_quantity():
    row = normalize_event_row(
        {
            "user_id": "u1",
            "item_id": "i1",
            "event_type": "purchase",
            "occurred_at": "2026-09-11T12:00:00Z",
        }
    )
    assert row["quantity"] == 1


def test_jsonable_row_coerces_numpy_and_nan():
    payload = jsonable_row({"item_id": pd.Series(["i1"]).iloc[0], "score": math.nan})
    assert payload["item_id"] == "i1"
    assert payload["score"] is None


def test_user_row_or_none_missing():
    assert user_row_or_none(pd.DataFrame([{"user_id": "u2"}]), "u1") is None
