from __future__ import annotations

import math

import pandas as pd
import pytest

from cicerone.weighting import event_row_weights


def test_event_row_weights_maps_types_and_scales_quantity():
    event_type = pd.Series(["view", "purchase", "unknown"])
    quantity = pd.Series([2, 3, 9])
    weights = event_row_weights(
        event_type,
        quantity,
        event_weights={"view": 0.3, "purchase": 4.0},
        quantity_scaled_events=["purchase"],
    )
    assert weights.iloc[0] == pytest.approx(0.3)
    assert weights.iloc[1] == pytest.approx(4.0 * math.log1p(3))
    assert pd.isna(weights.iloc[2])


def test_event_row_weights_clips_negative_quantity_on_scaled_types():
    weights = event_row_weights(
        pd.Series(["purchase"]),
        pd.Series([-5]),
        event_weights={"purchase": 4.0},
        quantity_scaled_events=["purchase"],
    )
    assert weights.iloc[0] == pytest.approx(0.0)


def test_event_row_weights_non_numeric_quantity_is_nan_only_when_scaled():
    weights = event_row_weights(
        pd.Series(["purchase", "view"]),
        pd.Series(["n/a", "n/a"]),
        event_weights={"purchase": 4.0, "view": 0.3},
        quantity_scaled_events=["purchase"],
    )
    assert pd.isna(weights.iloc[0])
    assert weights.iloc[1] == pytest.approx(0.3)


def test_event_row_weights_accepts_set_without_copying_semantics():
    weights = event_row_weights(
        pd.Series(["purchase", "view"]),
        pd.Series([3, 1]),
        event_weights={"purchase": 4.0, "view": 0.3},
        quantity_scaled_events={"purchase"},
    )
    assert weights.iloc[0] == pytest.approx(4.0 * math.log1p(3))
    assert weights.iloc[1] == pytest.approx(0.3)
