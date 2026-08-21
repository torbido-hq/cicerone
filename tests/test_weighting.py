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
        quantity_scaled_events={"purchase"},
    )
    assert weights.iloc[0] == pytest.approx(0.3)
    assert weights.iloc[1] == pytest.approx(4.0 * math.log1p(3))
    assert pd.isna(weights.iloc[2])
