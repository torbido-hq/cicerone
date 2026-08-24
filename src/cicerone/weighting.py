"""Per-row event weights shared by training interactions and incremental popular ranking."""

from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np
import pandas as pd


def event_row_weights(
    event_type: pd.Series,
    quantity: pd.Series,
    *,
    event_weights: Mapping[str, float],
    quantity_scaled_events: Collection[str],
) -> pd.Series:
    """``event_weights[type]`` times ``log1p(qty)`` when the type is quantity-scaled.

    Unknown types are NaN. Quantity is coerced to numeric; negatives clip to 0.
    Non-numeric quantity is NaN, so a scaled type yields NaN (unscaled types
    ignore quantity).
    """
    base = event_type.map(event_weights)
    qty = pd.to_numeric(quantity, errors="coerce")
    scaled = (
        quantity_scaled_events
        if isinstance(quantity_scaled_events, (set, frozenset))
        else set(quantity_scaled_events)
    )
    multiplier = np.where(
        event_type.isin(scaled),
        np.log1p(qty.clip(lower=0)),
        1.0,
    )
    return base * multiplier
