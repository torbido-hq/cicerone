from __future__ import annotations

import pandas as pd


def synthetic_events() -> pd.DataFrame:
    now = pd.Timestamp.utcnow()
    rows = []
    # Enough interactions for LightFM to fit.
    interactions = {
        "u1": ["i1", "i2"],
        "u2": ["i2", "i3"],
        "u3": ["i1", "i3"],
    }
    for user, items in interactions.items():
        for item in items:
            rows.append(
                {
                    "user_id": user,
                    "item_id": item,
                    "event_type": "purchase",
                    "quantity": 1,
                    "occurred_at": now,
                }
            )
    return pd.DataFrame(rows)
