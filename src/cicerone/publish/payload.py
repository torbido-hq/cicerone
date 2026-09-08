"""Per-user recommendation JSON for the publish sidecar."""

from __future__ import annotations

import json
import math

import pandas as pd

from cicerone.io.recommendation_schema import USER_COLUMN, recommendation_output_columns


def _json_cell(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return value
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if missing is True:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray, str)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def user_recommendation_messages(df: pd.DataFrame) -> list[tuple[str, bytes]]:
    if df is None or df.empty or USER_COLUMN not in df.columns:
        return []
    columns = recommendation_output_columns(df)
    indexed = df[columns].copy()
    indexed[USER_COLUMN] = indexed[USER_COLUMN].astype(str)
    out: list[tuple[str, bytes]] = []
    for user_id, group in indexed.groupby(USER_COLUMN, sort=False):
        records = [
            {key: _json_cell(val) for key, val in row.items()} for row in group.to_dict(orient="records")
        ]
        body = json.dumps(
            {"user_id": str(user_id), "recommendations": records},
            allow_nan=False,
        ).encode("utf-8")
        out.append((str(user_id), body))
    return out
