"""Per-user recommendation JSON for the publish sidecar."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from contextlib import suppress
from hashlib import sha256

import pandas as pd

from cicerone.io.recommendation_schema import USER_COLUMN, recommendation_output_columns


def _json_cell(value: object) -> object:
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray, str, pd.Timestamp)):
        with suppress(ValueError, AttributeError):
            value = value.item()
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
    return value


def user_recommendation_messages(
    df: pd.DataFrame,
    *,
    user_ids: Sequence[str] | None = None,
) -> list[tuple[str, bytes, str]]:
    by_user: dict[str, list[dict[str, object]]] = {}
    if df is not None and not df.empty and USER_COLUMN in df.columns:
        columns = recommendation_output_columns(df)
        indexed = df[columns].copy()
        indexed[USER_COLUMN] = indexed[USER_COLUMN].astype(str)
        for user_id, group in indexed.groupby(USER_COLUMN, sort=False):
            by_user[str(user_id)] = [
                {key: _json_cell(val) for key, val in row.items()} for row in group.to_dict(orient="records")
            ]
    ordered = list(by_user) if user_ids is None else list(dict.fromkeys(str(user_id) for user_id in user_ids))
    out: list[tuple[str, bytes, str]] = []
    for user_id in ordered:
        content = {"user_id": user_id, "recommendations": by_user.get(user_id, [])}
        message_id = sha256(
            json.dumps(content, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        body = json.dumps({**content, "message_id": message_id}, allow_nan=False).encode()
        out.append((user_id, body, message_id))
    return out
