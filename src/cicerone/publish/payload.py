"""Per-user recommendation JSON for the publish sidecar."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from contextlib import suppress
from hashlib import sha256

import pandas as pd

from cicerone.io.recommendation_schema import USER_COLUMN, recommendation_output_columns

RecommendationMessage = tuple[str, bytes, str]


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


def _recommendation_message(user_id: str, group: pd.DataFrame | None) -> RecommendationMessage:
    recommendations = (
        []
        if group is None
        else [
            {key: _json_cell(value) for key, value in row.items()} for row in group.to_dict(orient="records")
        ]
    )
    content = {"user_id": user_id, "recommendations": recommendations}
    message_id = sha256(
        json.dumps(content, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    body = json.dumps({**content, "message_id": message_id}, allow_nan=False).encode()
    return user_id, body, message_id


class _RecommendationMessages(Sequence[RecommendationMessage]):
    def __init__(self, df: pd.DataFrame, user_ids: Sequence[str] | None) -> None:
        self._grouped = None
        if df is None or df.empty or USER_COLUMN not in df.columns:
            self._order = (
                [] if user_ids is None else list(dict.fromkeys(str(user_id) for user_id in user_ids))
            )
            return
        columns = recommendation_output_columns(df)
        indexed = df[columns].copy()
        indexed[USER_COLUMN] = indexed[USER_COLUMN].astype(str)
        self._grouped = indexed.groupby(USER_COLUMN, sort=False)
        if user_ids is None:
            self._order = [str(user_id) for user_id in self._grouped.groups]
            return
        self._order = list(dict.fromkeys(str(user_id) for user_id in user_ids))

    def __len__(self) -> int:
        return len(self._order)

    def __getitem__(self, index: int) -> RecommendationMessage:
        user_id = self._order[index]
        group = None
        if self._grouped is not None and user_id in self._grouped.groups:
            group = self._grouped.get_group(user_id)
        return _recommendation_message(user_id, group)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, list):
            return list(self) == other
        return NotImplemented


def user_recommendation_messages(
    df: pd.DataFrame,
    *,
    user_ids: Sequence[str] | None = None,
) -> _RecommendationMessages:
    return _RecommendationMessages(df, user_ids)
