"""Per-user recommendation JSON for the publish sidecar."""

from __future__ import annotations

import json

import pandas as pd

from cicerone.io.recommendation_schema import USER_COLUMN, recommendation_output_columns


def user_recommendation_messages(df: pd.DataFrame) -> list[tuple[str, bytes]]:
    if df is None or df.empty or USER_COLUMN not in df.columns:
        return []
    columns = recommendation_output_columns(df)
    indexed = df[columns].copy()
    indexed[USER_COLUMN] = indexed[USER_COLUMN].astype(str)
    out: list[tuple[str, bytes]] = []
    for user_id, group in indexed.groupby(USER_COLUMN, sort=False):
        records = json.loads(group.to_json(orient="records"))
        payload = {"user_id": str(user_id), "recommendations": records}
        out.append((str(user_id), json.dumps(payload, separators=(",", ":")).encode("utf-8")))
    return out
