"""Per-user slices of input events/users for the dashboard inspector."""

from __future__ import annotations

import pandas as pd

USER_COLUMN = "user_id"
OCCURRED_AT_COLUMN = "occurred_at"


def filter_rows_for_user(frame: pd.DataFrame, user_id: str) -> pd.DataFrame:
    if USER_COLUMN not in frame.columns:
        return pd.DataFrame()
    matched = frame.loc[frame[USER_COLUMN].astype(str) == str(user_id)]
    return matched.reset_index(drop=True)


def newest_events(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    work = frame
    if OCCURRED_AT_COLUMN in work.columns:
        occurred = pd.to_datetime(work[OCCURRED_AT_COLUMN], utc=True, errors="coerce")
        work = work.assign(_sort=occurred).sort_values("_sort", ascending=False, na_position="last")
        work = work.drop(columns=["_sort"])
    return work.head(limit).reset_index(drop=True)
