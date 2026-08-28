"""Per-user slices of input events/users for the dashboard inspector."""

from __future__ import annotations

import pandas as pd

from cicerone.io.recommendation_schema import USER_COLUMN

OCCURRED_AT_COLUMN = "occurred_at"


def filter_rows_for_user(frame: pd.DataFrame, user_id: str) -> pd.DataFrame:
    if USER_COLUMN not in frame.columns:
        return frame.iloc[0:0].copy()
    matched = frame.loc[frame[USER_COLUMN].astype(str) == str(user_id)]
    return matched.reset_index(drop=True)


_NEWEST_EVENTS_SORT = "_cicerone_newest_events_sort"


def newest_events(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    work = frame
    if OCCURRED_AT_COLUMN in work.columns:
        occurred = pd.to_datetime(work[OCCURRED_AT_COLUMN], utc=True, errors="coerce")
        work = work.assign(**{_NEWEST_EVENTS_SORT: occurred}).sort_values(
            _NEWEST_EVENTS_SORT, ascending=False, na_position="last", kind="mergesort"
        )
        work = work.drop(columns=[_NEWEST_EVENTS_SORT])
    return work.head(limit).reset_index(drop=True)
