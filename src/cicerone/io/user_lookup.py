"""Per-user slices of input events/users for the dashboard inspector."""

from __future__ import annotations

from collections.abc import Sequence

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


def newest_events_by_user(
    frame: pd.DataFrame, user_ids: Sequence[str], limit: int
) -> dict[str, pd.DataFrame]:
    ids = [str(user_id) for user_id in user_ids]
    empty = frame.iloc[0:0]
    if frame.empty or USER_COLUMN not in frame.columns:
        return {user_id: empty.copy() for user_id in ids}
    work = frame.copy()
    work[USER_COLUMN] = work[USER_COLUMN].astype(str)
    grouped = {str(key): group for key, group in work.groupby(USER_COLUMN, sort=False)}
    return {user_id: newest_events(grouped.get(user_id, empty), limit) for user_id in ids}
