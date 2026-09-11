"""Live consumed-item hide for serve GET (input history + incremental overlay)."""

from __future__ import annotations

import logging

import pandas as pd

from cicerone.events.consumed import ConsumedOverlay
from cicerone.io.base import UserHistoryReader
from cicerone.io.recommendation_schema import ITEM_COLUMN, USER_COLUMN

__all__ = [
    "ConsumedOverlay",
    "consumed_item_ids",
    "drop_consumed",
    "drop_item_ids",
    "merge_fill",
]

logger = logging.getLogger(__name__)


def consumed_item_ids(
    user_id: str,
    *,
    history: UserHistoryReader | None,
    overlay: ConsumedOverlay | None,
    lookback: int,
) -> set[str]:
    ids: set[str] = set()
    if overlay is not None:
        ids |= overlay.item_ids(user_id)
    if history is None or lookback < 1:
        return ids
    try:
        events = history.get_events_for_user(user_id, lookback)
    except Exception:
        logger.exception("Failed to read consumed items for user_id=%r; serving without hide", user_id)
        return ids
    if events.empty or ITEM_COLUMN not in events.columns:
        return ids
    ids.update(events[ITEM_COLUMN].astype(str).tolist())
    return ids


def drop_consumed(frame: pd.DataFrame, consumed: set[str]) -> pd.DataFrame:
    if frame.empty or not consumed or ITEM_COLUMN not in frame.columns:
        return frame
    keep = ~frame[ITEM_COLUMN].astype(str).isin(consumed)
    return frame.loc[keep].reset_index(drop=True)


def drop_item_ids(frame: pd.DataFrame, item_ids: set[str]) -> pd.DataFrame:
    return drop_consumed(frame, item_ids)


def merge_fill(
    primary: pd.DataFrame,
    filler: pd.DataFrame,
    *,
    k: int,
    exclude: set[str],
) -> pd.DataFrame:
    """Append filler rows not already in primary / exclude, then truncate to ``k``."""
    seen = set(exclude)
    parts: list[pd.DataFrame] = []
    if not primary.empty and ITEM_COLUMN in primary.columns:
        parts.append(primary)
        seen.update(primary[ITEM_COLUMN].astype(str).tolist())
    if not filler.empty and ITEM_COLUMN in filler.columns:
        extra = drop_consumed(filler, seen)
        if not extra.empty:
            parts.append(extra)
            seen.update(extra[ITEM_COLUMN].astype(str).tolist())
    if not parts:
        return primary.iloc[0:0] if not primary.empty else filler.iloc[0:0]
    merged = pd.concat(parts, ignore_index=True)
    if USER_COLUMN in merged.columns and not primary.empty and USER_COLUMN in primary.columns:
        merged[USER_COLUMN] = str(primary.iloc[0][USER_COLUMN])
    return merged.head(k).reset_index(drop=True)
