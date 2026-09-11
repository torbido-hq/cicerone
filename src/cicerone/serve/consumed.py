"""Live consumed-item hide for serve GET (input history + incremental overlay)."""

from __future__ import annotations

import logging
import threading

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

_HISTORY_FAIL_LOCK = threading.Lock()
_history_fail_logged = False
_MISSING_S3_CODES = frozenset({"NoSuchKey", "404", "NotFound"})


def _missing_history(exc: BaseException) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return response.get("Error", {}).get("Code") in _MISSING_S3_CODES
    return False


def _log_history_failure(user_id: str) -> None:
    global _history_fail_logged
    with _HISTORY_FAIL_LOCK:
        first = not _history_fail_logged
        _history_fail_logged = True
    if first:
        logger.exception(
            "Failed to read [input] history for user_id=%r; hide uses overlay only",
            user_id,
        )
        return
    logger.debug(
        "Failed to read [input] history for user_id=%r; hide uses overlay only",
        user_id,
        exc_info=True,
    )


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
    except Exception as exc:
        if not _missing_history(exc):
            _log_history_failure(user_id)
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
