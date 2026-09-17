"""Job-time popular / latest / item-neighbor snapshots for serve surfaces."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence

import pandas as pd

from cicerone.blending import LATEST_SOURCE, POPULAR_SOURCE, resolve_latest_date_column
from cicerone.feature_config import DEFAULT_LATEST_DATE_COLUMNS
from cicerone.io.recommendation_schema import ITEM_COLUMN, RANK_COLUMN, SCORE_COLUMN, SOURCE_COLUMN
from cicerone.values import is_missing

NEIGHBOR_ITEM_COLUMN = "neighbor_id"
POPULAR_FILENAME = "popular.parquet"
LATEST_FILENAME = "latest.parquet"
NEIGHBORS_FILENAME = "item_neighbors.parquet"
SURFACES_STAMP_FILENAME = "surfaces_stamp.json"
DEFAULT_NEIGHBOR_HISTORY_CAP = 200
SURFACE_FILES: tuple[str, str, str] = (POPULAR_FILENAME, LATEST_FILENAME, NEIGHBORS_FILENAME)


def surfaces_stamp_sha256(files: Sequence[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for filename, payload in files:
        digest.update(filename.encode())
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def surfaces_stamp_payload(files: Sequence[tuple[str, bytes]]) -> bytes:
    return json.dumps({"sha256": surfaces_stamp_sha256(files)}).encode()


SURFACE_COLUMNS: tuple[str, ...] = (ITEM_COLUMN, RANK_COLUMN, SCORE_COLUMN, SOURCE_COLUMN)
NEIGHBOR_COLUMNS: tuple[str, ...] = (ITEM_COLUMN, NEIGHBOR_ITEM_COLUMN, RANK_COLUMN, SCORE_COLUMN)


def empty_surface_frame(*, source: str) -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(SURFACE_COLUMNS))
    frame[SOURCE_COLUMN] = pd.Series(dtype=object)
    if source:
        frame[SOURCE_COLUMN] = frame[SOURCE_COLUMN].astype(object)
    return frame


def empty_neighbors_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(NEIGHBOR_COLUMNS))


def _clean_ids(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].map(lambda value: "" if is_missing(value) else str(value).strip())


def popular_from_events(events: pd.DataFrame, k: int) -> pd.DataFrame:
    """Global popularity: distinct users per item, then event count."""
    if k < 1 or events.empty or ITEM_COLUMN not in events.columns:
        return empty_surface_frame(source=POPULAR_SOURCE)
    frame = events[[ITEM_COLUMN]].copy()
    frame[ITEM_COLUMN] = _clean_ids(events, ITEM_COLUMN)
    keep = frame[ITEM_COLUMN] != ""
    if "user_id" in events.columns:
        frame["user_id"] = _clean_ids(events, "user_id")
        keep = keep & (frame["user_id"] != "")
    frame = frame.loc[keep]
    if frame.empty:
        return empty_surface_frame(source=POPULAR_SOURCE)
    if "user_id" in frame.columns:
        users = frame.drop_duplicates().groupby(ITEM_COLUMN, sort=False).size().rename(SCORE_COLUMN)
        event_count = frame.groupby(ITEM_COLUMN, sort=False).size().rename("_n_events")
        scored = pd.concat([users, event_count], axis=1).reset_index()
        scored = scored.sort_values(
            [SCORE_COLUMN, "_n_events", ITEM_COLUMN],
            ascending=[False, False, True],
            kind="mergesort",
        )
    else:
        scored = frame.groupby(ITEM_COLUMN, sort=False).size().rename(SCORE_COLUMN).reset_index()
        scored = scored.sort_values([SCORE_COLUMN, ITEM_COLUMN], ascending=[False, True], kind="mergesort")
    scored = scored.head(k).reset_index(drop=True)
    scored[RANK_COLUMN] = range(1, len(scored) + 1)
    scored[SOURCE_COLUMN] = POPULAR_SOURCE
    return scored[list(SURFACE_COLUMNS)]


def latest_from_items(
    items: pd.DataFrame | None,
    k: int,
    date_columns: Sequence[str] = DEFAULT_LATEST_DATE_COLUMNS,
) -> pd.DataFrame:
    if k < 1 or items is None or items.empty or ITEM_COLUMN not in items.columns:
        return empty_surface_frame(source=LATEST_SOURCE)
    date_column = resolve_latest_date_column(items, date_columns)
    if date_column is None:
        return empty_surface_frame(source=LATEST_SOURCE)
    frame = items[[ITEM_COLUMN, date_column]].copy()
    frame[ITEM_COLUMN] = _clean_ids(items, ITEM_COLUMN)
    frame = frame.loc[frame[ITEM_COLUMN] != ""]
    if frame.empty:
        return empty_surface_frame(source=LATEST_SOURCE)
    frame["_date"] = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
    frame = frame.dropna(subset=["_date"])
    if frame.empty:
        return empty_surface_frame(source=LATEST_SOURCE)
    frame = frame.sort_values(["_date", ITEM_COLUMN], ascending=[False, True], kind="mergesort")
    frame = frame.drop_duplicates(subset=[ITEM_COLUMN], keep="first").head(k).reset_index(drop=True)
    frame[RANK_COLUMN] = range(1, len(frame) + 1)
    frame[SCORE_COLUMN] = [float(len(frame) - i + 1) for i in frame[RANK_COLUMN]]
    frame[SOURCE_COLUMN] = LATEST_SOURCE
    return frame[list(SURFACE_COLUMNS)]


def neighbors_from_events(
    events: pd.DataFrame,
    k: int,
    *,
    history_cap: int = DEFAULT_NEIGHBOR_HISTORY_CAP,
) -> pd.DataFrame:
    """Item-item cosine on binary user overlap (users who interacted with both)."""
    if k < 1 or events.empty or ITEM_COLUMN not in events.columns or "user_id" not in events.columns:
        return empty_neighbors_frame()
    columns = ["user_id", ITEM_COLUMN]
    if "occurred_at" in events.columns:
        columns.append("occurred_at")
    pairs = events[columns].copy()
    pairs["user_id"] = _clean_ids(pairs, "user_id")
    pairs[ITEM_COLUMN] = _clean_ids(pairs, ITEM_COLUMN)
    pairs = pairs.loc[(pairs["user_id"] != "") & (pairs[ITEM_COLUMN] != "")]
    if "occurred_at" in pairs.columns:
        pairs["_at"] = pd.to_datetime(pairs["occurred_at"], errors="coerce", utc=True)
        pairs = pairs.sort_values("_at", ascending=False, kind="mergesort")
    pairs = pairs.drop_duplicates(subset=["user_id", ITEM_COLUMN], keep="first")
    if history_cap >= 1:
        pairs = pairs.groupby("user_id", sort=False).head(history_cap)
    if pairs.empty:
        return empty_neighbors_frame()
    item_users: dict[str, set[str]] = defaultdict(set)
    for user_id, item_id in zip(pairs["user_id"], pairs[ITEM_COLUMN], strict=True):
        item_users[item_id].add(user_id)
    if len(item_users) < 2:
        return empty_neighbors_frame()
    inverted: dict[str, set[str]] = defaultdict(set)
    for item_id, users in item_users.items():
        for user_id in users:
            inverted[user_id].add(item_id)
    rows: list[dict[str, object]] = []
    for item_id, users in item_users.items():
        shared: dict[str, int] = defaultdict(int)
        for user_id in users:
            for other in inverted[user_id]:
                if other != item_id:
                    shared[other] += 1
        if not shared:
            continue
        item_n = len(users)
        ranked = sorted(
            ((other, count / ((item_n * len(item_users[other])) ** 0.5)) for other, count in shared.items()),
            key=lambda pair: (-pair[1], pair[0]),
        )[:k]
        for rank, (neighbor_id, score) in enumerate(ranked, start=1):
            rows.append(
                {
                    ITEM_COLUMN: item_id,
                    NEIGHBOR_ITEM_COLUMN: neighbor_id,
                    RANK_COLUMN: rank,
                    SCORE_COLUMN: float(score),
                }
            )
    if not rows:
        return empty_neighbors_frame()
    return pd.DataFrame(rows, columns=list(NEIGHBOR_COLUMNS))
