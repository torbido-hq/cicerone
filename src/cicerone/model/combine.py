"""Combine per-strategy recommendation frames (priority or weighted RRF)."""

from __future__ import annotations

import pandas as pd
from rectools import Columns

from cicerone.model.constants import SOURCE_COLUMN
from cicerone.ranking import WEIGHT_COLUMN, fuse_weighted_rrf, keep_top_k, renumber_ranks
from cicerone.reasons import PRIORITY_BLOCK_COLUMN, SOURCE_CONTRIBS_COLUMN


def _as_rank(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return int(float(str(value)))


def _priority_contrib(label: object, rank: object) -> list[dict[str, object]]:
    return [
        {
            "label": str(label),
            "rank": _as_rank(rank),
            "weight": 1.0,
            "contribution": None,
        }
    ]


def combine_by_priority(frames: list[pd.DataFrame], top_k: int) -> pd.DataFrame:
    """Fill top-K in list order; earlier strategies win duplicate (user, item)."""
    tagged = []
    for priority, frame in enumerate(frames):
        part = frame.copy()
        part[PRIORITY_BLOCK_COLUMN] = priority
        tagged.append(part)
    combined = pd.concat(tagged, ignore_index=True)
    combined = combined.drop_duplicates(subset=[Columns.User, Columns.Item], keep="first")
    combined = combined.sort_values([Columns.User, PRIORITY_BLOCK_COLUMN, Columns.Rank], kind="mergesort")
    combined = keep_top_k(combined, top_k)
    combined[SOURCE_CONTRIBS_COLUMN] = [
        _priority_contrib(label, rank)
        for label, rank in zip(combined[SOURCE_COLUMN], combined[Columns.Rank], strict=True)
    ]
    return renumber_ranks(combined).drop(columns=[WEIGHT_COLUMN], errors="ignore")


def combine_by_weighted_fusion(
    frames: list[pd.DataFrame], top_k: int, rrf_k: float, source_label_order: list[str]
) -> pd.DataFrame:
    """Weighted RRF: ``weight / (rrf_k + rank)``, sources joined in order."""
    combined = pd.concat(frames, ignore_index=True)

    def _join_labels_in_order(labels: pd.Series) -> str:
        present = set(labels)
        return "+".join(label for label in source_label_order if label in present)

    return fuse_weighted_rrf(
        combined,
        rrf_k=rrf_k,
        top_k=top_k,
        source_label=_join_labels_in_order,
        source_order=source_label_order,
    )
