"""Combine per-strategy recommendation frames (priority or weighted RRF)."""

from __future__ import annotations

import pandas as pd
from rectools import Columns

from cicerone.model.constants import SOURCE_COLUMN, WEIGHT_COLUMN


def combine_by_priority(frames: list[pd.DataFrame], top_k: int) -> pd.DataFrame:
    """Fill top-K in list order; earlier strategies win duplicate (user, item)."""
    tagged = []
    for priority, frame in enumerate(frames):
        part = frame.copy()
        part["_priority"] = priority
        tagged.append(part)
    combined = pd.concat(tagged, ignore_index=True)
    combined = combined.drop_duplicates(subset=[Columns.User, Columns.Item], keep="first")
    combined = combined.sort_values([Columns.User, "_priority", Columns.Rank])
    combined = combined.groupby(Columns.User, as_index=False).head(top_k)
    combined[Columns.Rank] = combined.groupby(Columns.User).cumcount() + 1
    return combined.drop(columns=[WEIGHT_COLUMN, "_priority"])


def combine_by_weighted_fusion(
    frames: list[pd.DataFrame], top_k: int, rrf_k: float, source_label_order: list[str]
) -> pd.DataFrame:
    """Weighted RRF: ``weight / (rrf_k + rank)``, sources joined in order."""
    combined = pd.concat(frames, ignore_index=True)
    combined[Columns.Score] = combined[WEIGHT_COLUMN] / (rrf_k + combined[Columns.Rank])

    def _join_labels_in_order(labels: pd.Series) -> str:
        present = set(labels)
        return "+".join(label for label in source_label_order if label in present)

    fused = combined.groupby([Columns.User, Columns.Item], as_index=False).agg(
        **{
            Columns.Score: (Columns.Score, "sum"),
            SOURCE_COLUMN: (SOURCE_COLUMN, _join_labels_in_order),
        }
    )
    fused = fused.sort_values(
        [Columns.User, Columns.Score, Columns.Item],
        ascending=[True, False, True],
        kind="mergesort",
    )
    fused[Columns.Rank] = fused.groupby(Columns.User).cumcount() + 1
    fused = fused.groupby(Columns.User, as_index=False).head(top_k)
    return fused[[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN]]
