"""Combine per-strategy recommendation frames (priority or weighted RRF)."""

from __future__ import annotations

import pandas as pd
from rectools import Columns

from cicerone.model.constants import SOURCE_COLUMN, WEIGHT_COLUMN
from cicerone.reasons import SOURCE_CONTRIBS_COLUMN


def _as_rank(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return int(float(str(value)))


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


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
        part["_priority"] = priority
        tagged.append(part)
    combined = pd.concat(tagged, ignore_index=True)
    combined = combined.drop_duplicates(subset=[Columns.User, Columns.Item], keep="first")
    combined = combined.sort_values([Columns.User, "_priority", Columns.Rank])
    combined = combined.groupby(Columns.User, as_index=False).head(top_k)
    combined[SOURCE_CONTRIBS_COLUMN] = [
        _priority_contrib(label, rank)
        for label, rank in zip(combined[SOURCE_COLUMN], combined[Columns.Rank], strict=True)
    ]
    combined[Columns.Rank] = combined.groupby(Columns.User).cumcount() + 1
    drop = [column for column in (WEIGHT_COLUMN, "_priority") if column in combined.columns]
    return combined.drop(columns=drop)


def combine_by_weighted_fusion(
    frames: list[pd.DataFrame], top_k: int, rrf_k: float, source_label_order: list[str]
) -> pd.DataFrame:
    """Weighted RRF: ``weight / (rrf_k + rank)``, sources joined in order."""
    combined = pd.concat(frames, ignore_index=True)
    combined["contribution"] = combined[WEIGHT_COLUMN] / (rrf_k + combined[Columns.Rank])
    combined[Columns.Score] = combined["contribution"]
    combined["_contrib"] = [
        {
            "label": str(label),
            "rank": _as_rank(rank),
            "weight": _as_float(weight),
            "contribution": _as_float(contribution),
        }
        for label, rank, weight, contribution in zip(
            combined[SOURCE_COLUMN],
            combined[Columns.Rank],
            combined[WEIGHT_COLUMN],
            combined["contribution"],
            strict=True,
        )
    ]

    def _join_labels_in_order(labels: pd.Series) -> str:
        present = set(labels)
        return "+".join(label for label in source_label_order if label in present)

    fused = combined.groupby([Columns.User, Columns.Item], as_index=False).agg(
        **{
            Columns.Score: (Columns.Score, "sum"),
            SOURCE_COLUMN: (SOURCE_COLUMN, _join_labels_in_order),
            SOURCE_CONTRIBS_COLUMN: ("_contrib", list),
        }
    )
    order = {label: index for index, label in enumerate(source_label_order)}

    def _sort_contribs(items: object) -> list[dict[str, object]]:
        contribs = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
        contribs.sort(key=lambda item: order.get(str(item.get("label")), len(order)))
        return contribs

    fused[SOURCE_CONTRIBS_COLUMN] = fused[SOURCE_CONTRIBS_COLUMN].map(_sort_contribs)
    fused = fused.sort_values([Columns.User, Columns.Score], ascending=[True, False])
    fused[Columns.Rank] = fused.groupby(Columns.User).cumcount() + 1
    fused = fused.groupby(Columns.User, as_index=False).head(top_k)
    return fused[
        [Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN, SOURCE_CONTRIBS_COLUMN]
    ]
