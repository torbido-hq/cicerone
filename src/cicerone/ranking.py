"""Per-user ranking of recommendation frames."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd
from rectools import Columns

from cicerone.io.recommendation_schema import SOURCE_COLUMN
from cicerone.reasons import SOURCE_CONTRIBS_COLUMN

WEIGHT_COLUMN = "_weight"  # internal-only; dropped before returning to callers


def keep_top_k(frame: pd.DataFrame, top_k: int | None) -> pd.DataFrame:
    """Each user's first ``top_k`` rows in frame order (all rows when ``None``)."""
    return frame if top_k is None else frame.groupby(Columns.User, as_index=False).head(top_k)


def renumber_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    """Ranks 1..n per user, in frame order."""
    frame[Columns.Rank] = frame.groupby(Columns.User).cumcount() + 1
    return frame


def rerank_by_score(
    frame: pd.DataFrame, top_k: int | None = None, *, first: Sequence[str] = ()
) -> pd.DataFrame:
    """Per user: order by ``first``, then score (high first) and item; renumber; keep ``top_k``."""
    keys = [Columns.User, *first, Columns.Score, Columns.Item]
    ranked = frame.sort_values(keys, ascending=[key != Columns.Score for key in keys], kind="mergesort")
    return keep_top_k(renumber_ranks(ranked), top_k)


def drop_seen_pairs(frame: pd.DataFrame, seen: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``frame`` whose (user, item) pair does not appear in ``seen``."""
    if frame.empty or seen.empty:
        return frame
    user, item = Columns.User, Columns.Item
    if user not in frame.columns or item not in frame.columns:
        return frame
    if user not in seen.columns or item not in seen.columns:
        return frame
    pairs = seen.loc[:, [user, item]].drop_duplicates()
    merged = frame.merge(pairs, on=[user, item], how="left", indicator=True)
    return merged.loc[merged["_merge"] == "left_only"].drop(columns=["_merge"]).reset_index(drop=True)


def fuse_weighted_rrf(
    frame: pd.DataFrame,
    *,
    rrf_k: float,
    top_k: int,
    source_label: Callable[[pd.Series], str],
    source_order: Sequence[str],
) -> pd.DataFrame:
    """Sum ``weight / (rrf_k + rank)`` per (user, item) from each row's ``WEIGHT_COLUMN``, then rerank."""
    ranks = frame[Columns.Rank].to_numpy(dtype=float)
    weight_vals = frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    contributions = weight_vals / (rrf_k + ranks)
    parts = frame[[Columns.User, Columns.Item, SOURCE_COLUMN]].assign(
        **{
            Columns.Score: contributions,
            SOURCE_CONTRIBS_COLUMN: [
                {
                    "label": str(label),
                    "rank": int(rank),
                    "weight": float(weight),
                    "contribution": float(value),
                }
                for label, rank, weight, value in zip(
                    frame[SOURCE_COLUMN], ranks, weight_vals, contributions, strict=True
                )
            ],
        }
    )
    fused = parts.groupby([Columns.User, Columns.Item], as_index=False, sort=False).agg(
        **{
            Columns.Score: (Columns.Score, "sum"),
            SOURCE_COLUMN: (SOURCE_COLUMN, source_label),
            SOURCE_CONTRIBS_COLUMN: (SOURCE_CONTRIBS_COLUMN, list),
        }
    )
    order = {label: index for index, label in enumerate(source_order)}
    fused[SOURCE_CONTRIBS_COLUMN] = fused[SOURCE_CONTRIBS_COLUMN].map(
        lambda contribs: sorted(contribs, key=lambda item: order.get(item["label"], len(order)))
    )
    fused = rerank_by_score(fused, top_k)
    return fused[
        [Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN, SOURCE_CONTRIBS_COLUMN]
    ].reset_index(drop=True)
