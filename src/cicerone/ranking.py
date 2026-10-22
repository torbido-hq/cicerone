"""Per-user ranking of recommendation frames."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from rectools import Columns


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
