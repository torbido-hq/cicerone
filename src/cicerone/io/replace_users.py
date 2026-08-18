"""Shared helpers for user-scoped recommendation replaces."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from cicerone.io.recommendation_schema import USER_COLUMN


def normalize_replace_user_ids(df: pd.DataFrame, user_ids: Sequence[str]) -> list[str]:
    """Return sorted unique user ids to replace; ``[]`` means no-op.

    Raises ``ValueError`` when ``df`` has rows but ``user_ids`` is empty, lacks
    ``user_id``, or includes users outside ``user_ids``.
    """
    ids = sorted({str(user_id) for user_id in user_ids})
    if not ids:
        if not df.empty:
            raise ValueError("replace_recommendations_for_users requires user_ids when df has rows")
        return []
    if df.empty:
        return ids
    if USER_COLUMN not in df.columns:
        raise ValueError(f"replace_recommendations_for_users df is missing {USER_COLUMN} column")
    extras = set(df[USER_COLUMN].astype(str)) - set(ids)
    if extras:
        raise ValueError(
            f"replace_recommendations_for_users got rows for users outside user_ids: {sorted(extras)}"
        )
    return ids
