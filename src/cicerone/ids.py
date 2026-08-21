"""External user/item id helpers shared by dataset frames and recommend paths.

``BuiltDataset.interactions`` and raw items frames keep *external* ids (the
same values as ``target_users`` / event ``user_id`` / ``item_id``), under
rectools ``Columns.User`` / ``Columns.Item`` or the plain ``user_id`` /
``item_id`` aliases. Centralize resolution here so schema renames only
touch one module.
"""

from __future__ import annotations

from collections.abc import Hashable

import pandas as pd
from rectools import Columns

from cicerone.dataset import BuiltDataset

USER_ID_COLUMNS: tuple[str, ...] = (Columns.User, "user_id")
ITEM_ID_COLUMNS: tuple[str, ...] = (Columns.Item, "item_id")


def require_id_column(frame: pd.DataFrame, candidates: tuple[str, ...], *, frame_name: str) -> str:
    """Return the first present id column name, or raise a clear ValueError."""
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(
        f"{frame_name} is missing a required id column; expected one of {list(candidates)}, "
        f"got columns {list(frame.columns)}"
    )


def interactions_user_column(interactions: pd.DataFrame) -> str:
    return require_id_column(interactions, USER_ID_COLUMNS, frame_name="interactions")


def interactions_item_column(interactions: pd.DataFrame) -> str:
    return require_id_column(interactions, ITEM_ID_COLUMNS, frame_name="interactions")


def items_id_column(items: pd.DataFrame) -> str:
    return require_id_column(items, ITEM_ID_COLUMNS, frame_name="items")


def interacting_external_user_ids(built: BuiltDataset) -> set[Hashable]:
    """External user IDs with ≥1 interaction (same namespace as ``target_users``)."""
    interactions = built.interactions
    if interactions is None or interactions.empty:
        return set()
    user_col = interactions_user_column(interactions)
    return set(interactions[user_col].astype(str).unique())
