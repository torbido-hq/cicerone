"""Recommendation row schema constants and shared SQL name helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cicerone.io.options import sql_identifier

USER_COLUMN = "user_id"
ITEM_COLUMN = "item_id"
RANK_COLUMN = "rank"
SCORE_COLUMN = "score"
SOURCE_COLUMN = "source"
RECOMMENDATION_COLUMNS: tuple[str, ...] = (
    USER_COLUMN,
    ITEM_COLUMN,
    RANK_COLUMN,
    SCORE_COLUMN,
    SOURCE_COLUMN,
)


def recommendations_sql_names(
    options: Mapping[str, Any],
    *,
    default_table: str,
) -> tuple[str, str, str]:
    """Return ``(table, select_columns, user_column)`` as validated SQL identifiers."""
    table = sql_identifier(
        options.get("recommendations_table", default_table),
        option="recommendations_table",
    )
    columns = ", ".join(
        sql_identifier(column, option="recommendations_column") for column in RECOMMENDATION_COLUMNS
    )
    user_col = sql_identifier(USER_COLUMN, option="recommendations_column")
    return table, columns, user_col
