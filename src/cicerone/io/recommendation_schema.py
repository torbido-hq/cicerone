"""Recommendation row schema constants and shared SQL name helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cicerone.io.options import sql_identifier
from cicerone.values import is_missing

USER_COLUMN = "user_id"
ITEM_COLUMN = "item_id"
RANK_COLUMN = "rank"
SCORE_COLUMN = "score"
SOURCE_COLUMN = "source"
REASONS_COLUMN = "reasons"
VARIANT_COLUMN = "variant"
RECOMMENDATION_COLUMNS: tuple[str, ...] = (
    USER_COLUMN,
    ITEM_COLUMN,
    RANK_COLUMN,
    SCORE_COLUMN,
    SOURCE_COLUMN,
)
_OPTIONAL_OUTPUT_COLUMNS: tuple[str, ...] = (REASONS_COLUMN, VARIANT_COLUMN)


FALLBACK_VARIANT = "control"


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


def recommendation_output_columns(frame: Any) -> list[str]:
    """Required columns plus optional ``reasons`` / ``variant`` when present."""
    columns = list(RECOMMENDATION_COLUMNS)
    if not hasattr(frame, "columns"):
        return columns
    for column in _OPTIONAL_OUTPUT_COLUMNS:
        if column in frame.columns:
            columns.append(column)
    return columns


def pick_fallback_variant(names: Sequence[object]) -> str | None:
    """Prefer control when mixed leftover variant rows must collapse to one list."""
    values = [str(name) for name in names if not is_missing(name) and str(name)]
    if not values:
        return None
    if FALLBACK_VARIANT in values:
        return FALLBACK_VARIANT
    return sorted(values)[0]


def collapse_mixed_variants(frame: Any) -> Any:
    """Keep a single variant when experiments are off but leftover rows remain."""
    if not has_variant_column(frame) or getattr(frame, "empty", False):
        return frame
    series = frame[VARIANT_COLUMN]
    pick = pick_fallback_variant(series.tolist())
    if pick is None:
        return frame
    mask = ~series.map(is_missing) & series.astype(str).eq(pick)
    if bool(mask.all()):
        return frame
    return frame.loc[mask]


def filter_variant_rows(frame: Any, variant: str | None) -> Any:
    """Keep rows for ``variant``. Missing column is a no-op; ``None`` collapses mixed lists."""
    if not has_variant_column(frame):
        return frame
    if getattr(frame, "empty", False):
        return frame
    if variant is None:
        return collapse_mixed_variants(frame)
    series = frame[VARIANT_COLUMN]
    mask = ~series.map(is_missing) & series.astype(str).eq(str(variant))
    return frame.loc[mask]


def has_variant_column(frame: Any) -> bool:
    columns = getattr(frame, "columns", None)
    return columns is not None and VARIANT_COLUMN in columns
