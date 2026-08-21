"""User-id lookup of precomputed recommendations for the dashboard."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from cicerone.config import Settings
from cicerone.io.base import RecommendationReader
from cicerone.io.recommendation_schema import ITEM_COLUMN, RANK_COLUMN, SCORE_COLUMN, SOURCE_COLUMN

logger = logging.getLogger(__name__)

LOOKUP_FAILED = "Could not load recommendations."
MISSING = "—"


def empty_recommendations_context(
    *,
    user_id: str = "",
    queried: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "items": [],
        "fallback": False,
        "queried": queried,
        "show_category": False,
        "error": error,
    }


def lookup_k(top_k: int, cap: int) -> int:
    return min(top_k, cap)


def lookup_recommendations(
    settings: Settings,
    recommendation_reader: RecommendationReader | None,
    user_id: str,
) -> dict[str, Any]:
    user_id = user_id.strip()
    if not user_id:
        return empty_recommendations_context()
    if recommendation_reader is None:
        return empty_recommendations_context(
            user_id=user_id,
            queried=True,
            error="Recommendation store is not available.",
        )

    try:
        recommendation_reader.refresh()
    except Exception:
        logger.exception("Failed to refresh recommendation reader for dashboard lookup")

    k = lookup_k(settings.top_k, settings.dashboard.lookup_k)
    try:
        recs, used_fallback = _load_rows(recommendation_reader, user_id, k)
        category_column = settings.serve.category_column
        if category_column not in recs.columns:
            recs = _join_category(recs, recommendation_reader.get_items(), category_column)
        show_category = category_column in recs.columns
        return {
            "user_id": user_id,
            "items": format_recommendation_rows(
                recs, category_column=category_column if show_category else None
            ),
            "fallback": used_fallback,
            "queried": True,
            "show_category": show_category,
            "error": None,
        }
    except Exception:
        logger.exception("Failed to look up recommendations for user_id=%r", user_id)
        return empty_recommendations_context(user_id=user_id, queried=True, error=LOOKUP_FAILED)


def format_recommendation_rows(recs: pd.DataFrame, *, category_column: str | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in recs.to_dict(orient="records"):
        item = {
            "rank": _format_rank(record.get(RANK_COLUMN)),
            "item_id": str(record.get(ITEM_COLUMN, "")),
            "score": _format_score(record.get(SCORE_COLUMN)),
            "source": _format_text(record.get(SOURCE_COLUMN)),
        }
        if category_column is not None:
            item["category"] = _format_text(record.get(category_column))
        rows.append(item)
    return rows


def _load_rows(
    recommendation_reader: RecommendationReader, user_id: str, k: int
) -> tuple[pd.DataFrame, bool]:
    recs = recommendation_reader.get_recommendations(user_id, k)
    if not recs.empty:
        return recs, False
    fallback = recommendation_reader.get_cold_start_fallback(k)
    if fallback.empty:
        return recs, False
    return fallback, True


def _join_category(recs: pd.DataFrame, items: pd.DataFrame | None, category_column: str) -> pd.DataFrame:
    if (
        items is None
        or items.empty
        or recs.empty
        or ITEM_COLUMN not in recs.columns
        or ITEM_COLUMN not in items.columns
        or category_column not in items.columns
    ):
        return recs
    extra = items[[ITEM_COLUMN, category_column]].drop_duplicates(subset=[ITEM_COLUMN])
    return recs.merge(extra, on=ITEM_COLUMN, how="left")


def _is_missing(value: object) -> bool:
    if value is None or value == "":
        return True
    result = pd.isna(value)
    return bool(result) if isinstance(result, bool) else False


def _coerce_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _format_rank(value: object) -> str:
    number = _coerce_float(value)
    if number is None:
        return MISSING
    return str(int(number))


def _format_score(value: object) -> str:
    number = _coerce_float(value)
    if number is None:
        return MISSING
    return f"{number:.4f}"


def _format_text(value: object) -> str:
    if _is_missing(value):
        return MISSING
    return str(value)
