"""User-id lookup of precomputed recommendations and input events for the dashboard."""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from cicerone.config import Settings
from cicerone.experiment.assignment import resolve_assignment
from cicerone.experiment.store import ExperimentStore
from cicerone.io.base import RecommendationReader, UserHistoryReader
from cicerone.io.options import is_s3_not_found
from cicerone.io.recommendation_schema import (
    ITEM_COLUMN,
    RANK_COLUMN,
    SCORE_COLUMN,
    SOURCE_COLUMN,
    has_variant_column,
)
from cicerone.io.user_lookup import OCCURRED_AT_COLUMN
from cicerone.reasons import parse_reasons
from cicerone.values import as_list, is_missing, is_sequence_attr

logger = logging.getLogger(__name__)

LOOKUP_FAILED = "Could not load recommendations."
HISTORY_UNAVAILABLE = "Event history is not available."
HISTORY_FAILED = "Could not load event history."
MISSING = "—"
_LOOKUP_REFRESH_TTL_SECONDS = 5.0
_last_lookup_refresh: tuple[int, float] | None = None
_MAX_USER_ATTRS = 12
_EVENT_TYPE_COLUMN = "event_type"
_QUANTITY_COLUMN = "quantity"


def empty_history_fields() -> dict[str, Any]:
    return {
        "events": [],
        "events_error": None,
        "user_attrs": [],
        "source_mix": [],
        "overlap_item_ids": [],
        "warm": False,
        "show_quantity": False,
    }


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
        "show_reasons": False,
        "experiment_id": None,
        "variant": None,
        "error": error,
        **empty_history_fields(),
    }


def lookup_k(top_k: int, cap: int) -> int:
    return min(top_k, cap)


def lookup_inspector(
    settings: Settings,
    recommendation_reader: RecommendationReader | None,
    history_reader: UserHistoryReader | None,
    user_id: str,
) -> dict[str, Any]:
    recs = lookup_recommendations(settings, recommendation_reader, user_id)
    if not recs["queried"]:
        return recs
    recs.update(lookup_history(settings, history_reader, recs["user_id"]))
    rec_ids = {row["item_id"] for row in recs["items"] if row["item_id"] and row["item_id"] != MISSING}
    event_ids = {row["item_id"] for row in recs["events"] if row["item_id"] and row["item_id"] != MISSING}
    recs["overlap_item_ids"] = sorted(rec_ids & event_ids)
    recs["source_mix"] = _source_mix(recs["items"])
    recs["warm"] = bool(recs["events"])
    return recs


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

    global _last_lookup_refresh
    try:
        global _last_lookup_refresh
        key = id(recommendation_reader)
        now = time.monotonic()
        last_key, last_at = _last_lookup_refresh if _last_lookup_refresh is not None else (0, 0.0)
        if key != last_key or now - last_at >= _LOOKUP_REFRESH_TTL_SECONDS:
            recommendation_reader.refresh()
            _last_lookup_refresh = (key, now)
    except Exception:
        logger.exception("Failed to refresh recommendation reader for dashboard lookup")

    k = lookup_k(settings.top_k, settings.dashboard.lookup_k)
    experiment_id, variant = None, None
    if settings.experiment.enabled:
        promoted, active_pair = ExperimentStore(settings.output).assignment_overlay(settings.experiment.id)
        experiment_id, variant = resolve_assignment(
            settings, user_id, promoted_variant=promoted, active_pair=active_pair
        )
    try:
        recs, used_fallback = _load_rows(recommendation_reader, user_id, k, variant=variant)
        if not has_variant_column(recs):
            experiment_id, variant = None, None
        category_column = settings.serve.category_column
        if category_column not in recs.columns:
            recs = _join_category(recs, recommendation_reader.get_items(), category_column)
        show_category = category_column in recs.columns
        items = format_recommendation_rows(recs, category_column=category_column if show_category else None)
        return {
            "user_id": user_id,
            "items": items,
            "fallback": used_fallback,
            "queried": True,
            "show_category": show_category,
            "show_reasons": any(row.get("reasons") and row["reasons"] != MISSING for row in items),
            "experiment_id": experiment_id,
            "variant": variant,
            "error": None,
            **empty_history_fields(),
        }
    except Exception:
        logger.exception("Failed to look up recommendations for user_id=%r", user_id)
        return empty_recommendations_context(user_id=user_id, queried=True, error=LOOKUP_FAILED)


def lookup_history(
    settings: Settings,
    history_reader: UserHistoryReader | None,
    user_id: str,
) -> dict[str, Any]:
    empty = empty_history_fields()
    if history_reader is None:
        empty["events_error"] = HISTORY_UNAVAILABLE
        return empty

    try:
        events = history_reader.get_events_for_user(user_id, settings.dashboard.lookup_events)
    except Exception as exc:
        if _history_unavailable(exc):
            empty["events_error"] = HISTORY_UNAVAILABLE
        else:
            logger.exception("Failed to look up events for user_id=%r", user_id)
            empty["events_error"] = HISTORY_FAILED
    else:
        rows = format_event_rows(events)
        empty["events"] = rows
        empty["show_quantity"] = any(row["quantity"] != MISSING for row in rows)

    if settings.dashboard.lookup_user_attrs:
        try:
            user = history_reader.get_user(user_id)
            empty["user_attrs"] = format_user_attrs(user, allowed=settings.dashboard.lookup_user_attrs)
        except Exception:
            logger.exception("Failed to look up user attributes for user_id=%r", user_id)
    return empty


def format_recommendation_rows(recs: pd.DataFrame, *, category_column: str | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in recs.to_dict(orient="records"):
        item = {
            "rank": _format_rank(record.get(RANK_COLUMN)),
            "item_id": str(record.get(ITEM_COLUMN, "")),
            "score": _format_score(record.get(SCORE_COLUMN)),
            "source": _format_text(record.get(SOURCE_COLUMN)),
            "reasons": _format_reasons(record.get("reasons")),
        }
        if category_column is not None:
            item["category"] = _format_text(record.get(category_column))
        rows.append(item)
    return rows


def _format_reasons(value: object) -> str:
    parsed = parse_reasons(value)
    if parsed is None:
        return MISSING
    parts: list[str] = []
    labels = [item.label for item in parsed.sources]
    if labels:
        parts.append("+".join(labels))
    if parsed.similar_items:
        parts.append(f"like {parsed.similar_items[0].item_id}")
    return " · ".join(parts) if parts else MISSING


def format_event_rows(events: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in events.to_dict(orient="records"):
        rows.append(
            {
                "occurred_at": _format_occurred_at(record.get(OCCURRED_AT_COLUMN)),
                "item_id": _format_text(record.get(ITEM_COLUMN)),
                "event_type": _format_text(record.get(_EVENT_TYPE_COLUMN)),
                "quantity": _format_quantity(record.get(_QUANTITY_COLUMN)),
            }
        )
    return rows


def format_user_attrs(user: dict[str, Any] | None, *, allowed: Sequence[str] = ()) -> list[dict[str, str]]:
    if not user or not allowed:
        return []
    rows: list[dict[str, str]] = []
    for key in allowed:
        if key == "user_id" or key not in user:
            continue
        value = user[key]
        if _is_missing(value):
            continue
        rendered = _format_attr_value(value)
        if rendered == MISSING:
            continue
        rows.append({"name": str(key), "value": rendered})
        if len(rows) >= _MAX_USER_ATTRS:
            break
    return rows


def _load_rows(
    recommendation_reader: RecommendationReader, user_id: str, k: int, *, variant: str | None = None
) -> tuple[pd.DataFrame, bool]:
    recs = recommendation_reader.get_recommendations(user_id, k, variant=variant)
    if not recs.empty:
        return recs, False
    fallback = recommendation_reader.get_cold_start_fallback(k, variant=variant)
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


def _coerce_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _source_mix(items: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for item in items:
        source = item.get("source") or MISSING
        if source == MISSING:
            continue
        counts[source] += 1
    return [{"source": name, "count": count} for name, count in counts.most_common()]


def _history_unavailable(exc: BaseException) -> bool:
    if isinstance(exc, FileNotFoundError) or is_s3_not_found(exc):
        return True
    message = str(exc).lower()
    return "no such table" in message or "does not exist" in message or "no such file" in message


def _is_missing(value: object) -> bool:
    if is_sequence_attr(value):
        return False
    if is_missing(value):
        return True
    try:
        return value == ""
    except (ValueError, TypeError):
        return False


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


def _format_quantity(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if _is_missing(number):
        return MISSING
    as_float = float(number)
    if as_float == int(as_float):
        return str(int(as_float))
    return str(as_float)


def _format_occurred_at(value: object) -> str:
    if _is_missing(value):
        return MISSING
    if isinstance(value, pd.Timestamp):
        ts = value.tz_localize("UTC") if value.tzinfo is None else value
        return ts.isoformat()
    if isinstance(value, datetime):
        ts = value.replace(tzinfo=UTC) if value.tzinfo is None else value
        return ts.isoformat()
    return str(value)


def _format_text(value: object) -> str:
    if is_missing(value) or value == "":
        return MISSING
    return str(value)


def _format_attr_value(value: object) -> str:
    if is_sequence_attr(value):
        rendered = ", ".join(str(item) for item in as_list(value))
        return rendered if rendered else MISSING
    return _format_text(value)
