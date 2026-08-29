"""Attach structured recommendation reasons after combine + boosts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import pandas as pd
from rectools import Columns

from cicerone.config.settings import ExplainSettings
from cicerone.content_fallback import item_feature_tokens
from cicerone.feature_config import FeatureColumn
from cicerone.ids import interactions_item_column, interactions_user_column, items_id_column
from cicerone.io.recommendation_schema import REASONS_COLUMN, SOURCE_COLUMN
from cicerone.reasons import (
    BOOST_HITS_COLUMN,
    SOURCE_CONTRIBS_COLUMN,
    dump_reasons,
)

_MAX_HISTORY_ITEMS = 50
_INTERNAL_COLUMNS = (SOURCE_CONTRIBS_COLUMN, BOOST_HITS_COLUMN)


def _drop_internal(frame: pd.DataFrame) -> pd.DataFrame:
    drop = [column for column in _INTERNAL_COLUMNS if column in frame.columns]
    return frame.drop(columns=drop) if drop else frame


def _as_contribs(value: object, source: object) -> list[dict[str, object]]:
    items = [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
    if items:
        return items
    label = str(source) if source is not None and str(source) != "" else ""
    if not label:
        return []
    return [{"label": label, "rank": None, "weight": None, "contribution": None}]


def _as_boosts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    hits: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict) or item.get("name") is None or item.get("factor") is None:
            continue
        hits.append(item)
    return hits


def _item_token_index(
    items: pd.DataFrame | None,
    feature_columns: Sequence[FeatureColumn | tuple[str, str]],
    *,
    keep_ids: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    if items is None or items.empty or not feature_columns:
        return {}
    id_col = items_id_column(items)
    index: dict[str, dict[str, float]] = {}
    for record in items.to_dict(orient="records"):
        item_id = str(record[id_col])
        if keep_ids is not None and item_id not in keep_ids:
            continue
        tokens = item_feature_tokens(record, feature_columns)
        if tokens:
            index[item_id] = tokens
    return index


def _user_history(
    interactions: pd.DataFrame | None,
    *,
    cap: int = _MAX_HISTORY_ITEMS,
) -> dict[str, list[str]]:
    if interactions is None or interactions.empty:
        return {}
    user_col = interactions_user_column(interactions)
    item_col = interactions_item_column(interactions)
    ordered = interactions
    if Columns.Datetime in interactions.columns:
        ordered = interactions.sort_values(Columns.Datetime, kind="mergesort")
    history: dict[str, list[str]] = {}
    for user_id, item_id in zip(
        ordered[user_col].astype(str).tolist(),
        ordered[item_col].astype(str).tolist(),
        strict=True,
    ):
        items = history.setdefault(user_id, [])
        items.append(item_id)
    for user_id, items in history.items():
        if len(items) > cap:
            history[user_id] = items[-cap:]
    return history


def _token_column_value(token: str) -> tuple[str, str]:
    column, separator, value = token.partition("=")
    if not separator:
        return token, ""
    return column, value


def overlap_for_item(
    *,
    item_id: str,
    history_ids: Sequence[str],
    token_index: Mapping[str, dict[str, float]],
    max_similar_items: int,
    max_attributes: int,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    rec_tokens = token_index.get(item_id)
    if not rec_tokens:
        return [], []
    rec_keys = set(rec_tokens)
    scored: list[tuple[str, float, set[str]]] = []
    for history_id in history_ids:
        if history_id == item_id:
            continue
        hist_tokens = token_index.get(history_id)
        if not hist_tokens:
            continue
        hist_keys = set(hist_tokens)
        shared = rec_keys & hist_keys
        if not shared:
            continue
        union = rec_keys | hist_keys
        scored.append((history_id, len(shared) / len(union), shared))
    scored.sort(key=lambda row: (-row[1], row[0]))
    similar: list[dict[str, object]] = []
    if max_similar_items > 0:
        top = scored[: max(0, max_similar_items)]
        similar = [{"item_id": item, "score": float(score)} for item, score, _shared in top]
        attr_rows = top
    else:
        attr_rows = scored
    counts: Counter[tuple[str, str]] = Counter()
    for _item, _score, shared in attr_rows:
        for token in shared:
            counts[_token_column_value(token)] += 1
    matched = [
        {"column": column, "value": str(value)} for (column, value), _count in counts.most_common() if column
    ]
    matched.sort(key=lambda row: (-counts[(row["column"], row["value"])], row["column"], row["value"]))
    return similar, matched[: max(0, max_attributes)]


def attach_reasons(
    recs: pd.DataFrame,
    *,
    items: pd.DataFrame | None,
    interactions: pd.DataFrame | None,
    feature_columns: Sequence[FeatureColumn | tuple[str, str]],
    settings: ExplainSettings,
) -> pd.DataFrame:
    """Serialize ``reasons`` JSON; drop internal combiner/boost columns."""
    if recs.empty:
        return _drop_internal(recs)
    if not settings.enabled:
        return _drop_internal(recs)

    out = recs.copy()
    want_overlap = settings.max_similar_items > 0 or settings.max_attributes > 0
    history = _user_history(interactions) if want_overlap else {}
    keep_ids: set[str] | None = None
    if want_overlap:
        keep_ids = set(out[Columns.Item].astype(str))
        for item_ids in history.values():
            keep_ids.update(item_ids)
    token_index = _item_token_index(items, feature_columns, keep_ids=keep_ids) if want_overlap else {}
    want_overlap = bool(token_index) and want_overlap

    payloads: list[str] = []
    for record in out.to_dict(orient="records"):
        user_id = str(record[Columns.User])
        item_id = str(record[Columns.Item])
        similar: list[dict[str, object]] = []
        matched: list[dict[str, str]] = []
        if want_overlap:
            similar, matched = overlap_for_item(
                item_id=item_id,
                history_ids=history.get(user_id, ()),
                token_index=token_index,
                max_similar_items=settings.max_similar_items,
                max_attributes=settings.max_attributes,
            )
        payload = {
            "sources": _as_contribs(record.get(SOURCE_CONTRIBS_COLUMN), record.get(SOURCE_COLUMN)),
            "boosts": _as_boosts(record.get(BOOST_HITS_COLUMN)),
            "similar_items": similar,
            "matched_attributes": matched,
        }
        payloads.append(dump_reasons(payload))
    out[REASONS_COLUMN] = payloads
    return _drop_internal(out)
