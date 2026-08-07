"""Per-user weighted blending of personalized / popular / latest sources.

See ``[blending]`` in ``config/features.toml``. Interaction counts are distinct
``(user, item)`` pairs after dataset aggregation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pandas as pd
from rectools import Columns

from cicerone.feature_config import BlendingConfig

SOURCE_COLUMN = "source"
BLENDED_SOURCE = "blended"
PERSONALIZED_SOURCE = "personalized"
POPULAR_SOURCE = "popular_fallback"
LATEST_SOURCE = "latest"
COLD_START_USER_ID = "__cold_start__"

_WEIGHT_EPS = 1e-9
_SIGMOID_EXP_CLAMP = 50.0
_EMPTY_COLS = [Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN]


def _empty_recs() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COLS)


def personalized_weight(n_interactions: int, config: BlendingConfig) -> float:
    """Map interaction count → weight in [0, 1] for the personalized source."""
    n = max(0, int(n_interactions))
    if n == 0:
        return 0.0
    if config.curve == "linear":
        saturate = config.saturate_at if config.saturate_at > 0 else 1.0
        return min(1.0, n / saturate)
    # Clamp to avoid math.exp overflow for extreme n / steepness.
    exponent = max(-_SIGMOID_EXP_CLAMP, min(_SIGMOID_EXP_CLAMP, -config.steepness * (n - config.midpoint)))
    return 1.0 / (1.0 + math.exp(exponent))


def source_weights(
    n_interactions: int,
    config: BlendingConfig,
    *,
    latest_available: bool,
) -> dict[str, float]:
    """Return non-negative weights for personalized / popular / latest.

    Weights sum to 1. When ``latest`` is unavailable its share is absorbed by
    ``popular``.
    """
    p = personalized_weight(n_interactions, config)
    remainder = max(0.0, 1.0 - p)
    if latest_available:
        popular = remainder * config.popular_share
        latest = remainder * (1.0 - config.popular_share)
    else:
        popular = remainder
        latest = 0.0
    return {
        PERSONALIZED_SOURCE: p,
        POPULAR_SOURCE: popular,
        LATEST_SOURCE: latest,
    }


def resolve_latest_date_column(
    items: pd.DataFrame | None,
    candidates: Sequence[str],
) -> str | None:
    """First usable datetime column from ``candidates`` present in ``items``."""
    if items is None or items.empty or not candidates:
        return None
    for column in candidates:
        if column not in items.columns:
            continue
        parsed = pd.to_datetime(items[column], errors="coerce", utc=True)
        if parsed.notna().any():
            return column
    return None


def interaction_counts(interactions: pd.DataFrame) -> dict[str, int]:
    """Count distinct (user, item) rows per user after dataset aggregation."""
    if interactions.empty or Columns.User not in interactions.columns:
        return {}
    counts = interactions.groupby(Columns.User).size()
    return {str(user): int(count) for user, count in counts.items()}


def collapse_best_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the best (lowest) rank per (user, item); ties break on higher score."""
    if frame.empty:
        return _empty_recs()
    out = frame.sort_values(
        [Columns.User, Columns.Item, Columns.Rank, Columns.Score],
        ascending=[True, True, True, False],
    )
    return out.drop_duplicates(subset=[Columns.User, Columns.Item], keep="first").reset_index(drop=True)


def rank_latest_items(
    items: pd.DataFrame,
    date_column: str,
    allowed_item_ids: Sequence,
    top_k: int,
) -> list[tuple[str, int, float]]:
    """Non-personalized top-K by newest ``date_column`` as ``(item_id, rank, score)``."""
    if top_k < 1 or not allowed_item_ids:
        return []

    allowed = {str(i) for i in allowed_item_ids}
    if Columns.Item not in items.columns:
        return []
    item_col = Columns.Item
    frame = items.loc[items[item_col].astype(str).isin(allowed)].copy()
    if frame.empty:
        return []

    frame["_date"] = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
    frame = frame.dropna(subset=["_date"])
    if frame.empty:
        return []

    frame = frame.sort_values(["_date", item_col], ascending=[False, True]).head(top_k)
    frame = frame.reset_index(drop=True)
    ranks = list(range(1, len(frame) + 1))
    scores = [float(len(frame) - i + 1) for i in ranks]  # synthetic; blend uses ranks
    item_ids = frame[item_col].astype(str).tolist()
    return list(zip(item_ids, ranks, scores, strict=True))


def build_latest_ranking(
    items: pd.DataFrame,
    date_column: str,
    allowed_item_ids: Sequence,
    top_k: int,
    target_users: Sequence[str],
) -> pd.DataFrame:
    """Non-personalized top-K by newest ``date_column``, expanded to ``target_users``.

    Prefer ``rank_latest_items`` + blend-time expansion when many users share one
    allowlist; this helper remains for callers that need an explicit frame.
    """
    if not target_users:
        return _empty_recs()
    ranked = rank_latest_items(items, date_column, allowed_item_ids, top_k)
    if not ranked:
        return _empty_recs()

    rows = [
        {
            Columns.User: user_id,
            Columns.Item: item_id,
            Columns.Rank: rank,
            Columns.Score: score,
            SOURCE_COLUMN: LATEST_SOURCE,
        }
        for user_id in target_users
        for item_id, rank, score in ranked
    ]
    return pd.DataFrame(rows)


def expand_latest_ranking(
    ranked: Sequence[tuple[str, int, float]],
    target_users: Sequence[str],
) -> pd.DataFrame:
    """Broadcast a shared latest ranking to ``target_users`` without re-sorting items."""
    if not ranked or not target_users:
        return _empty_recs()
    rows = [
        {
            Columns.User: user_id,
            Columns.Item: item_id,
            Columns.Rank: rank,
            Columns.Score: score,
            SOURCE_COLUMN: LATEST_SOURCE,
        }
        for user_id in target_users
        for item_id, rank, score in ranked
    ]
    return pd.DataFrame(rows)


def _normalize_source_frame(frame: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if frame.empty:
        return _empty_recs()
    out = collapse_best_rank(frame)
    out[Columns.User] = out[Columns.User].astype(str)
    out[Columns.Item] = out[Columns.Item].astype(str)
    out[SOURCE_COLUMN] = source_label
    return out[[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN]]


def _rrf_contrib(rank: float, weight: float, rrf_k: float) -> float:
    return weight / (rrf_k + rank)


def _rows_by_user(frame: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """Map user_id → [(item_id, rank), ...] for O(1) blend lookups."""
    if frame.empty:
        return {}
    grouped: dict[str, list[tuple[str, float]]] = {}
    for row in frame.itertuples(index=False):
        user_id = str(getattr(row, Columns.User))
        item_id = str(getattr(row, Columns.Item))
        rank = float(getattr(row, Columns.Rank))
        grouped.setdefault(user_id, []).append((item_id, rank))
    return grouped


def blend_for_users(
    *,
    personalized: pd.DataFrame,
    popular: pd.DataFrame,
    latest: pd.DataFrame | None,
    counts: Mapping[str, int],
    target_users: Sequence[str],
    config: BlendingConfig,
    top_k: int,
    latest_available: bool,
    shared_latest: Sequence[tuple[str, int, float]] | None = None,
) -> pd.DataFrame:
    """Weighted RRF with per-user source weights (best rank per item within each source).

    When ``shared_latest`` is set, that ranking is applied to every target user
    instead of reading a Cartesian ``latest`` frame (avoids U×K row explosion).
    """
    frames = {
        PERSONALIZED_SOURCE: _normalize_source_frame(personalized, PERSONALIZED_SOURCE),
        POPULAR_SOURCE: _normalize_source_frame(popular, POPULAR_SOURCE),
        LATEST_SOURCE: (
            _normalize_source_frame(latest, LATEST_SOURCE)
            if shared_latest is None and latest is not None and latest_available
            else _empty_recs()
        ),
    }
    by_source_user = {label: _rows_by_user(frame) for label, frame in frames.items()}

    by_user_item: dict[tuple[str, str], tuple[float, set[str]]] = {}
    for user_id in dict.fromkeys(str(u) for u in target_users):
        weights = source_weights(
            counts.get(user_id, 0),
            config,
            latest_available=latest_available,
        )
        for source_label, user_index in by_source_user.items():
            weight = weights.get(source_label, 0.0)
            if weight <= _WEIGHT_EPS:
                continue
            for item_id, rank in user_index.get(user_id, ()):
                key = (user_id, item_id)
                score, sources = by_user_item.get(key, (0.0, set()))
                score += _rrf_contrib(rank, weight, config.rrf_k)
                sources.add(source_label)
                by_user_item[key] = (score, sources)

        if shared_latest is not None and latest_available:
            weight = weights.get(LATEST_SOURCE, 0.0)
            if weight > _WEIGHT_EPS:
                for item_id, rank, _score in shared_latest:
                    key = (user_id, str(item_id))
                    score, sources = by_user_item.get(key, (0.0, set()))
                    score += _rrf_contrib(float(rank), weight, config.rrf_k)
                    sources.add(LATEST_SOURCE)
                    by_user_item[key] = (score, sources)

    if not by_user_item:
        return _empty_recs()

    rows = []
    for (user_id, item_id), (score, sources) in by_user_item.items():
        label = BLENDED_SOURCE if len(sources) > 1 else next(iter(sources))
        rows.append(
            {
                Columns.User: user_id,
                Columns.Item: item_id,
                Columns.Score: score,
                SOURCE_COLUMN: label,
            }
        )

    combined = pd.DataFrame(rows)
    combined = combined.sort_values(
        [Columns.User, Columns.Score, Columns.Item], ascending=[True, False, True]
    )
    combined[Columns.Rank] = combined.groupby(Columns.User).cumcount() + 1
    combined = combined.groupby(Columns.User, as_index=False).head(top_k)
    return combined[[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN]].reset_index(
        drop=True
    )


def append_cold_start_rows(
    recommendations: pd.DataFrame,
    *,
    popular: pd.DataFrame,
    latest: pd.DataFrame | None,
    config: BlendingConfig,
    top_k: int,
    latest_available: bool,
    shared_latest: Sequence[tuple[str, int, float]] | None = None,
) -> pd.DataFrame:
    """Append a ``__cold_start__`` row set for serve fallback (global allowlist)."""
    cold = blend_for_users(
        personalized=_empty_recs(),
        popular=popular,
        latest=latest,
        counts={COLD_START_USER_ID: 0},
        target_users=[COLD_START_USER_ID],
        config=config,
        top_k=top_k,
        latest_available=latest_available,
        shared_latest=shared_latest,
    )
    if cold.empty:
        return recommendations
    without = recommendations[recommendations[Columns.User].astype(str) != COLD_START_USER_ID]
    return pd.concat([without, cold], ignore_index=True)
