"""Per-user weighted blending of personalized / popular / latest sources.

See ``[blending]`` in ``config/features.toml``. Interaction counts are distinct
``(user, item)`` pairs after dataset aggregation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
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
    return expand_latest_ranking(
        rank_latest_items(items, date_column, allowed_item_ids, top_k),
        target_users,
    )


def expand_latest_ranking(
    ranked: Sequence[tuple[str, int, float]],
    target_users: Sequence[str],
) -> pd.DataFrame:
    """Broadcast a shared latest ranking to ``target_users`` without re-sorting items."""
    if not ranked or not target_users:
        return _empty_recs()
    item_ids, ranks, scores = zip(*ranked, strict=True)
    n_items = len(item_ids)
    n_users = len(target_users)
    return pd.DataFrame(
        {
            Columns.User: np.repeat(np.asarray(target_users, dtype=object), n_items),
            Columns.Item: np.tile(np.asarray(item_ids, dtype=object), n_users),
            Columns.Rank: np.tile(np.asarray(ranks, dtype=np.int64), n_users),
            Columns.Score: np.tile(np.asarray(scores, dtype=np.float64), n_users),
            SOURCE_COLUMN: LATEST_SOURCE,
        }
    )


def _normalize_source_frame(frame: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if frame.empty:
        return _empty_recs()
    out = collapse_best_rank(frame)
    out[Columns.User] = out[Columns.User].astype(str)
    out[Columns.Item] = out[Columns.Item].astype(str)
    out[SOURCE_COLUMN] = source_label
    return out[[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN]]


def _weighted_rrf_frame(
    frame: pd.DataFrame,
    *,
    source_label: str,
    user_weights: pd.Series,
    rrf_k: float,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Score, SOURCE_COLUMN])
    weights = frame[Columns.User].map(user_weights)
    keep = weights.notna() & (weights > _WEIGHT_EPS)
    if not bool(keep.any()):
        return pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Score, SOURCE_COLUMN])
    out = frame.loc[keep, [Columns.User, Columns.Item, Columns.Rank]].copy()
    ranks = out[Columns.Rank].to_numpy(dtype=float)
    out[Columns.Score] = weights.loc[keep].to_numpy(dtype=float) / (rrf_k + ranks)
    out[SOURCE_COLUMN] = source_label
    return out[[Columns.User, Columns.Item, Columns.Score, SOURCE_COLUMN]]


def _source_label_from_parts(labels: pd.Series) -> str:
    uniq = list(dict.fromkeys(labels.tolist()))
    return BLENDED_SOURCE if len(uniq) > 1 else uniq[0]


def _latest_index_frame(
    latest_index: Mapping[str, Sequence[tuple[str, int, float]]],
    target_users: Sequence[str],
) -> pd.DataFrame:
    wanted = set(target_users)
    rows: list[dict[str, object]] = []
    for user_id, ranking in latest_index.items():
        if user_id not in wanted or not ranking:
            continue
        for item_id, rank, _score in ranking:
            rows.append(
                {
                    Columns.User: user_id,
                    Columns.Item: str(item_id),
                    Columns.Rank: float(rank),
                    Columns.Score: 0.0,
                    SOURCE_COLUMN: LATEST_SOURCE,
                }
            )
    return pd.DataFrame(rows) if rows else _empty_recs()


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
    latest_by_user: Mapping[str, Sequence[tuple[str, int, float]]] | None = None,
) -> pd.DataFrame:
    """Weighted RRF with per-user source weights (best rank per item within each source).

    Prefer ``shared_latest`` (one ranking for all users) or ``latest_by_user``
    (per-user / per-cohort rankings; keys must be ``str`` user ids) over a
    Cartesian ``latest`` frame.
    """
    unique_users = list(dict.fromkeys(str(u) for u in target_users))
    if not unique_users:
        return _empty_recs()

    use_indexed_latest = shared_latest is not None or latest_by_user is not None
    latest_index = (
        None if latest_by_user is None else {str(uid): ranking for uid, ranking in latest_by_user.items()}
    )
    frames = {
        PERSONALIZED_SOURCE: _normalize_source_frame(personalized, PERSONALIZED_SOURCE),
        POPULAR_SOURCE: _normalize_source_frame(popular, POPULAR_SOURCE),
        LATEST_SOURCE: (
            _normalize_source_frame(latest, LATEST_SOURCE)
            if not use_indexed_latest and latest is not None and latest_available
            else _empty_recs()
        ),
    }
    weight_map = {
        user_id: source_weights(counts.get(user_id, 0), config, latest_available=latest_available)
        for user_id in unique_users
    }
    personalized_w = pd.Series({u: w[PERSONALIZED_SOURCE] for u, w in weight_map.items()})
    popular_w = pd.Series({u: w[POPULAR_SOURCE] for u, w in weight_map.items()})
    latest_w = pd.Series({u: w[LATEST_SOURCE] for u, w in weight_map.items()})
    rrf_k = config.rrf_k

    contribs = [
        _weighted_rrf_frame(
            frames[PERSONALIZED_SOURCE],
            source_label=PERSONALIZED_SOURCE,
            user_weights=personalized_w,
            rrf_k=rrf_k,
        ),
        _weighted_rrf_frame(
            frames[POPULAR_SOURCE],
            source_label=POPULAR_SOURCE,
            user_weights=popular_w,
            rrf_k=rrf_k,
        ),
        _weighted_rrf_frame(
            frames[LATEST_SOURCE],
            source_label=LATEST_SOURCE,
            user_weights=latest_w,
            rrf_k=rrf_k,
        ),
    ]
    if latest_available:
        indexed_latest = _empty_recs()
        if shared_latest is not None:
            indexed_latest = expand_latest_ranking(shared_latest, unique_users)
        elif latest_index is not None:
            indexed_latest = _latest_index_frame(latest_index, unique_users)
        contribs.append(
            _weighted_rrf_frame(
                indexed_latest,
                source_label=LATEST_SOURCE,
                user_weights=latest_w,
                rrf_k=rrf_k,
            )
        )

    contribs = [frame for frame in contribs if not frame.empty]
    if not contribs:
        return _empty_recs()
    stacked = pd.concat(contribs, ignore_index=True)

    combined = stacked.groupby([Columns.User, Columns.Item], as_index=False, sort=False).agg(
        **{
            Columns.Score: (Columns.Score, "sum"),
            SOURCE_COLUMN: (SOURCE_COLUMN, _source_label_from_parts),
        }
    )
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
