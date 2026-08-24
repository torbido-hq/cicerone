"""Content-based cold-item fallback: one-hot item features, cosine vs user history.

No free-text / TF-IDF — ``item_features`` are categoricals / lists.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cicerone.feature_config import FeatureColumn
from cicerone.ids import interactions_item_column, interactions_user_column, items_id_column
from cicerone.values import is_missing as _is_missing

logger = logging.getLogger(__name__)

CONTENT_FALLBACK_SOURCE = "content_fallback"
# Cap history length so cold×history cosine stays bounded for heavy users.
_MAX_HISTORY_ITEMS = 50
# Soft cap on a single dense cosine block; larger cold sets are scored in batches.
_DENSE_SIM_BATCH_PRODUCT = 50_000
_RECOMMEND_THREAD_MIN_USERS = 8
_RECOMMEND_THREAD_MAX_WORKERS = 8


def _feature_dict(
    row: Mapping[str, Any] | pd.Series,
    feature_columns: Sequence[FeatureColumn | tuple[str, str]],
) -> dict[str, float]:
    """Map one item row to {feature=value: 1.0} tokens for DictVectorizer."""
    tokens: dict[str, float] = {}
    for spec in feature_columns:
        if isinstance(spec, FeatureColumn):
            column = spec.column
            ftype: str = spec.type
        else:
            column, ftype = spec
        if isinstance(row, pd.Series):
            if column not in row.index:
                continue
            value: Any = row[column]
        else:
            if column not in row:
                continue
            value = row[column]
        if ftype == "list":
            if _is_missing(value):
                continue
            if isinstance(value, (list, tuple, set)):
                values = list(value)
            elif isinstance(value, str):
                # Parquet/DB often store list features as comma-separated or JSON-ish strings.
                stripped = value.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    try:
                        import json

                        parsed = json.loads(stripped)
                        values = list(parsed) if isinstance(parsed, list) else [value]
                    except (json.JSONDecodeError, TypeError):
                        values = [part.strip() for part in stripped.strip("[]").split(",") if part.strip()]
                elif "," in stripped:
                    values = [part.strip() for part in stripped.split(",") if part.strip()]
                else:
                    values = [value]
            else:
                values = [value]
            for entry in values:
                if _is_missing(entry):
                    continue
                tokens[f"{column}={entry}"] = 1.0
        else:
            if _is_missing(value):
                continue
            tokens[f"{column}={value}"] = 1.0
    return tokens


def _max_cosine_scores(cold_matrix, hist_matrix) -> np.ndarray:
    """Per-cold-item max cosine vs history (exact).

    Batches the cold axis when a single dense cold×history block would be large,
    so catalogs with many zero-interaction items do not allocate one huge matrix.
    History is already capped at ``_MAX_HISTORY_ITEMS`` by the caller.
    """
    n_cold = int(cold_matrix.shape[0])
    n_hist = int(hist_matrix.shape[0])
    if n_cold == 0 or n_hist == 0:
        return np.zeros(n_cold, dtype=float)

    batch_size = max(1, _DENSE_SIM_BATCH_PRODUCT // max(n_hist, 1))
    if n_cold <= batch_size:
        sim = cosine_similarity(cold_matrix, hist_matrix, dense_output=True)
        return sim.max(axis=1)

    scores = np.empty(n_cold, dtype=float)
    for start in range(0, n_cold, batch_size):
        end = min(start + batch_size, n_cold)
        block = cosine_similarity(cold_matrix[start:end], hist_matrix, dense_output=True)
        scores[start:end] = block.max(axis=1)
    return scores


def _recommend_thread_workers(n_users: int, *, min_users: int, max_workers: int) -> int:
    if n_users < min_users:
        return 1
    cpus = os.cpu_count() or 1
    return max(1, min(max_workers, cpus, n_users))


class ContentFallbackModel:
    """RecommenderModel-compatible strategy for brand-new (zero-event) items."""

    def __init__(
        self,
        feature_columns: Sequence[FeatureColumn | tuple[str, str]] | None = None,
        max_neighbors: int = 50,
        items: pd.DataFrame | None = None,
        interactions: pd.DataFrame | None = None,
        recommend_thread_min_users: int = _RECOMMEND_THREAD_MIN_USERS,
        recommend_thread_max_workers: int = _RECOMMEND_THREAD_MAX_WORKERS,
    ) -> None:
        self.feature_columns: list[FeatureColumn | tuple[str, str]] = list(feature_columns or [])
        self.max_neighbors = max_neighbors
        self.items = items
        self.interactions = interactions
        self.recommend_thread_min_users = max(1, int(recommend_thread_min_users))
        self.recommend_thread_max_workers = max(1, int(recommend_thread_max_workers))
        self._item_ids: list[Any] = []
        self._item_index: dict[Any, int] = {}
        self._matrix: csr_matrix | None = None
        self._cold_ids: list[Any] = []
        self._cold_indices: np.ndarray = np.array([], dtype=int)
        self._user_history: dict[Any, list[Any]] = {}
        self._vectorizer: DictVectorizer | None = None

    def _reset_item_state(self) -> None:
        self._item_ids = []
        self._item_index = {}
        self._matrix = None
        self._cold_ids = []
        self._cold_indices = np.array([], dtype=int)

    def _release_fit_frames(self) -> None:
        # Drop source frames once matrices / history are built (artifact size + RAM).
        self.items = None
        self.interactions = None

    def fit(self, dataset: Dataset) -> ContentFallbackModel:
        del dataset  # history/cold set come from interactions + items frames
        self._user_history = defaultdict(list)
        if self.interactions is not None and not self.interactions.empty:
            user_col = interactions_user_column(self.interactions)
            item_col = interactions_item_column(self.interactions)
            for user_id, item_id in zip(
                self.interactions[user_col].tolist(),
                self.interactions[item_col].tolist(),
                strict=True,
            ):
                self._user_history[str(user_id)].append(str(item_id))

        if self.items is None or self.items.empty or not self.feature_columns:
            logger.info("Content fallback: no items/features — strategy will emit no rows")
            self._reset_item_state()
            self._release_fit_frames()
            return self

        id_col = items_id_column(self.items)
        interacted = set()
        if self.interactions is not None and not self.interactions.empty:
            item_col = interactions_item_column(self.interactions)
            interacted = {str(item_id) for item_id in self.interactions[item_col].tolist()}

        feature_names = []
        for spec in self.feature_columns:
            column = spec.column if isinstance(spec, FeatureColumn) else spec[0]
            if column in self.items.columns and column not in feature_names:
                feature_names.append(column)
        cols = [id_col, *feature_names]
        records = self.items.loc[:, cols].to_dict(orient="records")

        dicts = []
        item_ids = []
        for row in records:
            item_id = str(row[id_col])
            tokens = _feature_dict(row, self.feature_columns)
            if not tokens:
                continue
            dicts.append(tokens)
            item_ids.append(item_id)

        if not dicts:
            self._reset_item_state()
            self._release_fit_frames()
            return self

        self._vectorizer = DictVectorizer(sparse=True)
        self._matrix = self._vectorizer.fit_transform(dicts)
        self._item_ids = item_ids
        self._item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
        cold_indices = [i for i, item_id in enumerate(item_ids) if item_id not in interacted]
        self._cold_indices = np.asarray(cold_indices, dtype=int)
        self._cold_ids = [item_ids[i] for i in cold_indices]
        logger.info(
            "Content fallback: %d catalog items with features, %d cold (zero-interaction)",
            len(item_ids),
            len(self._cold_ids),
        )
        self._release_fit_frames()
        return self

    def recommend(
        self,
        *,
        users: list,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        items_to_recommend: list | None = None,
    ) -> pd.DataFrame:
        del dataset
        empty = pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Score, Columns.Rank])
        if self._matrix is None or len(self._cold_ids) == 0 or k < 1:
            return empty
        matrix = self._matrix

        allow = None if items_to_recommend is None else {str(i) for i in items_to_recommend}
        cold_mask = []
        cold_ids_filtered = []
        for idx, item_id in zip(self._cold_indices.tolist(), self._cold_ids, strict=True):
            if allow is not None and str(item_id) not in allow:
                continue
            cold_mask.append(idx)
            cold_ids_filtered.append(item_id)
        if not cold_mask:
            return empty

        cold_matrix = matrix[cold_mask]
        item_index = self._item_index
        take = min(k, self.max_neighbors, len(cold_ids_filtered))
        cold_ids_arr = np.asarray(cold_ids_filtered, dtype=object)

        def _score_user(user: object) -> list[dict]:
            history = self._user_history.get(str(user), [])
            if not history:
                return []
            recent_history = history[-_MAX_HISTORY_ITEMS:]
            hist_indices = [item_index[i] for i in recent_history if i in item_index]
            if not hist_indices:
                return []
            hist_matrix = matrix[hist_indices]
            scores = _max_cosine_scores(cold_matrix, hist_matrix)
            if filter_viewed:
                seen = {str(item) for item in history}
                for i, item_id in enumerate(cold_ids_arr):
                    if str(item_id) in seen:
                        scores[i] = 0.0
            positive = np.flatnonzero(scores > 0)
            if positive.size == 0:
                return []
            if take >= positive.size:
                order = positive[np.argsort(-scores[positive], kind="mergesort")]
            else:
                pos_scores = scores[positive]
                top_local = np.argpartition(pos_scores, -take)[-take:]
                order = positive[top_local[np.argsort(-pos_scores[top_local], kind="mergesort")]]
            return [
                {
                    Columns.User: user,
                    Columns.Item: cold_ids_arr[idx],
                    Columns.Score: float(scores[idx]),
                    Columns.Rank: rank,
                }
                for rank, idx in enumerate(order, start=1)
            ]

        workers = _recommend_thread_workers(
            len(users),
            min_users=self.recommend_thread_min_users,
            max_workers=self.recommend_thread_max_workers,
        )
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                user_rows = list(pool.map(_score_user, users))
        else:
            user_rows = [_score_user(user) for user in users]
        rows = [row for part in user_rows for row in part]
        if not rows:
            return empty
        return (
            pd.DataFrame(rows)
            .sort_values([Columns.User, Columns.Rank], kind="mergesort")
            .reset_index(drop=True)
        )


def build_content_fallback_model(
    feature_columns: Sequence[FeatureColumn],
    max_neighbors: int,
    items: pd.DataFrame | None,
    interactions: pd.DataFrame | None,
) -> ContentFallbackModel:
    """Factory used by model._fit_strategy (ProcessPool-picklable FeatureColumns)."""
    return ContentFallbackModel(
        feature_columns=list(feature_columns),
        max_neighbors=max_neighbors,
        items=items,
        interactions=interactions,
    )
