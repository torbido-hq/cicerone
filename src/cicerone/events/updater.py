"""Cheap incremental update: refresh popular/latest slices; write-through."""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Callable, Collection, Sequence
from datetime import UTC, datetime

import pandas as pd

from cicerone.blending import COLD_START_USER_ID, LATEST_SOURCE, POPULAR_SOURCE
from cicerone.config import IOSettings
from cicerone.events.base import NormalizedEvent
from cicerone.events.normalize import events_to_dataframe
from cicerone.events.store import (
    empty_recommendations_frame,
    load_recommendations_for_users,
)
from cicerone.feature_config import FeatureConfig
from cicerone.io.base import OutputSink
from cicerone.io.recommendation_reader import (
    ITEM_COLUMN,
    RANK_COLUMN,
    RECOMMENDATION_COLUMNS,
    SCORE_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
)

logger = logging.getLogger(__name__)

INCREMENTAL_SOURCE = "incremental"
_PRESERVE_LABELS = frozenset(
    {
        "personalized",
        "item_based",
        "content_fallback",
        "blended",
    }
)
# Reserve slots so recent interactions can enter top-K even when preserved rows fill it.
_BOOST_SLOT_FRACTION = 0.3
# Bound in-process per-user frames for long-lived serve workers.
DEFAULT_USER_CACHE_MAX_SIZE = 2048


def _is_preserved_source(source: str) -> bool:
    if source in _PRESERVE_LABELS:
        return True
    # Priority/RRF compound labels, e.g. personalized+popular_fallback.
    return any(part in _PRESERVE_LABELS for part in source.split("+"))


class IncrementalUpdater:
    """Merge micro-batch events into existing top-K rows and write via OutputSink.

    Does not import LightFM / RecTools. Personalized / blended rows are preserved
    (with reserved slots for incremental boosts); popular/latest slices refresh
    from the flushed batch. Full collaborative refits wait for ``job.run()``.

    Reads and writes only affected users (plus ``__cold_start__``), not the
    full recommendations table/file. Per-user frames are cached with LRU eviction.
    """

    def __init__(
        self,
        *,
        sink: OutputSink,
        output_settings: IOSettings,
        feature_config: FeatureConfig | None,
        top_k: int,
        busy_check: Callable[[], bool] | None = None,
        on_success: Callable[[], None] | None = None,
        user_cache_max_size: int = DEFAULT_USER_CACHE_MAX_SIZE,
    ):
        if user_cache_max_size < 1:
            raise ValueError("user_cache_max_size must be >= 1")
        self._sink = sink
        self._output_settings = output_settings
        self._feature_config = feature_config
        self._top_k = top_k
        self._busy_check = busy_check
        self._on_success = on_success
        self._last_success_at: datetime | None = None
        self._events_applied = 0
        self._user_cache_max_size = user_cache_max_size
        self._cached_by_user: OrderedDict[str, pd.DataFrame] = OrderedDict()

    @property
    def last_success_at(self) -> datetime | None:
        return self._last_success_at

    @property
    def events_applied(self) -> int:
        return self._events_applied

    def invalidate_cache(self) -> None:
        self._cached_by_user.clear()

    def apply(self, events: Sequence[NormalizedEvent]) -> int:
        if not events:
            return 0
        if self._busy_check is not None and self._busy_check():
            # Retrain may rewrite output; drop cache so the next apply reloads.
            self.invalidate_cache()
            logger.info("Skipping incremental update: full retrain in progress")
            return 0

        batch = events_to_dataframe(events)
        affected_users = sorted(set(batch[USER_COLUMN].astype(str)))
        affected_set = set(affected_users) | {COLD_START_USER_ID}
        existing = self._load_users(affected_set)

        if USER_COLUMN in existing.columns and not existing.empty:
            existing = existing.copy()
            existing[USER_COLUMN] = existing[USER_COLUMN].astype(str)

        popular_ranking = self._popular_ranking(batch)
        latest_ranking = self._latest_ranking(batch)

        by_user = (
            {user_id: group for user_id, group in existing.groupby(USER_COLUMN, sort=False)}
            if not existing.empty
            else {}
        )

        frames: list[pd.DataFrame] = []
        for user_id in affected_users:
            prior = by_user.get(user_id, empty_recommendations_frame())
            frames.append(self._merge_user_rows(user_id, prior, popular_ranking, latest_ranking, batch))
        prior_cold = by_user.get(COLD_START_USER_ID, empty_recommendations_frame())
        frames.append(self._cold_start_rows(prior_cold, popular_ranking, latest_ranking))
        frames = [frame for frame in frames if frame is not None and not frame.empty]

        merged = pd.concat(frames, ignore_index=True) if frames else empty_recommendations_frame()
        if not merged.empty:
            merged = merged[list(RECOMMENDATION_COLUMNS)]
        n_users = self._sink.replace_recommendations_for_users(merged, user_ids=sorted(affected_set))
        self._store_users_in_cache(affected_set, merged)

        now = datetime.now(UTC)
        manifest = {
            "triggered_by": "incremental",
            "status": "success",
            "error": None,
            "generated_at": now.isoformat(),
            "n_events": len(events),
            "incremental_events_applied": len(events),
            "last_incremental_at": now.isoformat(),
            "n_users_with_recommendations": n_users,
            "top_k": self._top_k,
            "partial_outputs": True,
        }
        self._sink.write_manifest(manifest)
        self._last_success_at = now
        self._events_applied += len(events)
        if self._on_success is not None:
            self._on_success()
        logger.info(
            "Incremental update wrote recommendations for %d user(s) from %d event(s)",
            len(affected_users),
            len(events),
        )
        return len(events)

    def _cache_put(self, user_id: str, frame: pd.DataFrame, *, protect: Collection[str]) -> None:
        self._cached_by_user[user_id] = frame
        self._cached_by_user.move_to_end(user_id)
        self._trim_user_cache(protect=protect)

    def _trim_user_cache(self, *, protect: Collection[str]) -> None:
        protect_set = set(protect)
        while len(self._cached_by_user) > self._user_cache_max_size:
            victim = next((key for key in self._cached_by_user if key not in protect_set), None)
            if victim is None:
                # Active batch larger than the cap; allow temporary oversize.
                return
            self._cached_by_user.pop(victim)

    def _load_users(self, user_ids: set[str]) -> pd.DataFrame:
        missing = sorted(user_id for user_id in user_ids if user_id not in self._cached_by_user)
        if missing:
            loaded = load_recommendations_for_users(self._output_settings, missing)
            if USER_COLUMN in loaded.columns and not loaded.empty:
                loaded = loaded.copy()
                loaded[USER_COLUMN] = loaded[USER_COLUMN].astype(str)
                by_user = {
                    user_id: group.copy() for user_id, group in loaded.groupby(USER_COLUMN, sort=False)
                }
            else:
                by_user = {}
            for user_id in missing:
                self._cache_put(
                    user_id,
                    by_user.get(user_id, empty_recommendations_frame()),
                    protect=user_ids,
                )
        for user_id in user_ids:
            if user_id in self._cached_by_user:
                self._cached_by_user.move_to_end(user_id)
        parts = [
            self._cached_by_user[user_id]
            for user_id in sorted(user_ids)
            if not self._cached_by_user[user_id].empty
        ]
        if not parts:
            return empty_recommendations_frame()
        return pd.concat(parts, ignore_index=True)

    def _store_users_in_cache(self, user_ids: set[str], merged: pd.DataFrame) -> None:
        # After a successful replace, every id in user_ids was written. Missing from
        # merged means cleared rows — store empty frames (key present = loaded).
        if merged.empty or USER_COLUMN not in merged.columns:
            by_user: dict[str, pd.DataFrame] = {}
        else:
            frame = merged.copy()
            frame[USER_COLUMN] = frame[USER_COLUMN].astype(str)
            by_user = {user_id: group.copy() for user_id, group in frame.groupby(USER_COLUMN, sort=False)}
        for user_id in user_ids:
            self._cache_put(
                user_id,
                by_user.get(user_id, empty_recommendations_frame()),
                protect=user_ids,
            )

    def _event_weight(self, event_type: str, quantity: int) -> float:
        if self._feature_config is None:
            return float(quantity)
        weights = self._feature_config.event_weights
        if event_type not in weights:
            return 0.0
        weight = float(weights[event_type])
        if event_type in self._feature_config.quantity_scaled_events:
            weight *= math.log1p(max(quantity, 0))
        return weight

    def _known_event_type(self, event_type: str) -> bool:
        if self._feature_config is None:
            return True
        return event_type in self._feature_config.event_weights

    def _popular_ranking(self, batch: pd.DataFrame) -> pd.DataFrame:
        scores: dict[str, float] = {}
        for row in batch.itertuples(index=False):
            weight = self._event_weight(str(row.event_type), int(row.quantity))
            if weight == 0.0:
                continue
            item_id = str(row.item_id)
            scores[item_id] = scores.get(item_id, 0.0) + weight
        if not scores:
            return pd.DataFrame(columns=[ITEM_COLUMN, SCORE_COLUMN, SOURCE_COLUMN])
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: self._top_k]
        return pd.DataFrame(
            [
                {ITEM_COLUMN: item_id, SCORE_COLUMN: score, SOURCE_COLUMN: POPULAR_SOURCE}
                for item_id, score in ranked
            ]
        )

    def _latest_ranking(self, batch: pd.DataFrame) -> pd.DataFrame:
        if batch.empty:
            return pd.DataFrame(columns=[ITEM_COLUMN, SCORE_COLUMN, SOURCE_COLUMN])
        frame = batch.copy()
        frame[ITEM_COLUMN] = frame[ITEM_COLUMN].astype(str)
        if self._feature_config is not None:
            frame = frame[frame["event_type"].astype(str).map(self._known_event_type)]
            if frame.empty:
                return pd.DataFrame(columns=[ITEM_COLUMN, SCORE_COLUMN, SOURCE_COLUMN])
        latest = (
            frame.sort_values("occurred_at", ascending=False)
            .drop_duplicates(subset=[ITEM_COLUMN], keep="first")
            .head(self._top_k)
        )
        rows = []
        for rank, row in enumerate(latest.itertuples(index=False), start=1):
            rows.append(
                {
                    ITEM_COLUMN: str(row.item_id),
                    SCORE_COLUMN: float(self._top_k - rank + 1),
                    SOURCE_COLUMN: LATEST_SOURCE,
                }
            )
        return (
            pd.DataFrame(rows) if rows else pd.DataFrame(columns=[ITEM_COLUMN, SCORE_COLUMN, SOURCE_COLUMN])
        )

    def _merge_user_rows(
        self,
        user_id: str,
        prior: pd.DataFrame,
        popular: pd.DataFrame,
        latest: pd.DataFrame,
        batch: pd.DataFrame,
    ) -> pd.DataFrame:
        if not prior.empty and SOURCE_COLUMN in prior.columns:
            mask = prior[SOURCE_COLUMN].astype(str).map(_is_preserved_source)
            preserved = prior.loc[mask].copy()
        else:
            preserved = prior.iloc[0:0] if not prior.empty else prior

        user_batch = batch[batch[USER_COLUMN].astype(str) == user_id]
        if self._feature_config is not None and not user_batch.empty:
            user_batch = user_batch[user_batch["event_type"].astype(str).map(self._known_event_type)]
        boost_items = (
            user_batch.sort_values("occurred_at", ascending=False)[ITEM_COLUMN]
            .astype(str)
            .drop_duplicates()
            .tolist()
            if not user_batch.empty
            else []
        )
        boost_slots = max(1, int(self._top_k * _BOOST_SLOT_FRACTION)) if boost_items else 0
        boost = pd.DataFrame(
            [
                {
                    ITEM_COLUMN: item_id,
                    SCORE_COLUMN: float(len(boost_items) - index),
                    SOURCE_COLUMN: INCREMENTAL_SOURCE,
                }
                for index, item_id in enumerate(boost_items[:boost_slots])
            ]
        )
        preserve_cap = max(0, self._top_k - boost_slots)
        if not preserved.empty:
            preserved = preserved.head(preserve_cap)

        # Boost first so reserved slots win; then preserved; then popular/latest fill.
        parts = [frame for frame in (boost, preserved, popular, latest) if not frame.empty]
        combined = pd.concat(parts, ignore_index=True) if parts else empty_recommendations_frame()
        if combined.empty:
            return empty_recommendations_frame()
        combined[USER_COLUMN] = user_id
        combined[ITEM_COLUMN] = combined[ITEM_COLUMN].astype(str)
        combined = combined.drop_duplicates(subset=[ITEM_COLUMN], keep="first").head(self._top_k)
        combined[RANK_COLUMN] = range(1, len(combined) + 1)
        return combined[[USER_COLUMN, ITEM_COLUMN, RANK_COLUMN, SCORE_COLUMN, SOURCE_COLUMN]].reset_index(
            drop=True
        )

    def _cold_start_rows(
        self,
        prior: pd.DataFrame,
        popular: pd.DataFrame,
        latest: pd.DataFrame,
    ) -> pd.DataFrame:
        # Prefers batch popular/latest, then keeps prior cold-start fill.
        parts = [frame for frame in (popular, latest, prior) if not frame.empty]
        combined = pd.concat(parts, ignore_index=True) if parts else empty_recommendations_frame()
        if combined.empty:
            return empty_recommendations_frame()
        if ITEM_COLUMN in combined.columns:
            combined = combined.drop_duplicates(subset=[ITEM_COLUMN], keep="first").head(self._top_k)
        combined[USER_COLUMN] = COLD_START_USER_ID
        combined[RANK_COLUMN] = range(1, len(combined) + 1)
        return combined[[USER_COLUMN, ITEM_COLUMN, RANK_COLUMN, SCORE_COLUMN, SOURCE_COLUMN]].reset_index(
            drop=True
        )
