"""Cheap incremental update: refresh popular/latest slices; write-through."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Protocol

import pandas as pd

from cicerone.blending import COLD_START_USER_ID
from cicerone.config import IOSettings
from cicerone.events.base import NormalizedEvent
from cicerone.events.normalize import events_to_dataframe
from cicerone.events.online_result import OnlineRefreshResult, empty_online_rows
from cicerone.events.store import empty_recommendations_frame
from cicerone.events.updater_cache import UpdaterUserCache
from cicerone.events.updater_merge import (
    INCREMENTAL_SOURCE,  # noqa: F401
    UpdaterMerge,
    _is_preserved_source,
)
from cicerone.events.updater_ranking import UpdaterRanking
from cicerone.feature_config import FeatureConfig
from cicerone.io.base import OutputSink
from cicerone.io.recommendation_reader import SOURCE_COLUMN, USER_COLUMN
from cicerone.io.recommendation_schema import recommendation_output_columns
from cicerone.locks import LockLostError
from cicerone.publish.base import RecommendationPublisher

logger = logging.getLogger(__name__)

# Bound in-process per-user frames for long-lived serve workers.
DEFAULT_USER_CACHE_MAX_SIZE = 2048


class OnlineRefresher(Protocol):
    def refresh(self, events: Sequence[NormalizedEvent]) -> OnlineRefreshResult: ...

    def invalidate(self) -> None: ...

    def commit(self) -> None: ...

    def abort(self) -> None: ...


class IncrementalUpdater(UpdaterUserCache, UpdaterRanking, UpdaterMerge):
    """Write-through popular/latest slices for affected users; preserves personalized rows."""

    def __init__(
        self,
        *,
        sink: OutputSink,
        output_settings: IOSettings,
        feature_config: FeatureConfig | None,
        top_k: int,
        busy_check: Callable[[], bool] | None = None,
        write_busy_check: Callable[[], bool] | None = None,
        on_success: Callable[[], None] | None = None,
        fence_check: Callable[[], bool] | None = None,
        user_cache_max_size: int = DEFAULT_USER_CACHE_MAX_SIZE,
        online: OnlineRefresher | None = None,
        variant_names: Sequence[str] = (),
        assign_variant: Callable[[str], str | None] | None = None,
        explain_enabled: bool = True,
        publisher: RecommendationPublisher | None = None,
    ):
        if user_cache_max_size < 1:
            raise ValueError("user_cache_max_size must be >= 1")
        self._sink = sink
        self._output_settings = output_settings
        self._feature_config = feature_config
        self._top_k = top_k
        self._busy_check = busy_check
        self._write_busy_check = busy_check if write_busy_check is None else write_busy_check
        self._on_success = on_success
        self._fence_check = fence_check
        self._online = online
        self._last_success_at: datetime | None = None
        self._events_applied = 0
        self._user_cache_max_size = user_cache_max_size
        self._cached_by_user: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._variant_names = tuple(str(name) for name in variant_names)
        self._assign_variant = assign_variant
        self._explain_enabled = explain_enabled
        self._publisher = publisher

    @property
    def last_success_at(self) -> datetime | None:
        return self._last_success_at

    @property
    def events_applied(self) -> int:
        return self._events_applied

    @property
    def cached_user_ids(self) -> frozenset[str]:
        """User ids currently held in the in-process LRU recommendation cache."""
        return frozenset(self._cached_by_user)

    def invalidate_cache(self) -> None:
        self._cached_by_user.clear()
        if self._online is not None:
            self._online.invalidate()

    def retrain_busy(self) -> bool:
        return self._busy_check is not None and self._busy_check()

    def persist_online(self) -> None:
        if self._write_busy_check is not None and self._write_busy_check():
            logger.info("Skipping online persist: full retrain in progress")
            self._abort_online()
            return
        self._commit_online()

    def abort_online(self) -> None:
        self._abort_online()

    def apply(self, events: Sequence[NormalizedEvent], *, persist_online: bool = True) -> int:
        if not events:
            return 0
        if self.retrain_busy():
            # Retrain may rewrite output; drop cache so the next apply reloads.
            self.invalidate_cache()
            logger.info("Skipping incremental update: full retrain in progress")
            return 0

        batch = events_to_dataframe(events)
        weights = self._row_signal_weights(batch)
        affected_users = sorted(set(batch[USER_COLUMN].astype(str)))
        affected_set = set(affected_users) | {COLD_START_USER_ID}
        existing = self._load_users(affected_set)

        if USER_COLUMN in existing.columns and not existing.empty:
            existing = existing.copy()
            existing[USER_COLUMN] = existing[USER_COLUMN].astype(str)

        popular_ranking = self._popular_ranking(batch, weights)
        latest_ranking = self._latest_ranking(batch, weights)
        online_result = self._refresh_online(events)
        online_by_user = {} if online_result.sequential_skipped else self._online_rows_by_user(online_result)

        by_user = (
            {user_id: group for user_id, group in existing.groupby(USER_COLUMN, sort=False)}
            if not existing.empty
            else {}
        )
        batch_by_user = (
            {
                str(user_id): group
                for user_id, group in batch.groupby(batch[USER_COLUMN].astype(str), sort=False)
            }
            if not batch.empty
            else {}
        )

        frames: list[pd.DataFrame] = []
        replace_ids: list[str] = []
        empty_user_batch = batch.iloc[0:0]
        for user_id in affected_users:
            prior = by_user.get(user_id, empty_recommendations_frame())
            user_batch = batch_by_user.get(user_id, empty_user_batch)
            merged_user = self._merge_user_rows(
                user_id,
                prior,
                popular_ranking,
                latest_ranking,
                user_batch,
                weights,
                online_rows=online_by_user.get(user_id),
            )
            if merged_user.empty:
                continue
            frames.append(merged_user)
            replace_ids.append(user_id)
        prior_cold = by_user.get(COLD_START_USER_ID, empty_recommendations_frame())
        cold = self._cold_start_rows(prior_cold, popular_ranking, latest_ranking)
        if not cold.empty:
            frames.append(cold)
            replace_ids.append(COLD_START_USER_ID)

        if not replace_ids:
            self._abort_online()
            now = datetime.now(UTC)
            self._last_success_at = now
            self._events_applied += len(events)
            if self._on_success is not None:
                self._on_success()
            logger.info(
                "Incremental update skipped write: %d event(s) had no ranking signal",
                len(events),
            )
            return len(events)

        merged = pd.concat(frames, ignore_index=True)
        merged = merged[recommendation_output_columns(merged)]
        if not self._ensure_write_allowed():
            self._abort_online()
            return 0
        self._ensure_fence()
        n_users = self._sink.replace_recommendations_for_users(merged, user_ids=sorted(set(replace_ids)))
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
        if self._online is not None:
            manifest["online_fit_partial_epochs"] = online_result.fit_partial_epochs
            manifest["online_users_refreshed"] = online_result.users_refreshed
            manifest["online_events_dropped_unknown"] = online_result.events_dropped_unknown
        self._ensure_fence()
        self._sink.write_manifest(manifest)
        if self._publisher is not None:
            self._publisher.publish(merged)
        self._store_users_in_cache(set(replace_ids), merged)
        if persist_online:
            self._commit_online()
        self._last_success_at = now
        self._events_applied += len(events)
        if self._on_success is not None:
            self._on_success()
        logger.info(
            "Incremental update wrote recommendations for %d user(s) from %d event(s)",
            len(replace_ids),
            len(events),
        )
        return len(events)

    def _commit_online(self) -> None:
        if self._online is None:
            return
        commit = getattr(self._online, "commit", None)
        if callable(commit):
            commit()

    def _abort_online(self) -> None:
        if self._online is None:
            return
        abort = getattr(self._online, "abort", None)
        if callable(abort):
            abort()

    def _refresh_online(self, events: Sequence[NormalizedEvent]) -> OnlineRefreshResult:
        if self._online is None or self._variant_names:
            return OnlineRefreshResult(rows=empty_online_rows())
        try:
            return self._online.refresh(events)
        except LockLostError:
            self._abort_online()
            raise
        except Exception:
            self._abort_online()
            logger.exception("Online collaborative refresh failed; nacking incremental batch")
            raise

    def _online_rows_by_user(self, result: OnlineRefreshResult) -> dict[str, pd.DataFrame]:
        frame = result.rows
        if frame is None or frame.empty or USER_COLUMN not in frame.columns:
            return {}
        keyed = frame.assign(**{USER_COLUMN: frame[USER_COLUMN].astype(str)})
        if SOURCE_COLUMN in keyed.columns:
            mask = keyed[SOURCE_COLUMN].astype(str).map(_is_preserved_source)
            keyed = keyed.loc[mask]
        if keyed.empty:
            return {}
        return {
            user_id: group.reset_index(drop=True) for user_id, group in keyed.groupby(USER_COLUMN, sort=False)
        }

    def _ensure_write_allowed(self) -> bool:
        if self._write_busy_check is not None and self._write_busy_check():
            self.invalidate_cache()
            logger.info("Skipping incremental write: full retrain in progress")
            return False
        self._ensure_fence()
        return True

    def _ensure_fence(self) -> None:
        if self._fence_check is not None and not self._fence_check():
            raise LockLostError("events apply lock lost before write")
