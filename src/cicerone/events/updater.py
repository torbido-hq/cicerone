"""Cheap incremental update: refresh popular/latest slices; write-through."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from cicerone.blending import COLD_START_USER_ID
from cicerone.capability import CapabilityMeta
from cicerone.config import IOSettings
from cicerone.errors import OPTIONAL_IO_ERRORS, PUBLISH_ERRORS, log_caught
from cicerone.events.base import NormalizedEvent
from cicerone.events.normalize import events_to_dataframe
from cicerone.events.online_result import OnlineRefreshResult, empty_online_rows
from cicerone.events.post_write import PostWriteRepair
from cicerone.events.store import empty_recommendations_frame
from cicerone.events.updater_cache import UpdaterUserCache
from cicerone.events.updater_merge import (
    INCREMENTAL_SOURCE,  # noqa: F401
    UpdaterMerge,
    is_preserved_source,
)
from cicerone.events.updater_policy import incremental_allowlists, incremental_needs_users_frame
from cicerone.events.updater_ranking import UpdaterRanking
from cicerone.feature_config import FeatureConfig
from cicerone.io.applied_events import replay_key
from cicerone.io.base import AppliedEventLog, OutputSink, ensure_sink_writer_held, recommendations_write_guard
from cicerone.io.recommendation_reader import SOURCE_COLUMN, USER_COLUMN
from cicerone.io.recommendation_schema import recommendation_output_columns
from cicerone.locks import LockLostError, WriterLockBusyError
from cicerone.publish.base import (
    RecommendationPublisher,
    publish_recommendations,
    require_incremental_publisher,
)
from cicerone.publish.sidecar import (
    latest_manifest_generated_at,
    log_sidecar_generation_skip,
    sidecar_generation_current,
)

logger = logging.getLogger(__name__)

# After the incremental write, raising would nack the applied flush and re-apply it on redelivery.
_SIDECAR_PUBLISH_ERRORS: tuple[type[BaseException], ...] = (*PUBLISH_ERRORS, TypeError)
_ONLINE_REFRESH_ERRORS: tuple[type[BaseException], ...] = (
    *OPTIONAL_IO_ERRORS,
    *PUBLISH_ERRORS,
)

# Bound in-process per-user frames for long-lived serve workers.
DEFAULT_USER_CACHE_MAX_SIZE = 2048


class OnlineRefresher(Protocol):
    def refresh(self, events: Sequence[NormalizedEvent]) -> OnlineRefreshResult: ...

    def invalidate(self) -> None: ...


@runtime_checkable
class CommittingOnlineRefresher(Protocol, metaclass=CapabilityMeta):
    def commit(self) -> None: ...


@runtime_checkable
class AbortingOnlineRefresher(Protocol, metaclass=CapabilityMeta):
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
        on_success: Callable[[], object] | None = None,
        fence_check: Callable[[], bool] | None = None,
        user_cache_max_size: int = DEFAULT_USER_CACHE_MAX_SIZE,
        online: OnlineRefresher | None = None,
        variant_names: Sequence[str] = (),
        assign_variant: Callable[[str], str | None] | None = None,
        explain_enabled: bool = True,
        publisher: RecommendationPublisher | None = None,
        items_provider: Callable[[], pd.DataFrame | None] | None = None,
        users_provider: Callable[[], pd.DataFrame | None] | None = None,
        variant_feature_configs: Mapping[str, FeatureConfig] | None = None,
    ):
        if user_cache_max_size < 1:
            raise ValueError("user_cache_max_size must be >= 1")
        self._sink = sink
        self._output_settings = output_settings
        self._feature_config = feature_config
        self._top_k = top_k
        self._items_provider = items_provider
        self._users_provider = users_provider
        self._variant_feature_configs = {
            str(name): config for name, config in (variant_feature_configs or {}).items()
        }
        self._busy_check = busy_check
        self._write_busy_check = busy_check if write_busy_check is None else write_busy_check
        self._on_success = on_success
        self._fence_check = fence_check
        self._online = online
        self._last_success_at: datetime | None = None
        self._events_applied = 0
        self._replay_key: Callable[[NormalizedEvent], str | None] = lambda event: replay_key(
            "id", event.event_id
        )
        self._user_cache_max_size = user_cache_max_size
        self._cached_by_user: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._variant_names = tuple(str(name) for name in variant_names)
        self._assign_variant = assign_variant
        self._explain_enabled = explain_enabled
        self._publisher = require_incremental_publisher(publisher)
        self._repair = PostWriteRepair()
        self._last_publish_failed = False
        self._last_apply_written = False

    @property
    def last_success_at(self) -> datetime | None:
        return self._last_success_at

    def use_replay_key(self, key: Callable[[NormalizedEvent], str | None]) -> None:
        """Key under which applied events are recorded with the write; None leaves an event at-least-once."""
        self._replay_key = key

    def applied_before(self, events: Sequence[NormalizedEvent]) -> list[NormalizedEvent]:
        """Events already recorded by an earlier write, e.g. redelivered after a restart."""
        if not events or not isinstance(self._sink, AppliedEventLog):
            return []
        keyed = [(event, key) for event in events if (key := self._replay_key(event)) is not None]
        if not keyed:
            return []
        recorded = self._sink.applied_event_keys([key for _event, key in keyed])
        return [event for event, key in keyed if key in recorded]

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
        with recommendations_write_guard(self._sink):
            if self._write_busy_check is not None and self._write_busy_check():
                logger.info("Skipping online persist: full retrain in progress")
                self._abort_online()
                return
            self._ensure_fence()
            self._commit_online()

    def abort_online(self) -> None:
        self._abort_online()

    def apply(self, events: Sequence[NormalizedEvent], *, persist_online: bool = True) -> int:
        self._last_apply_written = False
        self._last_publish_failed = False
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
        popular_ranking = self._popular_ranking(batch, weights)
        latest_ranking = self._latest_ranking(batch, weights)
        online_result = self._refresh_online(events)
        online_by_user = {} if online_result.sequential_skipped else self._online_rows_by_user(online_result)

        pending_publish: tuple[pd.DataFrame, str, list[str]] | None = None
        written: tuple[dict[str, Any], list[str]] | None = None
        done: set[str] = set()

        def _persist() -> int:
            nonlocal pending_publish, written
            if not self._ensure_write_allowed():
                self._abort_online()
                return 0
            merged, replace_ids = self._merge_affected(
                batch,
                weights,
                affected_users,
                affected_set,
                popular_ranking,
                latest_ranking,
                online_by_user,
            )
            if merged is None or not replace_ids:
                # Nothing is written or kept online, so a redelivery is a no-op too and needs no replay key.
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
            if not self._ensure_write_allowed():
                self._abort_online()
                return 0
            self._ensure_fence()
            user_ids = sorted(set(replace_ids))
            if isinstance(self._sink, AppliedEventLog):
                keys = [key for event in events if (key := self._replay_key(event)) is not None]
                n_users = self._sink.replace_recommendations_for_users(
                    merged, user_ids=user_ids, applied_keys=keys
                )
            else:
                n_users = self._sink.replace_recommendations_for_users(merged, user_ids=user_ids)
            self._last_apply_written = True
            now = datetime.now(UTC)
            manifest: dict[str, Any] = {
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
            written = (manifest, user_ids)
            # The rows are committed: bring the cache and reader up to date before any step that can fail.
            self._store_users_in_cache(set(replace_ids), merged)
            self._last_success_at = now
            self._events_applied += len(events)
            # A reader refresh returns False when it kept the previous recommendations.
            if self._on_success is None or self._on_success() is not False:
                done.add("refresh")
            self._ensure_fence()
            ensure_sink_writer_held(self._sink)
            self._sink.write_manifest(manifest)
            done.add("manifest")
            if persist_online:
                self._commit_online()
            logger.info(
                "Incremental update wrote recommendations for %d user(s) from %d event(s)",
                len(replace_ids),
                len(events),
            )
            pending_publish = (merged, str(manifest["generated_at"]), list(replace_ids))
            return len(events)

        try:
            # The publish stays under the write guard so a full write cannot land between its check and send;
            # the lease renews while held and publishers bound their own timeouts.
            with recommendations_write_guard(self._sink):
                applied = _persist()
                if pending_publish is not None and self._publish_sidecar(*pending_publish):
                    done.add("publish")
            self._last_publish_failed = written is not None and "publish" not in done
        finally:
            if written is not None:
                self._queue_repair(*written, done)
        return applied

    @property
    def post_write_pending(self) -> bool:
        return self._repair.pending

    @property
    def last_apply_written(self) -> bool:
        """The last ``apply`` wrote its rows, even if it then raised: its events must not be applied again."""
        return self._last_apply_written

    @property
    def last_publish_failed(self) -> bool:
        """The last ``apply`` wrote its rows but its sidecar publish failed (queued for repair)."""
        return self._last_publish_failed

    def _queue_repair(self, manifest: dict[str, Any], user_ids: list[str], done: set[str]) -> None:
        if "refresh" not in done and self._on_success is not None:
            self._repair.refresh = True
        if "manifest" not in done:
            # A newer manifest supersedes an older pending one.
            self._repair.manifest = manifest
        if "publish" not in done and self._publisher is not None:
            self._repair.publish_user_ids.update(user_ids)

    def repair_post_write(self) -> None:
        """Retry the reader refresh, manifest and sidecar publish that failed after an earlier write."""
        repair = self._repair
        if not repair.due() or self.retrain_busy():
            return
        try:
            if repair.refresh:
                repair.refresh = self._on_success is not None and self._on_success() is False
            if repair.manifest is not None:
                with recommendations_write_guard(self._sink):
                    self._ensure_fence()
                    ensure_sink_writer_held(self._sink)
                    generated_at = str(repair.manifest["generated_at"])
                    self._sink.write_manifest(repair.manifest, skip_if_newer_than=generated_at)
                repair.manifest = None
            if repair.publish_user_ids:
                user_ids = set(repair.publish_user_ids)
                # Publish what storage holds now, under the guard so no full write lands before the send.
                with recommendations_write_guard(self._sink):
                    generation = latest_manifest_generated_at(self._output_settings)
                    self._evict_users(user_ids)
                    current = self._load_users(user_ids)
                    published = self._publish_sidecar(current, generation, sorted(user_ids))
                if published:
                    repair.publish_user_ids -= user_ids
        finally:
            if repair.pending:
                repair.back_off()

    def _publish_sidecar(
        self, merged: pd.DataFrame, generated_at: str | None, replace_ids: list[str]
    ) -> bool:
        """False when the publish failed or its generation could not be confirmed, so it should be retried."""
        if self._publisher is None:
            return True
        try:
            self._ensure_fence()
            self._publisher.connect()
            self._ensure_fence()
            current: bool | None = True
            if generated_at is not None:
                current = sidecar_generation_current(self._output_settings, generated_at)
            if current:
                publish_recommendations(self._publisher, merged, user_ids=replace_ids)
            else:
                log_sidecar_generation_skip(current, incremental=True)
            return current is not None
        except (LockLostError, WriterLockBusyError):
            raise
        except _SIDECAR_PUBLISH_ERRORS as exc:
            log_caught("Incremental publish failed after successful write", exc, log=logger)
            return False

    def _merge_affected(
        self,
        batch: pd.DataFrame,
        weights: pd.Series,
        affected_users: list[str],
        affected_set: set[str],
        popular_ranking: pd.DataFrame,
        latest_ranking: pd.DataFrame,
        online_by_user: dict[str, pd.DataFrame],
    ) -> tuple[pd.DataFrame | None, list[str]]:
        self._evict_users(affected_set)
        existing = self._load_users(affected_set)
        if USER_COLUMN in existing.columns and not existing.empty:
            existing = existing.copy()
            existing[USER_COLUMN] = existing[USER_COLUMN].astype(str)
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
        items = self._items_provider() if self._items_provider is not None else None
        users = (
            self._users_provider() if self._users_provider is not None and self._needs_users_frame() else None
        )
        allowlists = incremental_allowlists(
            affected_users,
            feature_config=self._feature_config,
            items=items,
            users=users,
        )
        allowlists_by_variant = {
            name: incremental_allowlists(
                affected_users,
                feature_config=config,
                items=items,
                users=users,
            )
            for name, config in self._variant_feature_configs.items()
        }
        cold_allowed = incremental_allowlists(
            [COLD_START_USER_ID],
            feature_config=self._feature_config,
            items=items,
            users=None,
        ).get(COLD_START_USER_ID)
        cold_allowed_by_variant = {
            name: incremental_allowlists(
                [COLD_START_USER_ID],
                feature_config=config,
                items=items,
                users=None,
            ).get(COLD_START_USER_ID)
            for name, config in self._variant_feature_configs.items()
        }
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
                allowed=allowlists.get(user_id),
                allowed_by_variant={
                    name: lists.get(user_id) for name, lists in allowlists_by_variant.items()
                },
            )
            if not merged_user.empty:
                frames.append(merged_user)
                replace_ids.append(user_id)
            elif not prior.empty or not self._signal_rows(user_batch, weights).empty:
                replace_ids.append(user_id)
        prior_cold = by_user.get(COLD_START_USER_ID, empty_recommendations_frame())
        cold = self._cold_start_rows(
            prior_cold,
            popular_ranking,
            latest_ranking,
            allowed=cold_allowed,
            allowed_by_variant=cold_allowed_by_variant,
        )
        if not cold.empty:
            frames.append(cold)
            replace_ids.append(COLD_START_USER_ID)
        elif not prior_cold.empty:
            replace_ids.append(COLD_START_USER_ID)
        if not replace_ids:
            return None, []
        merged = pd.concat(frames, ignore_index=True) if frames else empty_recommendations_frame()
        return merged[recommendation_output_columns(merged)], replace_ids

    def _commit_online(self) -> None:
        if self._online is None:
            return
        if isinstance(self._online, CommittingOnlineRefresher):
            self._online.commit()

    def _abort_online(self) -> None:
        if self._online is None:
            return
        if isinstance(self._online, AbortingOnlineRefresher):
            self._online.abort()

    def _refresh_online(self, events: Sequence[NormalizedEvent]) -> OnlineRefreshResult:
        if self._online is None or self._variant_names:
            return OnlineRefreshResult(rows=empty_online_rows())
        try:
            return self._online.refresh(events)
        except LockLostError as exc:
            self._abort_online()
            logger.error("%s; aborting pending online fit", exc)
            raise
        except WriterLockBusyError as exc:
            self._abort_online()
            logger.info("%s; aborting pending online fit", exc)
            raise
        except _ONLINE_REFRESH_ERRORS:
            self._abort_online()
            logger.exception("Online collaborative refresh failed; aborting pending fit")
            raise
        except Exception:
            self._abort_online()
            raise

    def _online_rows_by_user(self, result: OnlineRefreshResult) -> dict[str, pd.DataFrame]:
        frame = result.rows
        if frame is None or frame.empty or USER_COLUMN not in frame.columns:
            return {}
        keyed = frame.assign(**{USER_COLUMN: frame[USER_COLUMN].astype(str)})
        if SOURCE_COLUMN in keyed.columns:
            mask = keyed[SOURCE_COLUMN].astype(str).map(is_preserved_source)
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
            raise LockLostError("events apply lock lost before write", kind="apply")

    def _needs_users_frame(self) -> bool:
        return incremental_needs_users_frame(self._feature_config, self._variant_feature_configs)
