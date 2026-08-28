"""Cheap incremental update: refresh popular/latest slices; write-through."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Collection, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

import pandas as pd

from cicerone.blending import COLD_START_USER_ID, LATEST_SOURCE, POPULAR_SOURCE
from cicerone.config import IOSettings
from cicerone.events.base import NormalizedEvent
from cicerone.events.normalize import events_to_dataframe
from cicerone.events.online_result import OnlineRefreshResult, empty_online_rows
from cicerone.events.store import (
    empty_recommendations_frame,
    load_recommendations_for_users,
)
from cicerone.feature_config import FeatureConfig
from cicerone.io.base import OutputSink
from cicerone.io.recommendation_reader import (
    ITEM_COLUMN,
    RANK_COLUMN,
    SCORE_COLUMN,
    SOURCE_COLUMN,
    USER_COLUMN,
)
from cicerone.io.recommendation_schema import (
    FALLBACK_VARIANT,
    REASONS_COLUMN,
    VARIANT_COLUMN,
    collapse_mixed_variants,
    pick_fallback_variant,
    recommendation_output_columns,
)
from cicerone.locks import LockLostError
from cicerone.reasons import dump_source_reasons
from cicerone.weighting import event_row_weights

logger = logging.getLogger(__name__)

INCREMENTAL_SOURCE = "incremental"
_PRESERVE_LABELS = frozenset(
    {
        "personalized",
        "item_based",
        "sequential",
        "content_fallback",
        "blended",
    }
)
# Reserve slots so recent interactions can enter top-K even when preserved rows fill it.
_BOOST_SLOT_FRACTION = 0.3
# Bound in-process per-user frames for long-lived serve workers.
DEFAULT_USER_CACHE_MAX_SIZE = 2048


def _source_parts(source: str) -> set[str]:
    return {part for part in source.split("+") if part}


def _is_preserved_source(source: str) -> bool:
    return bool(_source_parts(source) & _PRESERVE_LABELS)


def _overlaps_source_parts(source: str, parts: set[str]) -> bool:
    return bool(_source_parts(source) & parts)


class OnlineRefresher(Protocol):
    def refresh(self, events: Sequence[NormalizedEvent]) -> OnlineRefreshResult: ...

    def invalidate(self) -> None: ...

    def commit(self) -> None: ...

    def abort(self) -> None: ...


class IncrementalUpdater:
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
        explain_enabled: bool = True,
        publisher: Any | None = None,
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
        if self._publisher is not None:
            self._publisher.publish(merged)

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

    def _frames_by_user(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if frame.empty or USER_COLUMN not in frame.columns:
            return {}
        # One assign for str user ids; reset_index materializes each group (no extra .copy()).
        keyed = frame.assign(**{USER_COLUMN: frame[USER_COLUMN].astype(str)})
        return {
            user_id: group.reset_index(drop=True) for user_id, group in keyed.groupby(USER_COLUMN, sort=False)
        }

    def _load_users(self, user_ids: set[str]) -> pd.DataFrame:
        missing = sorted(user_id for user_id in user_ids if user_id not in self._cached_by_user)
        if missing:
            loaded = load_recommendations_for_users(self._output_settings, missing)
            by_user = self._frames_by_user(loaded)
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
        by_user = self._frames_by_user(merged)
        for user_id in user_ids:
            self._cache_put(
                user_id,
                by_user.get(user_id, empty_recommendations_frame()),
                protect=user_ids,
            )

    def _row_signal_weights(self, batch: pd.DataFrame) -> pd.Series:
        if self._feature_config is None:
            return pd.to_numeric(batch["quantity"], errors="coerce").fillna(0.0)
        return event_row_weights(
            batch["event_type"].astype(str),
            batch["quantity"],
            event_weights=self._feature_config.event_weights,
            quantity_scaled_events=self._feature_config.quantity_scaled_events,
        ).fillna(0.0)

    def _aligned_weights(self, batch: pd.DataFrame, weights: pd.Series | None) -> pd.Series:
        if weights is None:
            return self._row_signal_weights(batch)
        return weights.reindex(batch.index)

    def _signal_rows(self, batch: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame:
        if batch.empty:
            return batch
        aligned = self._aligned_weights(batch, weights)
        keep = aligned > 0
        scored = batch.loc[keep]
        if scored.empty:
            return scored
        return scored.assign(_signal_weight=aligned.loc[keep])

    def _popular_ranking(self, batch: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame:
        empty = pd.DataFrame(columns=[ITEM_COLUMN, SCORE_COLUMN, SOURCE_COLUMN])
        scored = self._signal_rows(batch, weights)
        if scored.empty:
            return empty
        summed = scored.groupby(scored["item_id"].astype(str), sort=False)["_signal_weight"].sum()
        summed = summed.loc[summed > 0]
        if summed.empty:
            return empty
        ranked = summed.reset_index()
        ranked.columns = [ITEM_COLUMN, SCORE_COLUMN]
        ranked = ranked.sort_values(
            [SCORE_COLUMN, ITEM_COLUMN], ascending=[False, True], kind="mergesort"
        ).head(self._top_k)
        ranked[SOURCE_COLUMN] = POPULAR_SOURCE
        if self._explain_enabled:
            ranked[REASONS_COLUMN] = [
                dump_source_reasons(POPULAR_SOURCE, rank=rank) for rank in range(1, len(ranked) + 1)
            ]
        return ranked.reset_index(drop=True)

    def _latest_ranking(self, batch: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame:
        if batch.empty:
            return pd.DataFrame(columns=[ITEM_COLUMN, SCORE_COLUMN, SOURCE_COLUMN])
        frame = self._signal_rows(batch, weights)
        if frame.empty:
            return pd.DataFrame(columns=[ITEM_COLUMN, SCORE_COLUMN, SOURCE_COLUMN])
        frame = frame.copy()
        frame[ITEM_COLUMN] = frame[ITEM_COLUMN].astype(str)
        latest = (
            frame.sort_values("occurred_at", ascending=False, kind="mergesort")
            .drop_duplicates(subset=[ITEM_COLUMN], keep="first")
            .head(self._top_k)
        )
        rows = []
        for rank, row in enumerate(latest.itertuples(index=False), start=1):
            item = {
                ITEM_COLUMN: str(row.item_id),
                SCORE_COLUMN: float(self._top_k - rank + 1),
                SOURCE_COLUMN: LATEST_SOURCE,
            }
            if self._explain_enabled:
                item[REASONS_COLUMN] = dump_source_reasons(LATEST_SOURCE, rank=rank)
            rows.append(item)
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
        weights: pd.Series | None = None,
        online_rows: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        variants = self._variants_for()
        if not variants:
            prior = collapse_mixed_variants(prior)
            merged = self._merge_one_list(
                user_id, prior, popular, latest, batch, weights, online_rows=online_rows
            )
            return self._stamp_collapsed_variant(merged, prior)
        has_variant = VARIANT_COLUMN in prior.columns and not prior.empty
        primary = FALLBACK_VARIANT if FALLBACK_VARIANT in variants else variants[0]
        parts = []
        for variant in variants:
            prior_slice = (
                prior[prior[VARIANT_COLUMN].astype(str) == variant]
                if has_variant
                else (prior if variant == primary else empty_recommendations_frame())
            )
            merged = self._merge_one_list(
                user_id, prior_slice, popular, latest, batch, weights, online_rows=online_rows
            )
            if merged.empty:
                continue
            merged = merged.copy()
            merged[VARIANT_COLUMN] = variant
            parts.append(merged)
        if not parts:
            return empty_recommendations_frame()
        return pd.concat(parts, ignore_index=True)

    def _variants_for(self) -> tuple[str, ...]:
        return self._variant_names

    def _stamp_collapsed_variant(self, merged: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
        if merged.empty or VARIANT_COLUMN not in prior.columns or prior.empty:
            return merged
        pick = pick_fallback_variant(prior[VARIANT_COLUMN].tolist())
        if pick is None:
            return merged
        stamped = merged.copy()
        stamped[VARIANT_COLUMN] = pick
        return stamped

    def _merge_one_list(
        self,
        user_id: str,
        prior: pd.DataFrame,
        popular: pd.DataFrame,
        latest: pd.DataFrame,
        batch: pd.DataFrame,
        weights: pd.Series | None = None,
        online_rows: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not prior.empty and SOURCE_COLUMN in prior.columns:
            mask = prior[SOURCE_COLUMN].astype(str).map(_is_preserved_source)
            preserved = prior.loc[mask].copy()
        else:
            preserved = prior.iloc[0:0] if not prior.empty else prior
        online_part, kept = self._split_online_preserved(preserved, online_rows)

        user_batch = self._signal_rows(batch[batch[USER_COLUMN].astype(str) == user_id], weights)
        has_signal = not user_batch.empty
        preserved_ids: set[str] = set()
        if not kept.empty:
            preserved_ids.update(kept[ITEM_COLUMN].astype(str))
        if not online_part.empty:
            preserved_ids.update(online_part[ITEM_COLUMN].astype(str))
        boost_items = (
            user_batch.sort_values("occurred_at", ascending=False, kind="mergesort")[ITEM_COLUMN]
            .astype(str)
            .drop_duplicates()
            .tolist()
            if has_signal
            else []
        )
        boost_items = [item_id for item_id in boost_items if item_id not in preserved_ids]
        boost_slots = max(1, int(self._top_k * _BOOST_SLOT_FRACTION)) if boost_items else 0
        boost = pd.DataFrame(
            [
                {
                    ITEM_COLUMN: item_id,
                    SCORE_COLUMN: float(len(boost_items) - index),
                    SOURCE_COLUMN: INCREMENTAL_SOURCE,
                    **(
                        {REASONS_COLUMN: dump_source_reasons(INCREMENTAL_SOURCE, rank=index + 1)}
                        if self._explain_enabled
                        else {}
                    ),
                }
                for index, item_id in enumerate(boost_items[:boost_slots])
            ]
        )
        preserve_cap = max(0, self._top_k - boost_slots)
        if not kept.empty:
            if RANK_COLUMN in kept.columns:
                kept = kept.sort_values(RANK_COLUMN, kind="mergesort")
            kept = kept.head(preserve_cap)
        preserved_parts = [frame for frame in (online_part, kept) if not frame.empty]
        preserved = (
            pd.concat(preserved_parts, ignore_index=True)
            if preserved_parts
            else empty_recommendations_frame()
        )

        # Batch-global popular/latest only for users with signal or preserved rows
        # (unknown/zero-weight events must not rewrite popular-only users).
        use_global = has_signal or not preserved.empty
        parts = [boost, preserved]
        if use_global:
            parts.extend((popular, latest))
        parts = [frame for frame in parts if not frame.empty]
        combined = pd.concat(parts, ignore_index=True) if parts else empty_recommendations_frame()
        if combined.empty:
            return empty_recommendations_frame()
        combined[USER_COLUMN] = user_id
        combined[ITEM_COLUMN] = combined[ITEM_COLUMN].astype(str)
        combined = combined.drop_duplicates(subset=[ITEM_COLUMN], keep="first").head(self._top_k)
        combined[RANK_COLUMN] = range(1, len(combined) + 1)
        return combined[recommendation_output_columns(combined)].reset_index(drop=True)

    def _split_online_preserved(
        self, preserved: pd.DataFrame, online_rows: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        empty = preserved.iloc[0:0] if not preserved.empty else preserved
        if online_rows is None or online_rows.empty:
            return empty, preserved
        online = online_rows.copy()
        if SOURCE_COLUMN in online.columns:
            online = online.loc[online[SOURCE_COLUMN].astype(str).map(_is_preserved_source)]
        if online.empty:
            return empty, preserved
        online_parts: set[str] = set()
        if SOURCE_COLUMN in online.columns:
            for label in online[SOURCE_COLUMN].astype(str):
                online_parts.update(_source_parts(label))
        if preserved.empty:
            kept = preserved
        elif SOURCE_COLUMN in preserved.columns:
            drop = (
                preserved[SOURCE_COLUMN]
                .astype(str)
                .map(lambda source: _overlaps_source_parts(source, online_parts))
            )
            kept = preserved.loc[~drop]
        else:
            kept = empty
        return online, kept

    def _cold_start_rows(
        self,
        prior: pd.DataFrame,
        popular: pd.DataFrame,
        latest: pd.DataFrame,
    ) -> pd.DataFrame:
        variants = self._variants_for()
        if not variants:
            prior = collapse_mixed_variants(prior)
            return self._stamp_collapsed_variant(self._cold_start_one_list(prior, popular, latest), prior)
        has_variant = VARIANT_COLUMN in prior.columns and not prior.empty
        primary = FALLBACK_VARIANT if FALLBACK_VARIANT in variants else variants[0]
        parts = []
        for variant in variants:
            prior_slice = (
                prior[prior[VARIANT_COLUMN].astype(str) == variant]
                if has_variant
                else (prior if variant == primary else empty_recommendations_frame())
            )
            merged = self._cold_start_one_list(prior_slice, popular, latest)
            if merged.empty:
                continue
            merged = merged.copy()
            merged[VARIANT_COLUMN] = variant
            parts.append(merged)
        if not parts:
            return empty_recommendations_frame()
        return pd.concat(parts, ignore_index=True)

    def _cold_start_one_list(
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
        return combined[recommendation_output_columns(combined)].reset_index(drop=True)
