"""Start the serve-process event worker (poll → micro-batch → write-through)."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from cicerone.config import Settings
from cicerone.config.constants import (
    DEFAULT_EVENTS_APPLY_LOCK_TTL_SECONDS,
    DEFAULT_EVENTS_RETRAIN_PROBE_TTL_SECONDS,
)
from cicerone.errors import PUBLISH_ERRORS
from cicerone.events.base import has_ephemeral_event_ids
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.ha import poll_without_apply_lock
from cicerone.events.registry import build_event_source
from cicerone.events.updater import IncrementalUpdater
from cicerone.events.updater_policy import incremental_needs_users_frame
from cicerone.events.webhook import WebhookEventSource
from cicerone.events.worker import EventWorker
from cicerone.experiment.assignment import assign_user, experiment_variant_names
from cicerone.experiment.recipes import resolve_variant_policy_configs
from cicerone.experiment.store import ExperimentStore
from cicerone.feature_config import FeatureConfig
from cicerone.io.base import RecommendationReader
from cicerone.io.factory import build_input_source, build_output_sink
from cicerone.locks import (
    LockBackend,
    WriteFence,
    build_lock_backend,
    build_output_writer_lock,
    events_apply_lock_key,
    has_distributed_lock,
)
from cicerone.publish import RecommendationPublisher, build_publisher
from cicerone.refreshed_value import RefreshedValue

logger = logging.getLogger(__name__)


@dataclass
class EventsRuntime:
    webhook_source: WebhookEventSource | None
    worker: EventWorker | None
    apply_lock: LockBackend | None = None
    publisher: RecommendationPublisher | None = None

    def stop(self) -> bool:
        stopped = True
        try:
            if self.worker is not None:
                stopped = self.worker.stop()
                if not stopped:
                    logger.warning("Event worker did not stop in time")
                    return False
            return True
        finally:
            try:
                # A pending worker-stop error must still propagate.
                _close_publisher(self.publisher, quiet=sys.exception() is not None)
            finally:
                self.publisher = None


def _assign_incremental_variant(settings: Settings) -> Callable[[str], str | None] | None:
    if not settings.experiment.enabled:
        return None
    store = ExperimentStore(settings.output)
    experiment_id = settings.experiment.id
    overlay = RefreshedValue(
        lambda: store.assignment_overlay(experiment_id),
        ttl_seconds=DEFAULT_EVENTS_RETRAIN_PROBE_TTL_SECONDS,
    )

    def assigned(user_id: str) -> str | None:
        return assign_user(
            settings, str(user_id), overlay.get(), lambda: experiment_variant_names(settings) or None
        )[1]

    return assigned


def _variant_feature_configs(
    settings: Settings, feature_config: FeatureConfig | None
) -> dict[str, FeatureConfig]:
    if feature_config is None or not settings.experiment.enabled:
        return {}
    return resolve_variant_policy_configs(settings, feature_config)


def _input_users_provider(settings: Settings) -> Callable[[], pd.DataFrame | None]:
    snapshot = build_input_source(settings.input).read_users()

    def read_users() -> pd.DataFrame | None:
        return snapshot

    return read_users


def _stop_worker_quietly(worker: EventWorker | None) -> None:
    if worker is None:
        return
    try:
        worker.stop()
    except Exception:
        # The startup error must still propagate.
        logger.exception("Event worker stop after startup failure")


def _close_publisher(publisher: RecommendationPublisher | None, *, quiet: bool = False) -> None:
    if publisher is None:
        return
    try:
        publisher.close()
    except PUBLISH_ERRORS:
        logger.exception("Recommendation publisher close failed")
    except Exception:
        if not quiet:
            raise
        logger.exception("Recommendation publisher close failed")


def start_events_runtime(
    settings: Settings,
    *,
    feature_config: FeatureConfig | None,
    reader: RecommendationReader,
    start_worker: bool = True,
) -> EventsRuntime:
    if not settings.events.enabled:
        return EventsRuntime(webhook_source=None, worker=None)

    source = build_event_source(settings.events.kind, settings.events.options)
    webhook_source: WebhookEventSource | None = None
    if settings.events.kind == "webhook":
        if not isinstance(source, WebhookEventSource):
            raise TypeError(f"expected WebhookEventSource, got {type(source).__name__}")
        webhook_source = source

    apply_lock: LockBackend | None = None
    retrain_probe: LockBackend | None = None
    if has_distributed_lock(settings):
        apply_lock = build_lock_backend(
            settings,
            lock_key=events_apply_lock_key(settings.trigger.lock_key),
            ttl_seconds=min(
                settings.trigger.lock_ttl_seconds,
                DEFAULT_EVENTS_APPLY_LOCK_TTL_SECONDS,
            ),
        )
        retrain_probe = build_lock_backend(settings)
        logger.info(
            "Events apply lease enabled (backend=%s, key=%s, ha=%s)",
            settings.trigger.lock_backend,
            events_apply_lock_key(settings.trigger.lock_key),
            settings.events.ha,
        )

    combined_busy = retrain_probe.is_locked if retrain_probe is not None else None

    sink = build_output_sink(
        settings.output,
        fence=WriteFence(
            build_output_writer_lock(settings),
            apply_lock.owned if apply_lock is not None else None,
            "events apply lock lost before write",
            "apply",
        ),
    )
    publisher = build_publisher(settings, connect=False)
    worker: EventWorker | None = None
    ready = False
    try:
        online = None
        if settings.events.online.enabled and settings.experiment.enabled:
            logger.warning("Online collaborative refresh skipped while [experiment] is enabled")
        elif settings.events.online.enabled:
            from cicerone.events.online import OnlineTrainer

            online = OnlineTrainer(
                sink=sink,
                top_k=settings.top_k,
                half_life_days=settings.half_life_days,
                fit_partial_epochs=settings.events.online.fit_partial_epochs,
                fit_min_events=settings.events.online.fit_min_events,
                max_extra_interactions=settings.events.online.max_extra_interactions,
                fence_check=(apply_lock.owned if apply_lock is not None else None),
                explain=settings.explain,
                max_workers=settings.max_workers,
                hmac_key=settings.output.artifact_hmac_key,
            )
            online.ensure_loaded()
            logger.info(
                "Online collaborative refresh enabled (fit_partial_epochs=%d, fit_min_events=%d)",
                settings.events.online.fit_partial_epochs,
                settings.events.online.fit_min_events,
            )
        variant_feature_configs = _variant_feature_configs(settings, feature_config)
        users_provider = (
            _input_users_provider(settings)
            if incremental_needs_users_frame(feature_config, variant_feature_configs)
            else None
        )
        updater = IncrementalUpdater(
            sink=sink,
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=settings.top_k,
            busy_check=(
                RefreshedValue(combined_busy, ttl_seconds=DEFAULT_EVENTS_RETRAIN_PROBE_TTL_SECONDS).get
                if combined_busy is not None
                else None
            ),
            write_busy_check=combined_busy,
            fence_check=(apply_lock.owned if apply_lock is not None else None),
            on_success=reader.refresh,
            online=online,
            variant_names=experiment_variant_names(settings),
            assign_variant=_assign_incremental_variant(settings),
            explain_enabled=settings.explain.enabled,
            publisher=publisher,
            items_provider=getattr(reader, "get_items", None),
            users_provider=users_provider,
            variant_feature_configs=variant_feature_configs,
        )
        buffer = MicroBatchBuffer(
            batch_size=settings.events.incremental.batch_size,
            batch_window_seconds=settings.events.incremental.batch_window_seconds,
            fingerprint_dedupe=has_ephemeral_event_ids(source),
        )
        worker = EventWorker(
            source,
            buffer,
            updater,
            poll_interval_seconds=settings.events.incremental.poll_interval_seconds,
            apply_lock=apply_lock,
            poll_without_lock=poll_without_apply_lock(settings.events.kind, settings.events.options),
        )
        if start_worker:
            worker.start()
            logger.info(
                "Event worker started (kind=%s, batch_size=%d, window=%ss, poll=%ss, ha=%s)",
                settings.events.kind,
                settings.events.incremental.batch_size,
                settings.events.incremental.batch_window_seconds,
                settings.events.incremental.poll_interval_seconds,
                settings.events.ha,
            )
        else:
            source.connect()
        if apply_lock is None:
            logger.warning(
                "Incremental events assume a single writer process "
                "(kind=%s, output=%s); set job.trigger.lock_backend = postgres|redis "
                "to serialize apply (events.ha = true still requires db for track)",
                settings.events.kind,
                settings.output.kind,
            )
        runtime = EventsRuntime(
            webhook_source=webhook_source, worker=worker, apply_lock=apply_lock, publisher=publisher
        )
        ready = True
        return runtime
    finally:
        if not ready:
            _stop_worker_quietly(worker)
            _close_publisher(publisher, quiet=True)
