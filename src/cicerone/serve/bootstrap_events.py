"""Start the serve-process event worker (poll → micro-batch → write-through)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from cicerone.config import Settings
from cicerone.config.constants import (
    DEFAULT_EVENTS_APPLY_LOCK_TTL_SECONDS,
    DEFAULT_EVENTS_RETRAIN_PROBE_TTL_SECONDS,
)
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.ha import poll_without_apply_lock
from cicerone.events.registry import build_event_source
from cicerone.events.store import dispose_recommendation_engines
from cicerone.events.updater import IncrementalUpdater
from cicerone.events.webhook import WebhookEventSource
from cicerone.events.worker import EventWorker
from cicerone.experiment.assignment import experiment_variant_names
from cicerone.feature_config import FeatureConfig
from cicerone.io.base import RecommendationReader
from cicerone.io.factory import build_output_sink
from cicerone.locks import LockBackend, build_lock_backend, events_apply_lock_key
from cicerone.publish import RecommendationPublisher, build_publisher

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
                    logger.warning("Event worker did not stop in time; skipping engine dispose")
                    return False
            dispose_recommendation_engines()
            return True
        finally:
            _close_publisher(self.publisher)
            self.publisher = None


def _combine_busy_checks(*checks: Callable[[], bool] | None) -> Callable[[], bool] | None:
    active = [check for check in checks if check is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    return lambda: any(check() for check in active)


def _throttled_busy_check(
    check: Callable[[], bool] | None,
    *,
    ttl_seconds: float,
) -> Callable[[], bool] | None:
    if check is None:
        return None
    if ttl_seconds <= 0:
        return check
    cached_until = 0.0
    cached_value = False

    def _wrapped() -> bool:
        nonlocal cached_until, cached_value
        now = time.monotonic()
        if now < cached_until:
            return cached_value
        cached_value = check()
        cached_until = now + ttl_seconds
        return cached_value

    return _wrapped


def _close_publisher(publisher: RecommendationPublisher | None) -> None:
    if publisher is None:
        return
    try:
        publisher.close()
    except Exception:
        logger.exception("Recommendation publisher close failed")


def start_events_runtime(
    settings: Settings,
    *,
    feature_config: FeatureConfig | None,
    reader: RecommendationReader,
    busy_check: Callable[[], bool] | None = None,
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
    if settings.events.ha:
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
            "Events apply lease enabled (backend=%s, key=%s)",
            settings.trigger.lock_backend,
            events_apply_lock_key(settings.trigger.lock_key),
        )

    combined_busy = _combine_busy_checks(
        busy_check,
        (retrain_probe.is_locked if retrain_probe is not None else None),
    )

    sink = build_output_sink(settings.output)
    publisher = build_publisher(settings)
    worker: EventWorker | None = None
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
            )
            online.ensure_loaded()
            logger.info(
                "Online collaborative refresh enabled (fit_partial_epochs=%d, fit_min_events=%d)",
                settings.events.online.fit_partial_epochs,
                settings.events.online.fit_min_events,
            )
        updater = IncrementalUpdater(
            sink=sink,
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=settings.top_k,
            busy_check=_throttled_busy_check(
                combined_busy,
                ttl_seconds=DEFAULT_EVENTS_RETRAIN_PROBE_TTL_SECONDS,
            ),
            write_busy_check=combined_busy,
            fence_check=(apply_lock.owned if apply_lock is not None else None),
            on_success=reader.refresh,
            online=online,
            variant_names=experiment_variant_names(settings),
            explain_enabled=settings.explain.enabled,
            publisher=publisher,
        )
        buffer = MicroBatchBuffer(
            batch_size=settings.events.incremental.batch_size,
            batch_window_seconds=settings.events.incremental.batch_window_seconds,
        )
        worker = EventWorker(
            source,
            buffer,
            updater,
            poll_interval_seconds=settings.events.incremental.poll_interval_seconds,
            apply_lock=apply_lock,
            poll_without_lock=poll_without_apply_lock(settings.events.kind, settings.events.options),
        )
        worker.start()
        logger.info(
            "Event worker started (kind=%s, batch_size=%d, window=%ss, poll=%ss, ha=%s)",
            settings.events.kind,
            settings.events.incremental.batch_size,
            settings.events.incremental.batch_window_seconds,
            settings.events.incremental.poll_interval_seconds,
            apply_lock is not None,
        )
        if apply_lock is None:
            logger.warning(
                "Incremental events assume a single writer process "
                "(kind=%s, output=%s); set events.ha = true and "
                "job.trigger.lock_backend = postgres|redis for multi-replica apply",
                settings.events.kind,
                settings.output.kind,
            )
        return EventsRuntime(
            webhook_source=webhook_source, worker=worker, apply_lock=apply_lock, publisher=publisher
        )
    except Exception:
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                logger.exception("Event worker stop after startup failure")
        _close_publisher(publisher)
        raise
