"""Start the serve-process event worker (poll → micro-batch → write-through)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from cicerone.config import Settings
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.ha import poll_without_apply_lock
from cicerone.events.registry import build_event_source
from cicerone.events.store import dispose_recommendation_engines
from cicerone.events.updater import IncrementalUpdater
from cicerone.events.webhook import WebhookEventSource
from cicerone.events.worker import EventWorker
from cicerone.feature_config import FeatureConfig
from cicerone.io.base import RecommendationReader
from cicerone.io.factory import build_output_sink
from cicerone.locks import LockBackend, build_lock_backend, events_apply_lock_key
from cicerone.serve.metrics import update_events_leader

logger = logging.getLogger(__name__)


@dataclass
class EventsRuntime:
    webhook_source: WebhookEventSource | None
    worker: EventWorker | None
    apply_lock: LockBackend | None = None

    def stop(self) -> bool:
        stopped = True
        if self.worker is not None:
            stopped = self.worker.stop()
            if not stopped:
                logger.warning("Event worker did not stop in time; skipping engine dispose")
                return False
        dispose_recommendation_engines()
        return True


def _combine_busy_checks(*checks: Callable[[], bool] | None) -> Callable[[], bool] | None:
    active = [check for check in checks if check is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    return lambda: any(check() for check in active)


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
    if settings.trigger.lock_backend != "in_process":
        apply_lock = build_lock_backend(
            settings,
            lock_key=events_apply_lock_key(settings.trigger.lock_key),
        )
        retrain_probe = build_lock_backend(settings)
        logger.info(
            "Events apply lease enabled (backend=%s, key=%s)",
            settings.trigger.lock_backend,
            events_apply_lock_key(settings.trigger.lock_key),
        )
    else:
        update_events_leader(True)

    combined_busy = _combine_busy_checks(
        busy_check,
        (retrain_probe.is_locked if retrain_probe is not None else None),
    )

    sink = build_output_sink(settings.output)
    updater = IncrementalUpdater(
        sink=sink,
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=settings.top_k,
        busy_check=combined_busy,
        fence_check=(apply_lock.owned if apply_lock is not None else None),
        on_success=reader.refresh,
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
    return EventsRuntime(webhook_source=webhook_source, worker=worker, apply_lock=apply_lock)
