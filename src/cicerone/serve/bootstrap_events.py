"""Start the serve-process event worker (poll → micro-batch → write-through)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from cicerone.config import Settings
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.registry import build_event_source
from cicerone.events.updater import IncrementalUpdater
from cicerone.events.webhook import WebhookEventSource
from cicerone.events.worker import EventWorker
from cicerone.feature_config import FeatureConfig
from cicerone.io.base import RecommendationReader
from cicerone.io.factory import build_output_sink

logger = logging.getLogger(__name__)


@dataclass
class EventsRuntime:
    webhook_source: WebhookEventSource | None
    worker: EventWorker | None

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()


def start_events_runtime(
    settings: Settings,
    *,
    feature_config: FeatureConfig,
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

    sink = build_output_sink(settings.output)
    updater = IncrementalUpdater(
        sink=sink,
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=settings.top_k,
        busy_check=busy_check,
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
    )
    worker.start()
    logger.info(
        "Event worker started (kind=%s, batch_size=%d, window=%ss, poll=%ss)",
        settings.events.kind,
        settings.events.incremental.batch_size,
        settings.events.incremental.batch_window_seconds,
        settings.events.incremental.poll_interval_seconds,
    )
    return EventsRuntime(webhook_source=webhook_source, worker=worker)
