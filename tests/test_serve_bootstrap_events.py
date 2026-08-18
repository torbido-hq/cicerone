from __future__ import annotations

import pandas as pd

from cicerone.config import EventsIncrementalSettings, EventsSettings, IOSettings, make_settings
from cicerone.events.webhook import WebhookEventSource
from cicerone.feature_config import FeatureConfig
from cicerone.serve.bootstrap_events import start_events_runtime


def test_start_events_runtime_disabled_and_webhook(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)

    class _Reader:
        def __init__(self) -> None:
            self.refreshed = 0

        def refresh(self) -> None:
            self.refreshed += 1

    reader = _Reader()
    disabled = start_events_runtime(
        make_settings(
            output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)})
        ),
        feature_config=feature_config,
        reader=reader,  # type: ignore[arg-type]
    )
    assert disabled.webhook_source is None
    assert disabled.worker is None
    disabled.stop()

    enabled = start_events_runtime(
        make_settings(
            output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
            events=EventsSettings(
                enabled=True,
                kind="webhook",
                incremental=EventsIncrementalSettings(
                    batch_size=1, batch_window_seconds=60.0, poll_interval_seconds=0.05
                ),
            ),
        ),
        feature_config=feature_config,
        reader=reader,  # type: ignore[arg-type]
    )
    assert isinstance(enabled.webhook_source, WebhookEventSource)
    assert enabled.worker is not None
    assert enabled.worker._buffer._batch_size == 1
    assert enabled.worker._poll_interval_seconds == 0.05
    enabled.webhook_source.ingest(
        {
            "user_id": "u1",
            "item_id": "i9",
            "event_type": "purchase",
            "occurred_at": "2026-08-13T12:00:00Z",
            "event_id": "rt-1",
        }
    )
    assert enabled.worker.tick() == 1
    assert reader.refreshed == 1
    enabled.stop()
    assert enabled.worker._thread is None or not enabled.worker._thread.is_alive()


def test_start_events_runtime_without_feature_config(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)

    class _Reader:
        def refresh(self) -> None:
            return None

    runtime = start_events_runtime(
        make_settings(
            output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
            events=EventsSettings(
                enabled=True,
                kind="webhook",
                incremental=EventsIncrementalSettings(
                    batch_size=1, batch_window_seconds=60.0, poll_interval_seconds=0.05
                ),
            ),
        ),
        feature_config=None,
        reader=_Reader(),  # type: ignore[arg-type]
    )
    assert runtime.worker is not None
    runtime.stop()


def test_combine_busy_checks():
    from cicerone.serve.bootstrap_events import _combine_busy_checks

    assert _combine_busy_checks(None, None) is None
    single = _combine_busy_checks(lambda: True)
    assert single is not None
    assert single() is True
    both = _combine_busy_checks(lambda: False, lambda: True)
    assert both is not None
    assert both() is True
