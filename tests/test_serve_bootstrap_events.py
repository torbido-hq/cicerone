from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from cicerone.config import EventsIncrementalSettings, EventsSettings, IOSettings, make_settings
from cicerone.config.constants import ALLOCATION_THOMPSON, DEFAULT_EVENTS_RETRAIN_PROBE_TTL_SECONDS
from cicerone.config.settings import ExperimentSettings, TrackSettings, VariantSettings
from cicerone.events.webhook import WebhookEventSource
from cicerone.experiment.store import ExperimentStore, experiment_state
from cicerone.feature_config import EligibilityRule, FeatureConfig
from cicerone.serve.bootstrap_events import (
    _assign_incremental_variant,
    _input_users_provider,
    start_events_runtime,
)


def test_start_events_runtime_defers_publisher_connect(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    seen: dict[str, bool] = {}

    class _Pub:
        def close(self) -> None:
            return None

    class _Reader:
        def refresh(self) -> None:
            return None

    from cicerone.serve import bootstrap_events as bootstrap

    original = bootstrap.build_publisher

    def tracking(_settings, *, connect=True):
        seen["connect"] = connect
        return _Pub()

    bootstrap.build_publisher = tracking  # type: ignore[assignment]
    try:
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
            feature_config=feature_config,
            reader=_Reader(),  # type: ignore[arg-type]
        )
        runtime.stop()
    finally:
        bootstrap.build_publisher = original  # type: ignore[assignment]
    assert seen["connect"] is False


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


def test_start_events_runtime_can_skip_background_worker(tmp_path, feature_config: FeatureConfig):
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
        feature_config=feature_config,
        reader=reader,  # type: ignore[arg-type]
        start_worker=False,
    )
    assert runtime.worker is not None
    assert runtime.worker._thread is None
    assert isinstance(runtime.webhook_source, WebhookEventSource)
    runtime.webhook_source.ingest(
        {
            "user_id": "u1",
            "item_id": "i9",
            "event_type": "purchase",
            "occurred_at": "2026-08-13T12:00:00Z",
            "event_id": "skip-start-1",
        }
    )
    assert runtime.worker.tick() == 1
    assert reader.refreshed == 1
    runtime.stop()
    assert runtime.worker._thread is None


def test_start_events_runtime_wires_input_users_provider(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    inp = tmp_path / "in"
    out.mkdir()
    inp.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    pd.DataFrame([{"user_id": "u1", "region_slug": "lazio"}]).to_parquet(inp / "users.parquet", index=False)

    class _Reader:
        def refresh(self) -> None:
            return None

    runtime = start_events_runtime(
        make_settings(
            input=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(inp)}),
            output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
            events=EventsSettings(
                enabled=True,
                kind="webhook",
                incremental=EventsIncrementalSettings(
                    batch_size=1, batch_window_seconds=60.0, poll_interval_seconds=0.05
                ),
            ),
        ),
        feature_config=replace(
            feature_config,
            eligibility=[
                EligibilityRule(
                    name="region",
                    op="eq",
                    item_column="region_slug",
                    user_column="region_slug",
                )
            ],
        ),
        reader=_Reader(),  # type: ignore[arg-type]
        start_worker=False,
    )
    try:
        assert runtime.worker is not None
        users = runtime.worker._updater._users_provider()
        assert users is not None
        assert list(users["user_id"].astype(str)) == ["u1"]
    finally:
        runtime.stop()


def test_start_events_runtime_skips_users_provider_without_user_scoped_rules(
    tmp_path, feature_config: FeatureConfig, monkeypatch
):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)

    class _Reader:
        def refresh(self) -> None:
            return None

    def _fail_build(_input):
        raise AssertionError("input users must not be read without user-scoped eligibility")

    monkeypatch.setattr("cicerone.serve.bootstrap_events.build_input_source", _fail_build)
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
        feature_config=feature_config,
        reader=_Reader(),  # type: ignore[arg-type]
        start_worker=False,
    )
    try:
        assert runtime.worker is not None
        assert runtime.worker._updater._users_provider is None
    finally:
        runtime.stop()


def test_start_events_runtime_skips_users_when_variants_are_not_user_scoped(
    tmp_path, feature_config: FeatureConfig, monkeypatch
):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)

    class _Reader:
        def refresh(self) -> None:
            return None

    def _fail_build(_input):
        raise AssertionError("input users must not be read when experiment arms are not user-scoped")

    monkeypatch.setattr("cicerone.serve.bootstrap_events.build_input_source", _fail_build)
    scoped = replace(
        feature_config,
        eligibility=[
            EligibilityRule(
                name="region",
                op="eq",
                item_column="region_slug",
                user_column="region_slug",
            )
        ],
    )
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
            experiment=ExperimentSettings(
                enabled=True,
                id="ab",
                variants=(
                    VariantSettings(name="control", traffic=0.5, eligibility=False),
                    VariantSettings(name="treatment", traffic=0.5, eligibility=False),
                ),
            ),
        ),
        feature_config=scoped,
        reader=_Reader(),  # type: ignore[arg-type]
        start_worker=False,
    )
    try:
        assert runtime.worker is not None
        assert runtime.worker._updater._users_provider is None
    finally:
        runtime.stop()


def test_input_users_provider_falls_back_to_bootstrap_snapshot(monkeypatch) -> None:
    frames = iter([pd.DataFrame([{"user_id": "u1", "region_slug": "lazio"}]), None])

    class _Src:
        def read_users(self) -> pd.DataFrame | None:
            return next(frames)

    monkeypatch.setattr("cicerone.serve.bootstrap_events.build_input_source", lambda _input: _Src())
    users = _input_users_provider(make_settings())()
    assert users is not None
    assert list(users["user_id"].astype(str)) == ["u1"]


def test_start_events_runtime_wires_variant_feature_configs(tmp_path, feature_config: FeatureConfig):
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
            experiment=ExperimentSettings(
                enabled=True,
                id="ab",
                variants=(
                    VariantSettings(name="control", traffic=0.5),
                    VariantSettings(name="treatment", traffic=0.5),
                ),
            ),
        ),
        feature_config=feature_config,
        reader=_Reader(),  # type: ignore[arg-type]
        start_worker=False,
    )
    try:
        assert runtime.worker is not None
        assert set(runtime.worker._updater._variant_feature_configs) == {"control", "treatment"}
    finally:
        runtime.stop()


def test_start_events_runtime_wires_automl_challenger_variant_feature_configs(
    tmp_path, feature_config: FeatureConfig
):
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
            experiment=ExperimentSettings(enabled=True, id="ab", automl_challenger=True),
        ),
        feature_config=feature_config,
        reader=_Reader(),  # type: ignore[arg-type]
        start_worker=False,
    )
    try:
        assert runtime.worker is not None
        configs = runtime.worker._updater._variant_feature_configs
        assert set(configs) == {"control", "treatment"}
    finally:
        runtime.stop()


def test_start_events_runtime_wires_custom_automl_variant_policy_names(
    tmp_path, feature_config: FeatureConfig
):
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
            experiment=ExperimentSettings(
                enabled=True,
                id="ab",
                automl_challenger=True,
                variants=(
                    VariantSettings(name="champion", traffic=0.5, eligibility=False),
                    VariantSettings(name="challenger", traffic=0.5),
                ),
            ),
        ),
        feature_config=feature_config,
        reader=_Reader(),  # type: ignore[arg-type]
        start_worker=False,
    )
    try:
        assert runtime.worker is not None
        configs = runtime.worker._updater._variant_feature_configs
        assert set(configs) == {"champion", "challenger"}
        assert configs["champion"].eligibility == []
        assert configs["champion"].merge_item_availability is False
    finally:
        runtime.stop()


def test_start_events_runtime_connects_source_when_worker_not_started(
    tmp_path, feature_config: FeatureConfig
):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    connected = {"n": 0}

    class _Source:
        ephemeral_event_ids = False

        def connect(self) -> None:
            connected["n"] += 1

        def poll(self, max_events: int = 100) -> list:
            if connected["n"] == 0:
                raise RuntimeError("connect() required before poll")
            return []

        def ack(self, event_ids):
            return tuple(event_ids)

        def nack(self, events):
            return ()

        def health(self):
            from cicerone.events.base import EventSourceHealth

            return EventSourceHealth(connected=connected["n"] > 0, lag=0)

        def close(self) -> None:
            return None

    class _Reader:
        def refresh(self) -> None:
            return None

    from cicerone.serve import bootstrap_events as bootstrap

    original = bootstrap.build_event_source
    bootstrap.build_event_source = lambda _kind, _options: _Source()  # type: ignore[assignment]
    runtime = None
    try:
        runtime = start_events_runtime(
            make_settings(
                output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
                events=EventsSettings(
                    enabled=True,
                    kind="db",
                    incremental=EventsIncrementalSettings(
                        batch_size=1, batch_window_seconds=60.0, poll_interval_seconds=0.05
                    ),
                ),
            ),
            feature_config=feature_config,
            reader=_Reader(),  # type: ignore[arg-type]
            start_worker=False,
        )
        assert connected["n"] == 1
        assert runtime.worker is not None
        assert runtime.worker._thread is None
        assert runtime.worker.tick() == 0
    finally:
        bootstrap.build_event_source = original
        if runtime is not None:
            runtime.stop()


def test_start_events_runtime_closes_publisher(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    closed = {"n": 0}

    class _Pub:
        def close(self) -> None:
            closed["n"] += 1
            raise RuntimeError("close failed")

    class _Reader:
        def refresh(self) -> None:
            return None

    from cicerone.serve import bootstrap_events as bootstrap

    original = bootstrap.build_publisher
    bootstrap.build_publisher = lambda _settings, **_kwargs: _Pub()  # type: ignore[assignment]
    try:
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
            feature_config=feature_config,
            reader=_Reader(),  # type: ignore[arg-type]
        )
        runtime.stop()
    finally:
        bootstrap.build_publisher = original  # type: ignore[assignment]
    assert closed["n"] == 1


def test_stop_closes_publisher_when_worker_hangs(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    closed = {"n": 0}

    class _Pub:
        def close(self) -> None:
            closed["n"] += 1

    class _Reader:
        def refresh(self) -> None:
            return None

    from cicerone.serve import bootstrap_events as bootstrap

    original = bootstrap.build_publisher
    bootstrap.build_publisher = lambda _settings, **_kwargs: _Pub()  # type: ignore[assignment]
    try:
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
            feature_config=feature_config,
            reader=_Reader(),  # type: ignore[arg-type]
        )
        assert runtime.worker is not None
        real_stop = runtime.worker.stop
        runtime.worker.stop = lambda **_kwargs: False  # type: ignore[method-assign]
        try:
            assert runtime.stop() is False
        finally:
            runtime.worker.stop = real_stop  # type: ignore[method-assign]
            runtime.worker.stop()
        assert closed["n"] == 1
        assert runtime.publisher is None
    finally:
        bootstrap.build_publisher = original  # type: ignore[assignment]


def test_start_events_runtime_closes_publisher_on_startup_error(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    closed = {"n": 0}

    class _Pub:
        def close(self) -> None:
            closed["n"] += 1

    class _Reader:
        def refresh(self) -> None:
            return None

    from cicerone.events.worker import EventWorker
    from cicerone.serve import bootstrap_events as bootstrap

    original_pub = bootstrap.build_publisher
    original_start = EventWorker.start

    def _boom(self) -> None:
        raise RuntimeError("start fail")

    bootstrap.build_publisher = lambda _settings, **_kwargs: _Pub()  # type: ignore[assignment]
    EventWorker.start = _boom  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="start fail"):
            start_events_runtime(
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
                reader=_Reader(),  # type: ignore[arg-type]
            )
    finally:
        bootstrap.build_publisher = original_pub  # type: ignore[assignment]
        EventWorker.start = original_start  # type: ignore[method-assign]
    assert closed["n"] == 1


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


def _experiment_settings(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    return make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        events=EventsSettings(
            enabled=True,
            kind="webhook",
            incremental=EventsIncrementalSettings(
                batch_size=1, batch_window_seconds=60.0, poll_interval_seconds=0.05
            ),
        ),
        experiment=ExperimentSettings(
            enabled=True,
            id="exp-1",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
    )


def test_assign_incremental_variant_caches_promote_read_and_follows_live_winner(tmp_path, monkeypatch):
    settings = _experiment_settings(tmp_path)
    store = ExperimentStore(settings.output)
    store.write_state(experiment_state("exp-1", promoted_variant="treatment"))
    reads = {"n": 0}
    original = ExperimentStore.assignment_overlay

    def counting(self, experiment_id: str):
        reads["n"] += 1
        return original(self, experiment_id)

    monkeypatch.setattr(ExperimentStore, "assignment_overlay", counting)
    now = {"t": 0.0}
    assigned = _assign_incremental_variant(settings, clock=lambda: now["t"])
    assert assigned is not None
    assert assigned("u1") == "treatment"
    assert assigned("u2") == "treatment"
    assert reads["n"] == 1
    store.write_state(experiment_state("exp-1", promoted_variant="control"))
    assert assigned("u1") == "treatment"
    now["t"] += DEFAULT_EVENTS_RETRAIN_PROBE_TTL_SECONDS
    assert assigned("u1") == "control"
    assert assigned("u2") == "control"
    assert reads["n"] == 2


def test_assign_incremental_variant_hashes_config_names_without_pair(tmp_path):
    settings = _experiment_settings(tmp_path)
    settings = make_settings(
        output=settings.output,
        events=settings.events,
        experiment=ExperimentSettings(
            enabled=True,
            id="exp-1",
            allocation=ALLOCATION_THOMPSON,
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        ),
        track=TrackSettings(enabled=True),
    )
    assigned = _assign_incremental_variant(settings)
    assert assigned is not None
    seen = {assigned(f"u{i}") for i in range(40)}
    assert seen == {"control", "treatment"}


def test_start_events_runtime_assign_variant_caches_winner_within_ttl(tmp_path, feature_config):
    settings = _experiment_settings(tmp_path)
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(Path(settings.output.options["path"]) / "recommendations.parquet", index=False)
    ExperimentStore(settings.output).write_state(experiment_state("exp-1", promoted_variant="treatment"))

    class _Reader:
        def refresh(self) -> None:
            return None

    runtime = start_events_runtime(
        settings,
        feature_config=feature_config,
        reader=_Reader(),  # type: ignore[arg-type]
    )
    assert runtime.worker is not None
    assigned = runtime.worker._updater._assign_variant
    assert assigned is not None
    assert assigned("u1") == "treatment"
    ExperimentStore(settings.output).write_state(experiment_state("exp-1", promoted_variant="control"))
    assert assigned("u1") == "treatment"
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


def test_throttled_busy_check():
    from cicerone.serve.bootstrap_events import _throttled_busy_check

    assert _throttled_busy_check(None, ttl_seconds=1.0) is None
    calls = {"n": 0}

    def probe() -> bool:
        calls["n"] += 1
        return False

    live = _throttled_busy_check(probe, ttl_seconds=0)
    assert live is not None
    assert live() is False
    assert live() is False
    assert calls["n"] == 2

    calls["n"] = 0
    cached = _throttled_busy_check(probe, ttl_seconds=60.0)
    assert cached is not None
    assert cached() is False
    assert cached() is False
    assert calls["n"] == 1
