from __future__ import annotations

import pandas as pd
import pytest
from support.events import event_payload

from cicerone.artifact import build_artifact, dumps_artifact, loads_artifact
from cicerone.config import (
    EventsIncrementalSettings,
    EventsOnlineSettings,
    EventsSettings,
    IOSettings,
    make_settings,
)
from cicerone.dataset import build_dataset
from cicerone.events.normalize import normalize_event
from cicerone.events.online import OnlineArtifactError, OnlineTrainer
from cicerone.events.online_result import OnlineRefreshResult
from cicerone.events.store import load_recommendations_frame
from cicerone.events.updater import IncrementalUpdater
from cicerone.feature_config import FeatureConfig
from cicerone.io.factory import build_output_sink
from cicerone.model import train_and_recommend
from cicerone.serve.bootstrap_events import start_events_runtime


class _BoomSequential:
    def fit(self, dataset):
        return self

    def recommend(self, *, users, dataset, k, filter_viewed, items_to_recommend=None):
        raise AssertionError("sequential should be skipped")


def _spy_fit_partial(monkeypatch, model) -> dict[str, int]:
    calls = {"n": 0}
    original = type(model).fit_partial

    def _counted(self, dataset, epochs):
        calls["n"] += 1
        return original(self, dataset, epochs)

    monkeypatch.setattr(type(model), "fit_partial", _counted)
    return calls


def _write_artifact(
    tmp_path,
    feature_config,
    sample_events,
    sample_users,
    sample_items,
    *,
    models: list[str] | None = None,
    extra_fitted: dict | None = None,
) -> tuple[object, list[str]]:
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    enabled = models if models is not None else ["collaborative", "item_based", "popular"]
    built = build_dataset(sample_events, sample_users, sample_items, feature_config, half_life_days=90)
    fitted: dict = {}
    train_and_recommend(
        built,
        ["u1", "u2", "u3"],
        feature_config,
        top_k=3,
        enabled_models=[name for name in enabled if name != "sequential"],
        strategy_cache=fitted,
    )
    if extra_fitted:
        fitted.update(extra_fitted)
    artifact = build_artifact(
        fitted=fitted,
        built=built,
        feature_config=feature_config,
        models=enabled,
        model_weights=None,
        rrf_k=None,
    )
    sink = build_output_sink(
        IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)})
    )
    sink.write_model_artifact(dumps_artifact(artifact))
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"},
            {"user_id": "u2", "item_id": "keep", "rank": 1, "score": 0.5, "source": "personalized"},
        ]
    ).to_parquet(out / "recommendations.parquet", index=False)
    return sink, enabled


def _trainer(sink, *, epochs: int = 1, min_events: int = 1) -> OnlineTrainer:
    trainer = OnlineTrainer(
        sink=sink,
        top_k=3,
        half_life_days=90,
        fit_partial_epochs=epochs,
        fit_min_events=min_events,
    )
    trainer.ensure_loaded()
    return trainer


def _known_event(event_id: str):
    return normalize_event(event_payload(user_id="u2", item_id="i1", event_id=event_id))


def test_online_trainer_rejects_bad_knobs(tmp_path):
    sink = build_output_sink(
        IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    with pytest.raises(ValueError, match="top_k"):
        OnlineTrainer(sink=sink, top_k=0, half_life_days=90, fit_partial_epochs=1, fit_min_events=1)
    with pytest.raises(ValueError, match="fit_partial_epochs"):
        OnlineTrainer(sink=sink, top_k=3, half_life_days=90, fit_partial_epochs=-1, fit_min_events=1)
    with pytest.raises(ValueError, match="fit_min_events"):
        OnlineTrainer(sink=sink, top_k=3, half_life_days=90, fit_partial_epochs=1, fit_min_events=0)
    sink = build_output_sink(
        IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    trainer = OnlineTrainer(
        sink=sink,
        top_k=3,
        half_life_days=90,
        fit_partial_epochs=1,
        fit_min_events=1,
    )
    with pytest.raises(OnlineArtifactError, match="save_model_artifact"):
        trainer.ensure_loaded()


def test_online_trainer_known_ids_fit_partial(
    tmp_path, feature_config, sample_events, sample_users, sample_items, monkeypatch
):
    sink, _enabled = _write_artifact(tmp_path, feature_config, sample_events, sample_users, sample_items)
    trainer = _trainer(sink)
    calls = _spy_fit_partial(monkeypatch, trainer._artifact.fitted["collaborative"])
    result = trainer.refresh([_known_event("k1")])
    assert result.events_known == 1
    assert result.events_dropped_unknown == 0
    assert result.users_refreshed >= 1
    assert result.fit_partial_epochs == 1
    assert calls["n"] == 1
    assert "personalized" in set(result.rows["source"].astype(str))
    assert "u2" in set(result.rows["user_id"].astype(str))
    loaded = loads_artifact(sink.read_model_artifact())
    assert loaded.dataset.get_raw_interactions() is not None


def test_online_trainer_unknown_ids_dropped(
    tmp_path, feature_config, sample_events, sample_users, sample_items, monkeypatch
):
    sink, _enabled = _write_artifact(tmp_path, feature_config, sample_events, sample_users, sample_items)
    trainer = _trainer(sink)
    calls = _spy_fit_partial(monkeypatch, trainer._artifact.fitted["collaborative"])
    result = trainer.refresh(
        [
            normalize_event(event_payload(user_id="u9", item_id="i1", event_id="unk-user")),
            normalize_event(event_payload(user_id="u1", item_id="i9", event_id="unk-item")),
        ]
    )
    assert result.events_known == 0
    assert result.events_dropped_unknown == 2
    assert result.users_refreshed == 0
    assert result.rows.empty
    assert calls["n"] == 0


def test_online_trainer_epochs_zero_skips_sgd(
    tmp_path, feature_config, sample_events, sample_users, sample_items, monkeypatch
):
    sink, _enabled = _write_artifact(tmp_path, feature_config, sample_events, sample_users, sample_items)
    trainer = _trainer(sink, epochs=0)
    calls = _spy_fit_partial(monkeypatch, trainer._artifact.fitted["collaborative"])
    result = trainer.refresh([_known_event("e0")])
    assert result.events_known == 1
    assert result.fit_partial_epochs == 0
    assert calls["n"] == 0
    assert result.users_refreshed >= 1


def test_online_trainer_fit_min_events_holds_sgd(
    tmp_path, feature_config, sample_events, sample_users, sample_items, monkeypatch
):
    sink, _enabled = _write_artifact(tmp_path, feature_config, sample_events, sample_users, sample_items)
    trainer = _trainer(sink, min_events=10)
    calls = _spy_fit_partial(monkeypatch, trainer._artifact.fitted["collaborative"])
    result = trainer.refresh([_known_event("hold")])
    assert result.events_known == 1
    assert result.fit_partial_epochs == 0
    assert calls["n"] == 0


def test_online_trainer_skips_sequential_without_torch(
    tmp_path, feature_config, sample_events, sample_users, sample_items, monkeypatch
):
    sink, _enabled = _write_artifact(
        tmp_path,
        feature_config,
        sample_events,
        sample_users,
        sample_items,
        models=["collaborative", "sequential", "popular"],
        extra_fitted={"sequential": _BoomSequential()},
    )
    monkeypatch.setattr("cicerone.events.online.sequential_extra_available", lambda: False)
    trainer = _trainer(sink)
    result = trainer.refresh([_known_event("seq")])
    assert result.users_refreshed >= 1
    assert "sequential" not in set(result.rows["source"].astype(str))


def test_incremental_updater_splices_online_rows(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"},
            {"user_id": "u1", "item_id": "seq-keep", "rank": 2, "score": 0.9, "source": "sequential"},
            {"user_id": "u2", "item_id": "keep", "rank": 1, "score": 0.5, "source": "personalized"},
        ]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    sink = build_output_sink(settings.output)
    online_rows = pd.DataFrame(
        [{"user_id": "u1", "item_id": "fresh", "rank": 1, "score": 2.0, "source": "personalized"}]
    )

    class _FakeOnline:
        def refresh(self, events):
            del events
            return OnlineRefreshResult(rows=online_rows, users_refreshed=1, fit_partial_epochs=1)

        def invalidate(self) -> None:
            return None

    updater = IncrementalUpdater(
        sink=sink,
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        online=_FakeOnline(),
    )
    events = [normalize_event(event_payload(user_id="u1", item_id="i9", event_id="splice"))]
    assert updater.apply(events) == 1
    frame = load_recommendations_frame(settings.output)
    u1 = set(frame[frame["user_id"] == "u1"]["item_id"].astype(str))
    assert "fresh" in u1
    assert "old" not in u1
    assert "seq-keep" in u1
    assert "i9" in u1
    assert list(frame[frame["user_id"] == "u2"]["item_id"]) == ["keep"]
    manifest = (out / "manifest.json").read_text()
    assert "online_users_refreshed" in manifest


def test_incremental_updater_splices_overlapping_compound_sources(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "compound",
                "rank": 1,
                "score": 1.0,
                "source": "personalized+popular_fallback",
            },
            {"user_id": "u1", "item_id": "seq-keep", "rank": 2, "score": 0.9, "source": "sequential"},
        ]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    online_rows = pd.DataFrame(
        [{"user_id": "u1", "item_id": "fresh", "rank": 1, "score": 2.0, "source": "personalized"}]
    )

    class _FakeOnline:
        def refresh(self, events):
            del events
            return OnlineRefreshResult(rows=online_rows, users_refreshed=1, fit_partial_epochs=1)

        def invalidate(self) -> None:
            return None

    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        online=_FakeOnline(),
    )
    events = [normalize_event(event_payload(user_id="u1", item_id="i9", event_id="compound-splice"))]
    assert updater.apply(events) == 1
    frame = load_recommendations_frame(settings.output)
    u1 = set(frame[frame["user_id"] == "u1"]["item_id"].astype(str))
    assert "fresh" in u1
    assert "compound" not in u1
    assert "seq-keep" in u1


def test_incremental_updater_splices_online_rows_across_variants(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "old-control",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                "variant": "control",
            },
            {
                "user_id": "u1",
                "item_id": "old-treatment",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                "variant": "treatment",
            },
        ]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    online_rows = pd.DataFrame(
        [{"user_id": "u1", "item_id": "fresh", "rank": 1, "score": 2.0, "source": "personalized"}]
    )

    class _FakeOnline:
        def refresh(self, events):
            del events
            return OnlineRefreshResult(rows=online_rows, users_refreshed=1, fit_partial_epochs=1)

        def invalidate(self) -> None:
            return None

    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        online=_FakeOnline(),
        variant_names=("control", "treatment"),
    )
    events = [normalize_event(event_payload(user_id="u1", item_id="i9", event_id="variant-splice"))]
    assert updater.apply(events) == 1
    frame = load_recommendations_frame(settings.output)
    u1 = frame[frame["user_id"] == "u1"]
    for variant in ("control", "treatment"):
        rows = u1[u1["variant"] == variant]
        items = set(rows["item_id"].astype(str))
        assert "fresh" in items
        assert f"old-{variant}" not in items
        assert "i9" in items


def test_incremental_updater_busy_invalidates_online(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(columns=["user_id", "item_id", "rank", "score", "source"]).to_parquet(
        out / "recommendations.parquet", index=False
    )
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
    )
    calls = {"invalidate": 0, "refresh": 0}

    class _FakeOnline:
        def refresh(self, events):
            calls["refresh"] += 1
            return OnlineRefreshResult(rows=pd.DataFrame())

        def invalidate(self) -> None:
            calls["invalidate"] += 1

    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        busy_check=lambda: True,
        online=_FakeOnline(),
    )
    assert updater.apply([normalize_event(event_payload())]) == 0
    assert calls["invalidate"] == 1
    assert calls["refresh"] == 0


def test_start_events_runtime_online_requires_artifact(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)

    class _Reader:
        def refresh(self) -> None:
            return None

    with pytest.raises(OnlineArtifactError, match="save_model_artifact"):
        start_events_runtime(
            make_settings(
                output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
                events=EventsSettings(
                    enabled=True,
                    kind="webhook",
                    incremental=EventsIncrementalSettings(
                        batch_size=1, batch_window_seconds=60.0, poll_interval_seconds=0.05
                    ),
                    online=EventsOnlineSettings(enabled=True, fit_min_events=1),
                ),
            ),
            feature_config=feature_config,
            reader=_Reader(),  # type: ignore[arg-type]
        )


def test_start_events_runtime_online_loads_artifact(
    tmp_path, feature_config, sample_events, sample_users, sample_items
):
    sink, _enabled = _write_artifact(tmp_path, feature_config, sample_events, sample_users, sample_items)
    del sink

    class _Reader:
        def refresh(self) -> None:
            return None

    runtime = start_events_runtime(
        make_settings(
            output=IOSettings(
                kind="dataset", options={"storage_backend": "local", "path": str(tmp_path / "out")}
            ),
            events=EventsSettings(
                enabled=True,
                kind="webhook",
                incremental=EventsIncrementalSettings(
                    batch_size=1, batch_window_seconds=60.0, poll_interval_seconds=0.05
                ),
                online=EventsOnlineSettings(enabled=True, fit_min_events=1),
            ),
        ),
        feature_config=feature_config,
        reader=_Reader(),  # type: ignore[arg-type]
    )
    assert runtime.worker is not None
    runtime.stop()


def test_online_trainer_lock_lost(tmp_path, feature_config, sample_events, sample_users, sample_items):
    from cicerone.locks import LockLostError

    sink, _enabled = _write_artifact(tmp_path, feature_config, sample_events, sample_users, sample_items)
    trainer = OnlineTrainer(
        sink=sink,
        top_k=3,
        half_life_days=90,
        fit_partial_epochs=1,
        fit_min_events=1,
        fence_check=lambda: False,
    )
    trainer.ensure_loaded()
    with pytest.raises(LockLostError, match="online write"):
        trainer.refresh([_known_event("lock")])


def test_online_trainer_empty_events_skips_load(tmp_path):
    sink = build_output_sink(
        IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )
    trainer = OnlineTrainer(
        sink=sink,
        top_k=3,
        half_life_days=90,
        fit_partial_epochs=1,
        fit_min_events=1,
    )
    result = trainer.refresh([])
    assert result.rows.empty
    assert result.users_refreshed == 0


def test_online_trainer_accumulates_fit_min_events(
    tmp_path, feature_config, sample_events, sample_users, sample_items, monkeypatch
):
    sink, _enabled = _write_artifact(tmp_path, feature_config, sample_events, sample_users, sample_items)
    trainer = _trainer(sink, min_events=2)
    calls = _spy_fit_partial(monkeypatch, trainer._artifact.fitted["collaborative"])
    first = trainer.refresh([_known_event("acc-1")])
    assert first.fit_partial_epochs == 0
    assert calls["n"] == 0
    second = trainer.refresh([_known_event("acc-2")])
    assert second.fit_partial_epochs == 1
    assert calls["n"] == 1


def test_online_trainer_skips_download_when_fingerprint_unchanged(
    tmp_path, feature_config, sample_events, sample_users, sample_items
):
    sink, _enabled = _write_artifact(tmp_path, feature_config, sample_events, sample_users, sample_items)
    trainer = _trainer(sink)
    reads = {"n": 0}
    original = sink.read_model_artifact

    def _counted() -> bytes | None:
        reads["n"] += 1
        return original()

    sink.read_model_artifact = _counted  # type: ignore[method-assign]
    assert trainer._reload() is True
    assert reads["n"] == 0
    trainer._artifact_token = "stale"
    assert trainer._reload() is True
    assert reads["n"] == 1


def test_incremental_updater_online_trainer_replaces_personalized(
    tmp_path, feature_config, sample_events, sample_users, sample_items, caplog
):
    sink, _enabled = _write_artifact(tmp_path, feature_config, sample_events, sample_users, sample_items)
    settings = make_settings(
        output=IOSettings(
            kind="dataset", options={"storage_backend": "local", "path": str(tmp_path / "out")}
        ),
        top_k=5,
    )
    updater = IncrementalUpdater(
        sink=sink,
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        online=_trainer(sink, min_events=1),
    )
    with caplog.at_level("ERROR"):
        assert updater.apply([_known_event("e2e")]) == 1
    assert "Online collaborative refresh failed" not in caplog.text
    frame = load_recommendations_frame(settings.output)
    u2 = frame[frame["user_id"].astype(str) == "u2"]
    assert "keep" not in set(u2["item_id"].astype(str))
    assert "personalized" in set(u2["source"].astype(str))
    manifest = (tmp_path / "out" / "manifest.json").read_text()
    assert "online_users_refreshed" in manifest
    assert "online_events_dropped_unknown" in manifest


def test_incremental_updater_unknown_ids_keep_personalized(
    tmp_path, feature_config, sample_events, sample_users, sample_items
):
    sink, _enabled = _write_artifact(tmp_path, feature_config, sample_events, sample_users, sample_items)
    settings = make_settings(
        output=IOSettings(
            kind="dataset", options={"storage_backend": "local", "path": str(tmp_path / "out")}
        ),
        top_k=5,
    )
    updater = IncrementalUpdater(
        sink=sink,
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        online=_trainer(sink, min_events=1),
    )
    assert updater.apply([normalize_event(event_payload(user_id="u1", item_id="i9", event_id="unk"))]) == 1
    frame = load_recommendations_frame(settings.output)
    u1 = set(frame[frame["user_id"].astype(str) == "u1"]["item_id"].astype(str))
    assert "old" in u1
    assert "i9" in u1


def test_incremental_updater_preserves_sequential_without_torch(
    tmp_path, feature_config, sample_events, sample_users, sample_items, monkeypatch
):
    out = tmp_path / "out"
    sink, _enabled = _write_artifact(
        tmp_path,
        feature_config,
        sample_events,
        sample_users,
        sample_items,
        models=["collaborative", "sequential", "popular"],
        extra_fitted={"sequential": _BoomSequential()},
    )
    pd.DataFrame(
        [
            {"user_id": "u2", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"},
            {"user_id": "u2", "item_id": "seq-keep", "rank": 2, "score": 0.9, "source": "sequential"},
        ]
    ).to_parquet(out / "recommendations.parquet", index=False)
    monkeypatch.setattr("cicerone.events.online.sequential_extra_available", lambda: False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    updater = IncrementalUpdater(
        sink=sink,
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        online=_trainer(sink, min_events=1),
    )
    assert updater.apply([_known_event("seq-e2e")]) == 1
    frame = load_recommendations_frame(settings.output)
    u2 = frame[frame["user_id"].astype(str) == "u2"]
    assert "seq-keep" in set(u2["item_id"].astype(str))
    assert "old" not in set(u2["item_id"].astype(str))
    assert "personalized" in set(u2["source"].astype(str))


def test_incremental_updater_online_error_keeps_preserved(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )

    class _BoomOnline:
        def refresh(self, events):
            del events
            raise RuntimeError("online boom")

        def invalidate(self) -> None:
            return None

    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        online=_BoomOnline(),
    )
    assert updater.apply([normalize_event(event_payload(user_id="u1", item_id="i1", event_id="boom"))]) == 1
    frame = load_recommendations_frame(settings.output)
    assert "old" in set(frame[frame["user_id"] == "u1"]["item_id"].astype(str))
