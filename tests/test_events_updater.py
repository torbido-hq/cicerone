from __future__ import annotations

import pandas as pd
from support.events import event_payload

from cicerone.blending import COLD_START_USER_ID
from cicerone.config import IOSettings, make_settings
from cicerone.events.normalize import normalize_event
from cicerone.events.store import load_recommendations_frame
from cicerone.events.updater import INCREMENTAL_SOURCE, IncrementalUpdater
from cicerone.feature_config import FeatureConfig
from cicerone.io.factory import build_output_sink
from cicerone.io.recommendation_reader import RECOMMENDATION_COLUMNS


def test_incremental_updater_write_through(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    existing = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"},
            {"user_id": "u2", "item_id": "x", "rank": 1, "score": 0.5, "source": "personalized"},
            {
                "user_id": COLD_START_USER_ID,
                "item_id": "cold-keep",
                "rank": 1,
                "score": 0.2,
                "source": "popular_fallback",
            },
        ]
    )
    existing.to_parquet(out / "recommendations.parquet", index=False)

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    sink = build_output_sink(settings.output)
    called = {"n": 0}

    def on_success() -> None:
        called["n"] += 1

    updater = IncrementalUpdater(
        sink=sink,
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        on_success=on_success,
    )
    events = [
        normalize_event(event_payload(user_id="u1", item_id="i9", event_id="n1")),
        normalize_event(event_payload(user_id="u1", item_id="i8", event_type="view", event_id="n2")),
    ]
    assert updater.apply(events) == 2
    frame = load_recommendations_frame(settings.output)
    u1 = frame[frame["user_id"] == "u1"].sort_values("rank")
    assert "i9" in set(u1["item_id"].astype(str))
    assert INCREMENTAL_SOURCE in set(u1["source"].astype(str))
    assert "old" in set(u1["item_id"].astype(str))
    assert list(frame[frame["user_id"] == "u2"]["item_id"]) == ["x"]
    cold = frame[frame["user_id"] == COLD_START_USER_ID]
    assert "cold-keep" in set(cold["item_id"].astype(str))
    assert called["n"] == 1
    assert updater.events_applied == 2
    assert updater.last_success_at is not None


def test_incremental_updater_reserves_boost_slots(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    rows = [
        {"user_id": "u1", "item_id": f"p{i}", "rank": i, "score": float(10 - i), "source": "blended"}
        for i in range(1, 6)
    ]
    pd.DataFrame(rows).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
    )
    assert (
        updater.apply([normalize_event(event_payload(user_id="u1", item_id="boosted", event_id="b1"))]) == 1
    )
    u1 = load_recommendations_frame(settings.output)
    u1 = u1[u1["user_id"] == "u1"]
    assert "boosted" in set(u1["item_id"].astype(str))
    assert len(u1) == 5


def test_incremental_updater_preserves_compound_sources(tmp_path, feature_config: FeatureConfig):
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
            }
        ]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
    )
    updater.apply([normalize_event(event_payload(user_id="u1", item_id="new", event_id="c1"))])
    u1 = load_recommendations_frame(settings.output)
    assert "compound" in set(u1[u1["user_id"] == "u1"]["item_id"].astype(str))


def test_incremental_updater_skips_when_busy(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(columns=list(RECOMMENDATION_COLUMNS)).to_parquet(
        out / "recommendations.parquet", index=False
    )
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        busy_check=lambda: True,
    )
    assert updater.apply([normalize_event(event_payload())]) == 0


def test_incremental_updater_empty_and_unknown_event_type(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        on_success=lambda: None,
    )
    assert updater.apply([]) == 0
    applied = updater.apply(
        [normalize_event(event_payload(event_type="unknown_type", event_id="u", item_id="ix"))]
    )
    assert applied == 1
    frame = load_recommendations_frame(settings.output)
    # Unknown types do not boost / popular-score; cold-start stays empty.
    assert frame.empty or "ix" not in set(frame["item_id"].astype(str))


def test_incremental_updater_no_feature_config(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=2,
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=None,
        top_k=2,
    )
    assert updater.apply([normalize_event(event_payload(event_id="nfc"))]) == 1
    frame = load_recommendations_frame(settings.output)
    assert "i1" in set(frame["item_id"].astype(str))


def test_incremental_updater_caches_frame_across_applies(tmp_path, feature_config, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    loads = {"n": 0}
    real_load = load_recommendations_frame

    def counting_load(output):  # type: ignore[no-untyped-def]
        loads["n"] += 1
        return real_load(output)

    monkeypatch.setattr("cicerone.events.updater.load_recommendations_frame", counting_load)
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
    )
    assert updater.apply([normalize_event(event_payload(event_id="c1", item_id="a"))]) == 1
    assert loads["n"] == 1
    assert updater.apply([normalize_event(event_payload(event_id="c2", item_id="b"))]) == 1
    assert loads["n"] == 1


def test_incremental_updater_busy_invalidates_cache(tmp_path, feature_config, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    loads = {"n": 0}
    real_load = load_recommendations_frame

    def counting_load(output):  # type: ignore[no-untyped-def]
        loads["n"] += 1
        return real_load(output)

    monkeypatch.setattr("cicerone.events.updater.load_recommendations_frame", counting_load)
    busy = {"v": False}
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        busy_check=lambda: busy["v"],
    )
    assert updater.apply([normalize_event(event_payload(event_id="b1", item_id="a"))]) == 1
    assert loads["n"] == 1
    busy["v"] = True
    assert updater.apply([normalize_event(event_payload(event_id="b2", item_id="b"))]) == 0
    busy["v"] = False
    assert updater.apply([normalize_event(event_payload(event_id="b3", item_id="c"))]) == 1
    assert loads["n"] == 2
