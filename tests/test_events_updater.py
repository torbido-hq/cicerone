from __future__ import annotations

import pandas as pd
import pytest
from support.events import event_payload

from cicerone.blending import COLD_START_USER_ID
from cicerone.config import IOSettings, make_settings
from cicerone.events.normalize import normalize_event
from cicerone.events.store import load_recommendations_for_users, load_recommendations_frame
from cicerone.events.updater import INCREMENTAL_SOURCE, IncrementalUpdater
from cicerone.feature_config import FeatureConfig
from cicerone.io.factory import build_output_sink
from cicerone.io.recommendation_reader import RECOMMENDATION_COLUMNS
from cicerone.reasons import dump_source_reasons, parse_reasons


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


def test_incremental_updater_preserves_reasons_on_personalized_rows(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    kept = dump_source_reasons("personalized", rank=1)
    pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "old",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                "reasons": kept,
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
    updater.apply([normalize_event(event_payload(user_id="u1", item_id="new", event_id="r1"))])
    frame = load_recommendations_frame(settings.output)
    u1 = frame[frame["user_id"] == "u1"]
    old = u1[u1["item_id"] == "old"].iloc[0]
    assert parse_reasons(old["reasons"]).sources[0].label == "personalized"
    fresh = u1[u1["item_id"] == "new"].iloc[0]
    assert parse_reasons(fresh["reasons"]).sources[0].label == INCREMENTAL_SOURCE


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


def test_incremental_updater_rechecks_busy_before_write(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
    )
    checks = {"n": 0}
    writes = {"n": 0}

    def busy() -> bool:
        checks["n"] += 1
        return checks["n"] >= 2

    sink = build_output_sink(settings.output)
    real_replace = sink.replace_recommendations_for_users

    def counting_replace(df, *, user_ids):  # type: ignore[no-untyped-def]
        writes["n"] += 1
        return real_replace(df, user_ids=user_ids)

    sink.replace_recommendations_for_users = counting_replace  # type: ignore[method-assign]
    updater = IncrementalUpdater(
        sink=sink,
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        busy_check=busy,
    )
    assert updater.apply([normalize_event(event_payload())]) == 0
    assert writes["n"] == 0
    assert checks["n"] >= 2


def test_incremental_updater_write_busy_check_ignores_cached_start(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
    )
    writes = {"n": 0}
    sink = build_output_sink(settings.output)
    real_replace = sink.replace_recommendations_for_users

    def counting_replace(df, *, user_ids):  # type: ignore[no-untyped-def]
        writes["n"] += 1
        return real_replace(df, user_ids=user_ids)

    sink.replace_recommendations_for_users = counting_replace  # type: ignore[method-assign]
    updater = IncrementalUpdater(
        sink=sink,
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        busy_check=lambda: False,
        write_busy_check=lambda: True,
    )
    assert updater.apply([normalize_event(event_payload())]) == 0
    assert writes["n"] == 0


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
    real_load = load_recommendations_for_users

    def counting_load(output, user_ids):  # type: ignore[no-untyped-def]
        loads["n"] += 1
        return real_load(output, user_ids)

    monkeypatch.setattr("cicerone.events.updater.load_recommendations_for_users", counting_load)
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
    real_load = load_recommendations_for_users

    def counting_load(output, user_ids):  # type: ignore[no-untyped-def]
        loads["n"] += 1
        return real_load(output, user_ids)

    monkeypatch.setattr("cicerone.events.updater.load_recommendations_for_users", counting_load)
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


def test_incremental_updater_preserves_untouched_via_scoped_write(tmp_path, feature_config, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    existing = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"},
            {"user_id": "u2", "item_id": "keep", "rank": 1, "score": 0.5, "source": "personalized"},
        ]
    )
    existing.to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    sink = build_output_sink(settings.output)
    replace_calls: list[tuple[list[str], set[str]]] = []
    real_replace = sink.replace_recommendations_for_users

    def tracking_replace(df, *, user_ids):  # type: ignore[no-untyped-def]
        replace_calls.append((sorted(user_ids), set(df["user_id"].astype(str)) if not df.empty else set()))
        return real_replace(df, user_ids=user_ids)

    monkeypatch.setattr(sink, "replace_recommendations_for_users", tracking_replace)
    updater = IncrementalUpdater(
        sink=sink,
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
    )
    assert updater.apply([normalize_event(event_payload(user_id="u1", item_id="i9", event_id="s1"))]) == 1
    assert len(replace_calls) == 1
    user_ids, written_users = replace_calls[0]
    assert "u1" in user_ids
    assert COLD_START_USER_ID in user_ids
    assert "u2" not in user_ids
    assert "u2" not in written_users
    frame = load_recommendations_frame(settings.output)
    assert list(frame[frame["user_id"] == "u2"]["item_id"]) == ["keep"]
    assert frame[frame["user_id"] == "u1"]["item_id"].astype(str).tolist()  # non-empty updated
    assert "i9" in set(frame[frame["user_id"] == "u1"]["item_id"].astype(str))


def test_incremental_updater_user_cache_lru_evicts(tmp_path, feature_config):
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
        user_cache_max_size=2,
    )
    assert updater.apply([normalize_event(event_payload(user_id="u1", item_id="a", event_id="e1"))]) == 1
    assert updater.apply([normalize_event(event_payload(user_id="u2", item_id="b", event_id="e2"))]) == 1
    # Cap is 2; each apply also caches __cold_start__, so older users are evicted.
    assert len(updater.cached_user_ids) <= 2
    assert COLD_START_USER_ID in updater.cached_user_ids
    assert updater.apply([normalize_event(event_payload(user_id="u3", item_id="c", event_id="e3"))]) == 1
    assert len(updater.cached_user_ids) <= 2
    assert COLD_START_USER_ID in updater.cached_user_ids
    assert "u3" in updater.cached_user_ids


def test_incremental_updater_rejects_non_positive_cache_size(tmp_path, feature_config):
    out = tmp_path / "out"
    out.mkdir()
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
    )
    with pytest.raises(ValueError, match="user_cache_max_size"):
        IncrementalUpdater(
            sink=build_output_sink(settings.output),
            output_settings=settings.output,
            feature_config=feature_config,
            top_k=3,
            user_cache_max_size=0,
        )
