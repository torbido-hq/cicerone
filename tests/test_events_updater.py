from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace

import pandas as pd
import pytest
from support.events import event_payload

from cicerone.blending import COLD_START_USER_ID
from cicerone.config import IOSettings, make_settings
from cicerone.events.normalize import normalize_event
from cicerone.events.online_result import OnlineRefreshResult, empty_online_rows
from cicerone.events.store import load_recommendations_for_users, load_recommendations_frame
from cicerone.events.updater import INCREMENTAL_SOURCE, IncrementalUpdater
from cicerone.events.updater_policy import incremental_allowlists, incremental_needs_users_frame
from cicerone.feature_config import EligibilityRule, FeatureConfig
from cicerone.io.factory import build_output_sink
from cicerone.io.recommendation_reader import RECOMMENDATION_COLUMNS
from cicerone.locks import LockLostError, WriterLockBusyError
from cicerone.publish.base import PublishError
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


def test_incremental_updater_keeps_reasons_when_event_rehits_item(tmp_path, feature_config: FeatureConfig):
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
    updater.apply([normalize_event(event_payload(user_id="u1", item_id="old", event_id="r2"))])
    frame = load_recommendations_frame(settings.output)
    old = frame[(frame["user_id"] == "u1") & (frame["item_id"] == "old")].iloc[0]
    assert parse_reasons(old["reasons"]).sources[0].label == "personalized"
    assert old["source"] == "personalized"


def test_incremental_updater_unlabelled_prior_stays_on_control(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "old",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
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
        variant_names=("control", "treatment"),
    )
    updater.apply([normalize_event(event_payload(user_id="u1", item_id="i9", event_id="v1"))])
    frame = load_recommendations_frame(settings.output)
    u1 = frame[frame["user_id"] == "u1"]
    control = u1[u1["variant"] == "control"]
    treatment = u1[u1["variant"] == "treatment"]
    assert "old" in set(control["item_id"].astype(str))
    assert "old" not in set(treatment["item_id"].astype(str))


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


def test_incremental_updater_remakes_under_writer_lock_after_retrain(tmp_path, feature_config: FeatureConfig):
    from contextlib import contextmanager

    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    sink = build_output_sink(settings.output)
    retrain = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "job", "rank": 1, "score": 1.0, "source": "personalized"},
            {"user_id": "u2", "item_id": "x", "rank": 1, "score": 0.5, "source": "personalized"},
        ]
    )
    inner = sink.recommendations_write

    @contextmanager
    def after_retrain():
        sink.write_recommendations(retrain)
        with inner():
            yield

    sink.recommendations_write = after_retrain  # type: ignore[method-assign]
    updater = IncrementalUpdater(
        sink=sink,
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
    )
    assert updater.apply([normalize_event(event_payload(user_id="u1", item_id="i9"))]) == 1
    frame = load_recommendations_frame(settings.output)
    u1 = set(frame.loc[frame["user_id"] == "u1", "item_id"].astype(str))
    assert "job" in u1
    assert "i9" in u1
    assert list(frame.loc[frame["user_id"] == "u2", "item_id"]) == ["x"]


def test_incremental_updater_rechecks_busy_after_merge(tmp_path, feature_config: FeatureConfig):
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
        return checks["n"] >= 3

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
    assert checks["n"] >= 3


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


def test_persist_online_skipped_when_write_busy(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )

    class _FakeOnline:
        def __init__(self) -> None:
            self.commits = 0
            self.aborts = 0

        def refresh(self, events):  # type: ignore[no-untyped-def]
            del events
            return OnlineRefreshResult(rows=empty_online_rows())

        def invalidate(self) -> None:
            return None

        def commit(self) -> None:
            self.commits += 1

        def abort(self) -> None:
            self.aborts += 1

    online = _FakeOnline()
    busy = {"v": False}
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        busy_check=lambda: False,
        write_busy_check=lambda: busy["v"],
        online=online,
    )
    assert updater.apply([normalize_event(event_payload())], persist_online=False) == 1
    assert online.commits == 0
    busy["v"] = True
    updater.persist_online()
    assert online.commits == 0
    assert online.aborts == 1


def test_persist_online_rechecks_busy_after_writer_wait(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    busy = {"v": False}

    class _Sink:
        def recommendations_write(self):
            busy["v"] = True
            return nullcontext()

    class _FakeOnline:
        def __init__(self) -> None:
            self.commits = 0
            self.aborts = 0

        def refresh(self, events):  # type: ignore[no-untyped-def]
            del events
            return OnlineRefreshResult(rows=empty_online_rows())

        def invalidate(self) -> None:
            return None

        def commit(self) -> None:
            self.commits += 1

        def abort(self) -> None:
            self.aborts += 1

    online = _FakeOnline()
    updater = IncrementalUpdater(
        sink=_Sink(),  # type: ignore[arg-type]
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        write_busy_check=lambda: busy["v"],
        online=online,
    )
    updater.persist_online()
    assert online.commits == 0
    assert online.aborts == 1


def test_persist_online_holds_dataset_writer_lock(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    sink = build_output_sink(settings.output)
    depths: list[int] = []

    class _FakeOnline:
        def refresh(self, events):  # type: ignore[no-untyped-def]
            del events
            return OnlineRefreshResult(rows=empty_online_rows())

        def invalidate(self) -> None:
            return None

        def commit(self) -> None:
            depths.append(sink._recs_write_depth())

        def abort(self) -> None:
            return None

    updater = IncrementalUpdater(
        sink=sink,
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        online=_FakeOnline(),
    )
    updater.persist_online()
    assert depths == [1]


def test_persist_online_fences_without_dataset_lock(tmp_path, feature_config: FeatureConfig):
    commits: list[int] = []
    owned = {"v": True}

    class _Sink:
        pass

    class _FakeOnline:
        def refresh(self, events):  # type: ignore[no-untyped-def]
            del events
            return OnlineRefreshResult(rows=empty_online_rows())

        def invalidate(self) -> None:
            return None

        def commit(self) -> None:
            commits.append(1)

        def abort(self) -> None:
            return None

    updater = IncrementalUpdater(
        sink=_Sink(),  # type: ignore[arg-type]
        output_settings=IOSettings(kind="db", options={"database_url": "sqlite+pysqlite://"}),
        feature_config=feature_config,
        top_k=3,
        fence_check=lambda: owned["v"],
        online=_FakeOnline(),
    )
    updater.persist_online()
    assert commits == [1]
    owned["v"] = False
    with pytest.raises(LockLostError):
        updater.persist_online()
    assert commits == [1]


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


def test_incremental_updater_unknown_event_skips_new_user_without_prior(
    tmp_path, feature_config: FeatureConfig
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=3,
    )
    published: list[tuple[pd.DataFrame, list[str] | None]] = []

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame, *, user_ids=None) -> None:
            published.append((df.copy(), None if user_ids is None else list(user_ids)))

        def close(self) -> None:
            return None

    IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
        publisher=_Pub(),
    ).apply([normalize_event(event_payload(event_type="unknown_type", event_id="u", item_id="ix"))])
    assert published == []
    assert load_recommendations_frame(settings.output).empty


def test_incremental_updater_tombstones_new_user_when_signal_is_ineligible(
    tmp_path, feature_config: FeatureConfig
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    items = pd.DataFrame([{"item_id": "oos", "published": True, "in_stock": False}])
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    published: list[tuple[pd.DataFrame, list[str] | None]] = []

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame, *, user_ids=None) -> None:
            published.append((df.copy(), None if user_ids is None else list(user_ids)))

        def close(self) -> None:
            return None

    IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        items_provider=lambda: items,
        publisher=_Pub(),
    ).apply([normalize_event(event_payload(user_id="u1", item_id="oos", event_id="new-oos"))])
    assert load_recommendations_frame(settings.output).empty
    assert len(published) == 1
    merged, user_ids = published[0]
    assert user_ids == ["u1"]
    assert merged.empty


def test_incremental_updater_legacy_publisher_without_user_ids(
    tmp_path, feature_config: FeatureConfig
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    published: list[pd.DataFrame] = []

    class _Legacy:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            published.append(df.copy())

        def close(self) -> None:
            return None

    IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        publisher=_Legacy(),
    ).apply([normalize_event(event_payload(user_id="u1", item_id="i9", event_id="legacy-pub"))])
    assert len(published) == 1
    assert "i9" in set(load_recommendations_frame(settings.output)["item_id"].astype(str))


def test_incremental_updater_legacy_publisher_does_not_retry_after_tombstone(
    tmp_path, feature_config: FeatureConfig
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "oos", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    items = pd.DataFrame([{"item_id": "oos", "published": True, "in_stock": False}])
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )

    class _Legacy:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            raise AssertionError("legacy publish must not drop tombstones")

        def close(self) -> None:
            return None

    applied = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        items_provider=lambda: items,
        publisher=_Legacy(),
    ).apply([normalize_event(event_payload(user_id="u1", item_id="oos", event_id="legacy-tomb"))])
    assert applied == 1
    frame = load_recommendations_frame(settings.output)
    assert frame[frame["user_id"] == "u1"].empty


def test_incremental_updater_legacy_publisher_publishes_nonempty_before_tombstone(
    tmp_path, feature_config: FeatureConfig
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"},
            {"user_id": "u2", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"},
        ]
    ).to_parquet(out / "recommendations.parquet", index=False)
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
    items = pd.DataFrame(
        [
            {"item_id": "old", "published": True, "in_stock": True, "region_slug": "lazio"},
            {"item_id": "i9", "published": True, "in_stock": True, "region_slug": "lazio"},
        ]
    )
    users = pd.DataFrame(
        [
            {"user_id": "u1", "region_slug": "nowhere"},
            {"user_id": "u2", "region_slug": "lazio"},
        ]
    )
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    published: list[pd.DataFrame] = []

    class _Legacy:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame) -> None:
            published.append(df.copy())

        def close(self) -> None:
            return None

    applied = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=scoped,
        top_k=5,
        items_provider=lambda: items,
        users_provider=lambda: users,
        publisher=_Legacy(),
    ).apply(
        [
            normalize_event(event_payload(user_id="u1", item_id="i9", event_id="mix-tomb")),
            normalize_event(event_payload(user_id="u2", item_id="i9", event_id="mix-keep")),
        ]
    )
    assert applied == 2
    assert len(published) == 1
    assert set(published[0]["user_id"].astype(str)) == {"u2"}
    frame = load_recommendations_frame(settings.output)
    assert frame[frame["user_id"] == "u1"].empty
    assert "i9" in set(frame[frame["user_id"] == "u2"]["item_id"].astype(str))


def test_incremental_updater_unknown_event_keeps_popular_only_user(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "pop", "rank": 1, "score": 0.2, "source": "popular_fallback"},
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
    assert (
        updater.apply(
            [
                normalize_event(
                    event_payload(user_id="u1", event_type="unknown_type", event_id="u", item_id="ix")
                )
            ]
        )
        == 1
    )
    frame = load_recommendations_frame(settings.output)
    u1 = frame[frame["user_id"] == "u1"]
    assert list(u1["item_id"].astype(str)) == ["pop"]


def test_incremental_updater_mixed_batch_unknown_keeps_popular_only_user(
    tmp_path, feature_config: FeatureConfig
):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "pop", "rank": 1, "score": 0.2, "source": "popular_fallback"},
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
    assert (
        updater.apply(
            [
                normalize_event(
                    event_payload(user_id="u1", event_type="unknown_type", event_id="u", item_id="ix")
                ),
                normalize_event(
                    event_payload(user_id="u2", event_type="purchase", event_id="p", item_id="bought")
                ),
            ]
        )
        == 2
    )
    frame = load_recommendations_frame(settings.output)
    u1 = frame[frame["user_id"] == "u1"]
    assert list(u1["item_id"].astype(str)) == ["pop"]
    u2 = frame[frame["user_id"] == "u2"]
    assert "bought" in set(u2["item_id"].astype(str))


def test_incremental_updater_zero_weight_event_keeps_popular_only_user(
    tmp_path, feature_config: FeatureConfig
):
    from dataclasses import replace

    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "pop", "rank": 1, "score": 0.2, "source": "popular_fallback"},
        ]
    ).to_parquet(out / "recommendations.parquet", index=False)
    zero_weight = replace(
        feature_config,
        event_weights={**feature_config.event_weights, "view": 0.0},
    )
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=zero_weight,
        top_k=5,
    )
    assert (
        updater.apply(
            [
                normalize_event(event_payload(user_id="u1", event_type="view", event_id="z", item_id="ix")),
            ]
        )
        == 1
    )
    frame = load_recommendations_frame(settings.output)
    u1 = frame[frame["user_id"] == "u1"]
    assert list(u1["item_id"].astype(str)) == ["pop"]


def test_incremental_updater_preserves_best_ranks_when_capping(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    # Unsorted ranks: head() without sort_values would keep p2 and drop p1.
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "p2", "rank": 2, "score": 0.5, "source": "personalized"},
            {"user_id": "u1", "item_id": "p1", "rank": 1, "score": 1.0, "source": "personalized"},
            {"user_id": "u1", "item_id": "p3", "rank": 3, "score": 0.1, "source": "personalized"},
        ]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=2,
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=2,
    )
    event = normalize_event(event_payload(user_id="u1", item_id="boosted", event_id="b1"))
    assert updater.apply([event]) == 1
    u1 = load_recommendations_frame(settings.output)
    u1 = u1[u1["user_id"] == "u1"].sort_values("rank")
    assert list(u1["item_id"].astype(str)) == ["boosted", "p1"]


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


def test_incremental_updater_reloads_affected_users_each_apply(tmp_path, feature_config, monkeypatch):
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

    monkeypatch.setattr("cicerone.events.updater_cache.load_recommendations_for_users", counting_load)
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
    )
    assert updater.apply([normalize_event(event_payload(event_id="c1", item_id="a"))]) == 1
    assert loads["n"] == 1
    assert updater.apply([normalize_event(event_payload(event_id="c2", item_id="b"))]) == 1
    assert loads["n"] == 2


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

    monkeypatch.setattr("cicerone.events.updater_cache.load_recommendations_for_users", counting_load)
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
    assert COLD_START_USER_ID not in user_ids
    assert "u2" not in user_ids
    assert "u2" not in written_users
    frame = load_recommendations_frame(settings.output)
    assert list(frame[frame["user_id"] == "u2"]["item_id"]) == ["keep"]
    assert frame[frame["user_id"] == "u1"]["item_id"].astype(str).tolist()  # non-empty updated
    assert "i9" in set(frame[frame["user_id"] == "u1"]["item_id"].astype(str))


def test_incremental_updater_reloads_users_under_fence(tmp_path, feature_config):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
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
        fence_check=lambda: True,
    )
    assert updater.apply([normalize_event(event_payload(user_id="u1", item_id="i9", event_id="e1"))]) == 1
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "job", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    assert updater.apply([normalize_event(event_payload(user_id="u1", item_id="i8", event_id="e2"))]) == 1
    frame = load_recommendations_frame(settings.output)
    items = set(frame[frame["user_id"] == "u1"]["item_id"].astype(str))
    assert "job" in items
    assert "old" not in items


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
    assert len(updater.cached_user_ids) <= 2
    assert updater.apply([normalize_event(event_payload(user_id="u3", item_id="c", event_id="e3"))]) == 1
    assert len(updater.cached_user_ids) <= 2
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


def test_popular_ranking_drops_zero_weight_and_breaks_item_ties(tmp_path, feature_config: FeatureConfig):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
        top_k=2,
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=2,
    )
    batch = pd.DataFrame(
        [
            {"event_type": "view", "quantity": 1, "item_id": "b"},
            {"event_type": "view", "quantity": 1, "item_id": "a"},
            {"event_type": "unknown_type", "quantity": 9, "item_id": "z"},
        ]
    )
    ranked = updater._popular_ranking(batch)
    assert list(ranked["item_id"]) == ["a", "b"]
    assert list(ranked["source"]) == ["popular_fallback", "popular_fallback"]
    assert ranked.iloc[0]["score"] == pytest.approx(ranked.iloc[1]["score"])


def test_popular_ranking_drops_negative_weights(tmp_path, feature_config: FeatureConfig):
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
        top_k=2,
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=2,
    )
    batch = pd.DataFrame(
        [
            {"event_type": "review_negative", "quantity": 1, "item_id": "hate"},
            {"event_type": "view", "quantity": 1, "item_id": "ok"},
        ]
    )
    ranked = updater._popular_ranking(batch)
    assert list(ranked["item_id"]) == ["ok"]
    reused = updater._popular_ranking(batch, updater._row_signal_weights(batch))
    assert list(reused["item_id"]) == ["ok"]


def test_incremental_updater_preserves_variants(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    existing = pd.DataFrame(
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
            {
                "user_id": COLD_START_USER_ID,
                "item_id": "cold-control",
                "rank": 1,
                "score": 0.2,
                "source": "popular_fallback",
                "variant": "control",
            },
            {
                "user_id": COLD_START_USER_ID,
                "item_id": "cold-treatment",
                "rank": 1,
                "score": 0.2,
                "source": "popular_fallback",
                "variant": "treatment",
            },
        ]
    )
    existing.to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        variant_names=("control", "treatment"),
    )
    events = [normalize_event(event_payload(user_id="u1", item_id="i9", event_id="n1"))]
    assert updater.apply(events) == 1
    frame = load_recommendations_frame(settings.output)
    u1 = frame[frame["user_id"] == "u1"]
    assert set(u1["variant"].astype(str)) == {"control", "treatment"}
    control_items = set(u1[u1["variant"] == "control"]["item_id"].astype(str))
    treatment_items = set(u1[u1["variant"] == "treatment"]["item_id"].astype(str))
    assert "i9" in control_items
    assert "old-control" in control_items
    assert "i9" not in treatment_items
    assert "old-treatment" in treatment_items
    cold = frame[frame["user_id"] == COLD_START_USER_ID]
    assert set(cold["variant"].astype(str)) == {"control", "treatment"}


def test_incremental_updater_keeps_parked_popular_variant(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    existing = pd.DataFrame(
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
                "item_id": "parked-popular",
                "rank": 1,
                "score": 0.2,
                "source": "popular_fallback",
                "variant": "treatment",
            },
        ]
    )
    existing.to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        variant_names=("control", "treatment"),
        assign_variant=lambda _user_id: "control",
    )
    events = [normalize_event(event_payload(user_id="u1", item_id="i9", event_id="n1"))]
    assert updater.apply(events) == 1
    frame = load_recommendations_frame(settings.output)
    u1 = frame[frame["user_id"] == "u1"]
    assert set(u1["variant"].astype(str)) == {"control", "treatment"}
    parked = u1[u1["variant"] == "treatment"]
    assert list(parked["item_id"].astype(str)) == ["parked-popular"]


def test_incremental_updater_collapses_leftover_variants_when_experiment_off(
    tmp_path, feature_config: FeatureConfig
):
    out = tmp_path / "out"
    out.mkdir()
    existing = pd.DataFrame(
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
            {
                "user_id": COLD_START_USER_ID,
                "item_id": "cold-control",
                "rank": 1,
                "score": 0.2,
                "source": "popular_fallback",
                "variant": "control",
            },
            {
                "user_id": COLD_START_USER_ID,
                "item_id": "cold-treatment",
                "rank": 1,
                "score": 0.2,
                "source": "popular_fallback",
                "variant": "treatment",
            },
        ]
    )
    existing.to_parquet(out / "recommendations.parquet", index=False)
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
    events = [normalize_event(event_payload(user_id="u1", item_id="i9", event_id="n1"))]
    assert updater.apply(events) == 1
    frame = load_recommendations_frame(settings.output)
    u1 = frame[frame["user_id"] == "u1"]
    assert set(u1["variant"].astype(str)) == {"control"}
    assert "old-control" in set(u1["item_id"].astype(str))
    assert "old-treatment" not in set(u1["item_id"].astype(str))
    cold = frame[frame["user_id"] == COLD_START_USER_ID]
    assert set(cold["variant"].astype(str)) == {"control"}
    assert "cold-treatment" not in set(cold["item_id"].astype(str))


def test_incremental_updater_publish_failure_does_not_unsucceed(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    called = {"n": 0}

    class _Boom:
        def connect(self) -> None:
            return None

        def publish(self, _df: pd.DataFrame, *, user_ids=None) -> None:
            raise PublishError("broker down")

        def close(self) -> None:
            return None

    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        on_success=lambda: called.__setitem__("n", called["n"] + 1),
        publisher=_Boom(),
    )
    events = [normalize_event(event_payload(user_id="u1", item_id="i9", event_id="n1"))]
    assert updater.apply(events) == 1
    assert called["n"] == 1
    frame = load_recommendations_frame(settings.output)
    assert "i9" in set(frame[frame["user_id"] == "u1"]["item_id"].astype(str))


def test_incremental_updater_raises_when_fence_lost_after_connect(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    published: list[pd.DataFrame] = []
    lost_after_connect = {"lost": False}

    class _Pub:
        def connect(self) -> None:
            lost_after_connect["lost"] = True

        def publish(self, df: pd.DataFrame, *, user_ids=None) -> None:
            published.append(df.copy())

        def close(self) -> None:
            return None

    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        fence_check=lambda: not lost_after_connect["lost"],
        publisher=_Pub(),
    )
    events = [normalize_event(event_payload(user_id="u1", item_id="i9", event_id="n1"))]
    with pytest.raises(LockLostError, match="events apply lock lost before write"):
        updater.apply(events)
    assert published == []
    frame = load_recommendations_frame(settings.output)
    assert "i9" in set(frame[frame["user_id"] == "u1"]["item_id"].astype(str))


def test_incremental_updater_raises_when_fence_lost_before_connect(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    connected = {"n": 0}

    class _Pub:
        def connect(self) -> None:
            connected["n"] += 1
            raise RuntimeError("broker down")

        def publish(self, df: pd.DataFrame, *, user_ids=None) -> None:
            raise AssertionError("publish should not run")

        def close(self) -> None:
            return None

    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        fence_check=lambda: False,
        publisher=_Pub(),
    )
    with pytest.raises(LockLostError, match="events apply lock lost before write"):
        updater._publish_sidecar(
            pd.DataFrame([{"user_id": "u1", "item_id": "i9", "rank": 1, "score": 1.0}]),
            "2026-01-01T00:00:00+00:00",
            ["u1"],
        )
    assert connected["n"] == 0


def test_incremental_updater_skips_publish_when_manifest_generation_changes(
    tmp_path, feature_config: FeatureConfig
):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    published: list[pd.DataFrame] = []

    class _Pub:
        def connect(self) -> None:
            path = out / "manifest.json"
            payload = json.loads(path.read_text())
            payload["generated_at"] = "2099-01-01T00:00:00+00:00"
            path.write_text(json.dumps(payload))

        def publish(self, df: pd.DataFrame, *, user_ids=None) -> None:
            published.append(df.copy())

        def close(self) -> None:
            return None

    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        publisher=_Pub(),
    )
    events = [normalize_event(event_payload(user_id="u1", item_id="i9", event_id="n1"))]
    assert updater.apply(events) == 1
    assert published == []


def test_incremental_updater_reraises_unexpected_sidecar_generation_error(
    tmp_path, feature_config: FeatureConfig, monkeypatch
):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame, *, user_ids=None) -> None:
            raise AssertionError("publish should not run after generation check failure")

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "cicerone.events.updater.sidecar_generation_current",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("generation check bug")),
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        publisher=_Pub(),
    )
    events = [normalize_event(event_payload(user_id="u1", item_id="i9", event_id="n1"))]
    with pytest.raises(RuntimeError, match="generation check bug"):
        updater.apply(events)
    frame = load_recommendations_frame(settings.output)
    assert "i9" in set(frame[frame["user_id"] == "u1"]["item_id"].astype(str))


def test_incremental_updater_reraises_writer_lock_busy_from_generation_check(
    tmp_path, feature_config: FeatureConfig, monkeypatch
):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame, *, user_ids=None) -> None:
            raise AssertionError("publish should not run after lock-busy generation check")

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "cicerone.events.updater.sidecar_generation_current",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(WriterLockBusyError("dataset writer lock busy")),
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        publisher=_Pub(),
    )
    events = [normalize_event(event_payload(user_id="u1", item_id="i9", event_id="n1"))]
    with pytest.raises(WriterLockBusyError, match="dataset writer lock busy"):
        updater.apply(events)
    frame = load_recommendations_frame(settings.output)
    assert "i9" in set(frame[frame["user_id"] == "u1"]["item_id"].astype(str))


def test_incremental_allowlists_filters_unavailable_items(feature_config: FeatureConfig) -> None:
    items = pd.DataFrame(
        [
            {"item_id": "ok", "published": True, "in_stock": True},
            {"item_id": "oos", "published": True, "in_stock": False},
        ]
    )
    allowed = incremental_allowlists(
        ["u1"],
        feature_config=feature_config,
        items=items,
    )
    assert allowed["u1"] == frozenset({"ok"})


def test_incremental_allowlists_fail_open_without_items(feature_config: FeatureConfig) -> None:
    allowed = incremental_allowlists(["u1"], feature_config=feature_config, items=None)
    assert allowed["u1"] is None


def test_incremental_updater_drops_ineligible_boost(tmp_path, feature_config: FeatureConfig) -> None:
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    items = pd.DataFrame(
        [
            {"item_id": "old", "published": True, "in_stock": True},
            {"item_id": "oos", "published": True, "in_stock": False},
            {"item_id": "ok", "published": True, "in_stock": True},
        ]
    )
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        items_provider=lambda: items,
    )
    updater.apply(
        [
            normalize_event(event_payload(user_id="u1", item_id="oos", event_id="bad")),
            normalize_event(event_payload(user_id="u1", item_id="ok", event_id="good")),
        ]
    )
    u1 = load_recommendations_frame(settings.output)
    u1 = u1[u1["user_id"] == "u1"]
    assert "oos" not in set(u1["item_id"].astype(str))
    assert "ok" in set(u1["item_id"].astype(str))
    assert "old" in set(u1["item_id"].astype(str))


def test_incremental_updater_keeps_batch_popular_and_cold_start(
    tmp_path, feature_config: FeatureConfig
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "batch-pop", "rank": 1, "score": 0.4, "source": "popular_fallback"},
            {
                "user_id": COLD_START_USER_ID,
                "item_id": "cold-keep",
                "rank": 1,
                "score": 0.2,
                "source": "popular_fallback",
            },
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
    updater.apply([normalize_event(event_payload(user_id="u1", item_id="viral", event_id="v1"))])
    frame = load_recommendations_frame(settings.output)
    u1 = frame[frame["user_id"] == "u1"]
    assert "batch-pop" in set(u1["item_id"].astype(str))
    assert "viral" in set(u1["item_id"].astype(str))
    cold = frame[frame["user_id"] == COLD_START_USER_ID]
    assert list(cold["item_id"].astype(str)) == ["cold-keep"]


def test_incremental_allowlists_user_scoped_when_users_present(feature_config: FeatureConfig) -> None:
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
    items = pd.DataFrame(
        [
            {"item_id": "ok", "published": True, "in_stock": True, "region_slug": "lazio"},
            {"item_id": "other", "published": True, "in_stock": True, "region_slug": "toscana"},
        ]
    )
    users = pd.DataFrame([{"user_id": "u1", "region_slug": "lazio"}])
    allowed = incremental_allowlists(["u1"], feature_config=scoped, items=items, users=users)
    assert allowed["u1"] == frozenset({"ok"})
    without_users = incremental_allowlists(["u1"], feature_config=scoped, items=items, users=None)
    assert without_users["u1"] == frozenset({"ok", "other"})
    empty_users = incremental_allowlists(["u1"], feature_config=scoped, items=items, users=pd.DataFrame())
    assert empty_users["u1"] == frozenset()


def test_incremental_needs_users_frame_uses_applied_variant_recipes(feature_config: FeatureConfig) -> None:
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
    open_cfg = replace(feature_config, eligibility=[], merge_item_availability=False)
    assert incremental_needs_users_frame(scoped) is True
    assert incremental_needs_users_frame(scoped, {"control": open_cfg, "treatment": open_cfg}) is False
    assert incremental_needs_users_frame(scoped, {"control": open_cfg, "treatment": scoped}) is True


def test_incremental_updater_deletes_user_when_allowlist_empties_list(
    tmp_path, feature_config: FeatureConfig
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "oos", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    items = pd.DataFrame([{"item_id": "oos", "published": True, "in_stock": False}])
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        items_provider=lambda: items,
    )
    updater.apply([normalize_event(event_payload(user_id="u1", item_id="oos", event_id="gone"))])
    frame = load_recommendations_frame(settings.output)
    assert frame[frame["user_id"] == "u1"].empty


def test_incremental_updater_publishes_empty_list_when_allowlist_empties(
    tmp_path, feature_config: FeatureConfig
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "oos", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    items = pd.DataFrame([{"item_id": "oos", "published": True, "in_stock": False}])
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    published: list[tuple[pd.DataFrame, list[str] | None]] = []

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame, *, user_ids=None) -> None:
            published.append((df.copy(), None if user_ids is None else list(user_ids)))

        def close(self) -> None:
            return None

    IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        items_provider=lambda: items,
        publisher=_Pub(),
    ).apply([normalize_event(event_payload(user_id="u1", item_id="oos", event_id="gone-pub"))])
    frame = load_recommendations_frame(settings.output)
    assert frame[frame["user_id"] == "u1"].empty
    assert len(published) == 1
    merged, user_ids = published[0]
    assert user_ids == ["u1"]
    assert merged.empty


def test_incremental_updater_publishes_empty_list_when_users_frame_is_empty(
    tmp_path, feature_config: FeatureConfig
) -> None:
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
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "ok", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    items = pd.DataFrame([{"item_id": "ok", "published": True, "in_stock": True, "region_slug": "lazio"}])
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    published: list[tuple[pd.DataFrame, list[str] | None]] = []

    class _Pub:
        def connect(self) -> None:
            return None

        def publish(self, df: pd.DataFrame, *, user_ids=None) -> None:
            published.append((df.copy(), None if user_ids is None else list(user_ids)))

        def close(self) -> None:
            return None

    IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=scoped,
        top_k=5,
        items_provider=lambda: items,
        users_provider=lambda: pd.DataFrame(),
        publisher=_Pub(),
    ).apply([normalize_event(event_payload(user_id="u1", item_id="ok", event_id="empty-users"))])
    frame = load_recommendations_frame(settings.output)
    assert frame[frame["user_id"] == "u1"].empty
    assert len(published) == 1
    merged, user_ids = published[0]
    assert user_ids == ["u1"]
    assert merged.empty


def test_incremental_updater_deletes_cold_start_when_allowlist_empties_list(
    tmp_path, feature_config: FeatureConfig
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "ok", "rank": 1, "score": 1.0, "source": "personalized"},
            {
                "user_id": COLD_START_USER_ID,
                "item_id": "oos",
                "rank": 1,
                "score": 0.2,
                "source": "popular_fallback",
            },
        ]
    ).to_parquet(out / "recommendations.parquet", index=False)
    items = pd.DataFrame(
        [
            {"item_id": "ok", "published": True, "in_stock": True},
            {"item_id": "oos", "published": True, "in_stock": False},
        ]
    )
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        items_provider=lambda: items,
    )
    updater.apply([normalize_event(event_payload(user_id="u1", item_id="ok", event_id="keep"))])
    frame = load_recommendations_frame(settings.output)
    assert frame[frame["user_id"] == COLD_START_USER_ID].empty
    assert "ok" in set(frame[frame["user_id"] == "u1"]["item_id"].astype(str))


def test_incremental_updater_does_not_restore_ineligible_parked_variant(
    tmp_path, feature_config: FeatureConfig
) -> None:
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
                "score": 0.8,
                "source": "personalized",
                "variant": "treatment",
            },
        ]
    ).to_parquet(out / "recommendations.parquet", index=False)
    items = pd.DataFrame(
        [
            {"item_id": "old-control", "published": True, "in_stock": True},
            {"item_id": "old-treatment", "published": True, "in_stock": False},
            {"item_id": "ok", "published": True, "in_stock": True},
        ]
    )
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        variant_names=("control", "treatment"),
        assign_variant=lambda _user_id: "control",
        items_provider=lambda: items,
    )
    updater.apply([normalize_event(event_payload(user_id="u1", item_id="ok", event_id="v-elig"))])
    u1 = load_recommendations_frame(settings.output)
    u1 = u1[u1["user_id"] == "u1"]
    assert "old-treatment" not in set(u1["item_id"].astype(str))
    assert "old-control" in set(u1[u1["variant"] == "control"]["item_id"].astype(str))


def test_incremental_updater_does_not_restore_ineligible_cold_variant(
    tmp_path, feature_config: FeatureConfig
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {
                "user_id": COLD_START_USER_ID,
                "item_id": "cold-ok",
                "rank": 1,
                "score": 0.2,
                "source": "popular_fallback",
                "variant": "control",
            },
            {
                "user_id": COLD_START_USER_ID,
                "item_id": "cold-oos",
                "rank": 1,
                "score": 0.2,
                "source": "popular_fallback",
                "variant": "treatment",
            },
            {"user_id": "u1", "item_id": "ok", "rank": 1, "score": 1.0, "source": "personalized"},
        ]
    ).to_parquet(out / "recommendations.parquet", index=False)
    items = pd.DataFrame(
        [
            {"item_id": "cold-ok", "published": True, "in_stock": True},
            {"item_id": "cold-oos", "published": True, "in_stock": False},
            {"item_id": "ok", "published": True, "in_stock": True},
        ]
    )
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        variant_names=("control", "treatment"),
        items_provider=lambda: items,
    )
    updater.apply([normalize_event(event_payload(user_id="u1", item_id="ok", event_id="cold-elig"))])
    cold = load_recommendations_frame(settings.output)
    cold = cold[cold["user_id"] == COLD_START_USER_ID]
    assert "cold-oos" not in set(cold["item_id"].astype(str))
    assert "cold-ok" in set(cold["item_id"].astype(str))


def test_incremental_updater_skips_users_read_without_user_scoped_rules(
    tmp_path, feature_config: FeatureConfig
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "ok", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    items = pd.DataFrame([{"item_id": "ok", "published": True, "in_stock": True}])
    calls = {"n": 0}

    def users() -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame([{"user_id": "u1", "region_slug": "lazio"}])

    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        items_provider=lambda: items,
        users_provider=users,
    ).apply([normalize_event(event_payload(user_id="u1", item_id="ok", event_id="no-users"))])
    assert calls["n"] == 0


def test_incremental_updater_reads_users_for_user_scoped_rules(
    tmp_path, feature_config: FeatureConfig
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "ok", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    items = pd.DataFrame(
        [
            {"item_id": "ok", "published": True, "in_stock": True, "region_slug": "lazio"},
            {"item_id": "other", "published": True, "in_stock": True, "region_slug": "toscana"},
        ]
    )
    calls = {"n": 0}

    def users() -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame([{"user_id": "u1", "region_slug": "lazio"}])

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
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=scoped,
        top_k=5,
        items_provider=lambda: items,
        users_provider=users,
    ).apply([normalize_event(event_payload(user_id="u1", item_id="ok", event_id="need-users"))])
    assert calls["n"] == 1


def test_incremental_updater_uses_per_variant_eligibility(tmp_path, feature_config: FeatureConfig) -> None:
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "oos",
                "rank": 1,
                "score": 1.0,
                "source": "personalized",
                "variant": "control",
            },
            {
                "user_id": "u1",
                "item_id": "oos",
                "rank": 1,
                "score": 0.8,
                "source": "personalized",
                "variant": "treatment",
            },
        ]
    ).to_parquet(out / "recommendations.parquet", index=False)
    items = pd.DataFrame(
        [
            {"item_id": "oos", "published": True, "in_stock": False},
            {"item_id": "ok", "published": True, "in_stock": True},
        ]
    )
    open_cfg = replace(feature_config, eligibility=[], merge_item_availability=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        variant_names=("control", "treatment"),
        assign_variant=lambda _user_id: "control",
        items_provider=lambda: items,
        variant_feature_configs={"control": open_cfg, "treatment": feature_config},
    ).apply([normalize_event(event_payload(user_id="u1", item_id="ok", event_id="per-arm"))])
    u1 = load_recommendations_frame(settings.output)
    u1 = u1[u1["user_id"] == "u1"]
    control_items = set(u1[u1["variant"] == "control"]["item_id"].astype(str))
    treatment_items = set(u1[u1["variant"] == "treatment"]["item_id"].astype(str))
    assert "oos" in control_items
    assert "oos" not in treatment_items
