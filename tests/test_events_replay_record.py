from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import event, select
from support.events import event_payload
from support.prometheus_metrics import registry_metric_value

from cicerone.config import IOSettings, make_settings
from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.normalize import normalize_event
from cicerone.events.updater import IncrementalUpdater
from cicerone.events.webhook import WebhookEventSource
from cicerone.events.worker import EventWorker
from cicerone.feature_config import FeatureConfig
from cicerone.io import applied_events
from cicerone.io.applied_events import merge_applied_keys, replay_key
from cicerone.io.base import AppliedEventLog
from cicerone.io.factory import build_output_sink


def _outputs(tmp_path) -> dict[str, IOSettings]:  # type: ignore[no-untyped-def]
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i0", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    return {
        "dataset": IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        "db": IOSettings(kind="db", options={"database_url": f"sqlite+pysqlite:///{tmp_path / 'out.db'}"}),
    }


def _worker(
    output: IOSettings, feature_config: FeatureConfig, source: WebhookEventSource, batch_size: int = 1
):  # type: ignore[no-untyped-def]
    settings = make_settings(output=output, top_k=3)
    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=3,
    )
    return EventWorker(
        source, MicroBatchBuffer(batch_size=batch_size, batch_window_seconds=60.0), updater
    ), updater


@pytest.mark.parametrize("kind", ["dataset", "db"])
def test_a_redelivery_after_a_restart_is_acked_not_applied_again(tmp_path, feature_config, kind) -> None:
    output = _outputs(tmp_path)[kind]
    first_source = WebhookEventSource({})
    first_source.ingest(event_payload(event_id="replayed-1", user_id="u1", item_id="i9"))
    first, first_updater = _worker(output, feature_config, first_source)
    assert first.tick() == 1
    assert first_updater.events_applied == 1

    # A new process: nothing in memory, the source redelivers the same event.
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="replayed-1", user_id="u1", item_id="i9"))
    source.ingest(event_payload(event_id="fresh-2", user_id="u2", item_id="i8"))
    worker, updater = _worker(output, feature_config, source, batch_size=2)
    before = registry_metric_value("cicerone_events_replayed_total")
    assert worker.tick() == 1
    assert registry_metric_value("cicerone_events_replayed_total") == before + 1
    assert updater.events_applied == 1
    assert source.health().lag == 0
    recorded = updater.applied_before([_event("replayed-1"), _event("fresh-2")])
    assert [event.event_id for event in recorded] == ["replayed-1", "fresh-2"]


def _event(event_id: str):  # type: ignore[no-untyped-def]
    from cicerone.events.normalize import normalize_event

    return normalize_event(event_payload(event_id=event_id, user_id="u1", item_id="i9"))


@pytest.mark.parametrize("kind", ["dataset", "db"])
def test_sinks_record_applied_keys_only_with_the_rows(tmp_path, kind) -> None:
    sink = build_output_sink(_outputs(tmp_path)[kind])
    assert isinstance(sink, AppliedEventLog)
    assert sink.applied_event_keys(["k1"]) == set()
    rows = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}])
    sink.replace_recommendations_for_users(rows, user_ids=["u1"], applied_keys=["k1", "k2"])
    assert sink.applied_event_keys(["k1", "k3"]) == {"k1"}
    sink.replace_recommendations_for_users(rows, user_ids=["u1"])
    assert sink.applied_event_keys(["k1", "k2"]) == {"k1", "k2"}


def test_a_failed_dataset_write_records_nothing(tmp_path, monkeypatch) -> None:
    sink = build_output_sink(_outputs(tmp_path)["dataset"])

    def _write_fails(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(sink, "_write_bytes", _write_fails)
    rows = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}])
    with pytest.raises(OSError, match="disk full"):
        sink.replace_recommendations_for_users(rows, user_ids=["u1"], applied_keys=["k1"])
    monkeypatch.undo()
    assert sink.applied_event_keys(["k1"]) == set()


def test_db_record_drops_keys_older_than_the_retention(tmp_path, monkeypatch) -> None:
    sink = build_output_sink(_outputs(tmp_path)["db"])
    rows = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}])
    sink.replace_recommendations_for_users(rows, user_ids=["u1"], applied_keys=["old"])
    later = datetime.now(UTC) + applied_events.APPLIED_EVENTS_RETENTION + timedelta(minutes=1)
    monkeypatch.setattr("cicerone.io.db_store.datetime", _FrozenDatetime(later))
    sink.replace_recommendations_for_users(rows, user_ids=["u1"], applied_keys=["new"])
    assert sink.applied_event_keys(["old", "new"]) == {"new"}
    with sink._engine.connect() as conn:
        stored = {row[0] for row in conn.execute(select(sink._applied_events_table().c.event_key))}
    assert stored == {"new"}


class _FrozenDatetime:
    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self, tz=None):  # type: ignore[no-untyped-def]
        return self._now


def test_merge_applied_keys_drops_expired_and_caps_the_newest(monkeypatch) -> None:
    now = datetime(2026, 10, 20, tzinfo=UTC)
    expired = (now - applied_events.APPLIED_EVENTS_RETENTION - timedelta(seconds=1)).isoformat()
    kept = (now - timedelta(days=1)).isoformat()
    merged = merge_applied_keys({"expired": expired, "kept": kept}, ["new"], now)
    assert merged == {"kept": kept, "new": now.isoformat()}

    monkeypatch.setattr(applied_events, "APPLIED_EVENTS_MAX_KEYS", 2)
    capped = merge_applied_keys({"a": kept, "b": (now - timedelta(days=2)).isoformat()}, ["c"], now)
    assert capped == {"a": kept, "c": now.isoformat()}


def test_worker_returns_the_batch_when_the_record_lookup_fails(tmp_path, feature_config) -> None:
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="lookup-fails", user_id="u1", item_id="i9"))
    worker, updater = _worker(_outputs(tmp_path)["dataset"], feature_config, source)

    def _lookup_fails(_events):  # type: ignore[no-untyped-def]
        raise OSError("storage unreachable")

    updater.applied_before = _lookup_fails  # type: ignore[method-assign]
    assert worker.tick() == 0
    assert updater.events_applied == 0
    assert source.health().lag == 1


@pytest.mark.parametrize("kind", ["dataset", "db"])
def test_an_expired_key_no_longer_counts_before_any_prune(tmp_path, monkeypatch, kind) -> None:
    sink = build_output_sink(_outputs(tmp_path)[kind])
    rows = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}])
    sink.replace_recommendations_for_users(rows, user_ids=["u1"], applied_keys=["k1"])
    later = datetime.now(UTC) + applied_events.APPLIED_EVENTS_RETENTION + timedelta(minutes=1)
    monkeypatch.setattr(f"cicerone.io.{kind}_store.datetime", _FrozenDatetime(later))
    assert sink.applied_event_keys(["k1"]) == set()


def test_replay_keys_are_hashed_per_kind() -> None:
    assert replay_key("id", "abc") != replay_key("fingerprint", "abc")
    assert "abc" not in replay_key("id", "abc")


def test_generated_ids_get_no_durable_replay_key(tmp_path, feature_config) -> None:
    class _EphemeralSource(WebhookEventSource):
        ephemeral_event_ids = True

    worker, updater = _worker(_outputs(tmp_path)["dataset"], feature_config, _EphemeralSource({}))
    generated = replace(_event("generated-1"), generated_event_id=True)
    assert worker._replay_key(generated) is None
    assert worker._replay_key(_event("given-1")) == replay_key("id", "given-1")
    assert updater.applied_before([generated]) == []


def test_db_keys_are_scoped_to_their_recommendations_table(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'shared.db'}"
    first = build_output_sink(IOSettings(kind="db", options={"database_url": url}))
    second = build_output_sink(
        IOSettings(kind="db", options={"database_url": url, "recommendations_table": "other_recs"})
    )
    rows = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}])
    first.replace_recommendations_for_users(rows, user_ids=["u1"], applied_keys=["k1"])
    assert first.applied_event_keys(["k1"]) == {"k1"}
    assert second.applied_event_keys(["k1"]) == set()


def test_db_key_lookup_and_record_handle_more_keys_than_the_parameter_limit(tmp_path) -> None:
    sink = build_output_sink(_outputs(tmp_path)["db"])

    @event.listens_for(sink._engine, "connect")
    def _low_parameter_limit(dbapi_connection, _record):  # type: ignore[no-untyped-def]
        dbapi_connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 600)

    sink._engine.dispose()
    keys = [f"k{index}" for index in range(1_200)]
    rows = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}])
    sink.replace_recommendations_for_users(rows, user_ids=["u1"], applied_keys=keys)
    sink.replace_recommendations_for_users(rows, user_ids=["u1"], applied_keys=keys)
    assert sink.applied_event_keys(keys) == set(keys)


@pytest.mark.parametrize("kind", ["dataset", "db"])
def test_a_full_write_clears_the_applied_keys(tmp_path, kind) -> None:
    sink = build_output_sink(_outputs(tmp_path)[kind])
    rows = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}])
    sink.replace_recommendations_for_users(rows, user_ids=["u1"], applied_keys=["k1"])
    sink.write_recommendations(rows)
    assert sink.applied_event_keys(["k1"]) == set()


def test_worker_returns_the_batch_when_the_lookup_raises_unexpectedly(tmp_path, feature_config) -> None:
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="lookup-bug", user_id="u1", item_id="i9"))
    worker, updater = _worker(_outputs(tmp_path)["dataset"], feature_config, source)

    def _lookup_bug(_events):  # type: ignore[no-untyped-def]
        raise RuntimeError("lookup bug")

    updater.applied_before = _lookup_bug  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="lookup bug"):
        worker.tick()
    assert updater.events_applied == 0
    assert source.health().lag == 1


def test_merge_applied_keys_warns_when_the_cap_drops_unexpired_keys(monkeypatch, caplog) -> None:
    now = datetime(2026, 10, 20, tzinfo=UTC)
    monkeypatch.setattr(applied_events, "APPLIED_EVENTS_MAX_KEYS", 1)
    with caplog.at_level("WARNING", logger="cicerone.io.applied_events"):
        merge_applied_keys({"a": (now - timedelta(days=1)).isoformat()}, ["b"], now)
    assert "Dropping the oldest 1 applied-event key(s)" in caplog.text


def test_dataset_key_lookup_reads_no_recommendation_rows(tmp_path, monkeypatch) -> None:
    sink = build_output_sink(_outputs(tmp_path)["dataset"])
    rows = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}])
    sink.replace_recommendations_for_users(rows, user_ids=["u1"], applied_keys=["k1"])

    def _no_row_reads(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("the lookup must read only the parquet footer")

    monkeypatch.setattr("cicerone.io.dataset_store.read_parquet", _no_row_reads)
    assert sink.applied_event_keys(["k1", "k2"]) == {"k1"}


def test_a_dataset_full_write_drops_keys_carried_by_its_frame(tmp_path) -> None:
    sink = build_output_sink(_outputs(tmp_path)["dataset"])
    rows = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}])
    rows.attrs[applied_events.APPLIED_EVENTS_ATTR] = {"k1": datetime.now(UTC).isoformat()}
    sink.write_recommendations(rows)
    assert sink.applied_event_keys(["k1"]) == set()
    assert applied_events.APPLIED_EVENTS_ATTR in rows.attrs


def test_db_creates_the_key_table_outside_the_rows_transaction(tmp_path) -> None:
    sink = build_output_sink(_outputs(tmp_path)["db"])
    steps: list[str] = []

    @event.listens_for(sink._engine, "before_cursor_execute")
    def _statement(_conn, _cursor, statement, *_args):  # type: ignore[no-untyped-def]
        steps.append(statement.split("(")[0].strip())

    @event.listens_for(sink._engine, "commit")
    def _commit(_conn):  # type: ignore[no-untyped-def]
        steps.append("COMMIT")

    rows = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}])
    sink.replace_recommendations_for_users(rows, user_ids=["u1"], applied_keys=["k1"])
    create = next(i for i, step in enumerate(steps) if step.startswith("CREATE TABLE incremental_applied"))
    rows_delete = next(i for i, step in enumerate(steps) if step.startswith("DELETE FROM recommendations"))
    assert "COMMIT" in steps[create:rows_delete]


def test_worker_returns_fresh_events_when_acking_replays_raises(tmp_path, feature_config) -> None:
    output = _outputs(tmp_path)["dataset"]
    first_source = WebhookEventSource({})
    first_source.ingest(event_payload(event_id="replayed-x", user_id="u1", item_id="i9"))
    first, _first_updater = _worker(output, feature_config, first_source)
    assert first.tick() == 1

    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="replayed-x", user_id="u1", item_id="i9"))
    source.ingest(event_payload(event_id="fresh-x", user_id="u2", item_id="i8"))
    worker, updater = _worker(output, feature_config, source, batch_size=2)

    def _ack_bug(_events):  # type: ignore[no-untyped-def]
        raise RuntimeError("ack bug")

    worker._ack_unbuffered = _ack_bug  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="ack bug"):
        worker.tick()
    assert updater.events_applied == 0
    assert [event.event_id for event in source.poll(10)] == ["fresh-x"]


def test_db_keys_recorded_before_a_full_write_manifest_are_ignored(tmp_path) -> None:
    sink = build_output_sink(_outputs(tmp_path)["db"])
    rows = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}])
    sink.replace_recommendations_for_users(rows, user_ids=["u1"], applied_keys=["k1"])
    sink.write_manifest({"triggered_by": "incremental", "generated_at": datetime.now(UTC).isoformat()})
    assert sink.applied_event_keys(["k1"]) == {"k1"}

    # An older version's full write: rows and manifest replaced, the key table left alone.
    sink.write_manifest({"triggered_by": "cron", "generated_at": datetime.now(UTC).isoformat()})
    assert sink.applied_event_keys(["k1"]) == set()

    sink.replace_recommendations_for_users(rows, user_ids=["u1"], applied_keys=["k2"])
    assert sink.applied_event_keys(["k1", "k2"]) == {"k2"}


def test_db_lookup_without_a_match_skips_the_manifest_query(tmp_path, monkeypatch) -> None:
    sink = build_output_sink(_outputs(tmp_path)["db"])
    rows = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "popular"}])
    sink.replace_recommendations_for_users(rows, user_ids=["u1"], applied_keys=["k1"])

    def _manifest_query():  # type: ignore[no-untyped-def]
        raise AssertionError("manifest queried without a matching key")

    monkeypatch.setattr(sink, "_latest_full_write_at", _manifest_query)
    assert sink.applied_event_keys(["k2"]) == set()


def test_a_batch_carries_one_event_per_event_id() -> None:
    buffer = MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0)
    events = [
        normalize_event(event_payload(event_id="same", user_id="u1", item_id="i1")),
        normalize_event(event_payload(event_id="same", user_id="u2", item_id="i2")),
    ]
    result = buffer.extend(events)
    assert [event.user_id for event in result.duplicates] == ["u2"]
    assert [event.event_id for event in buffer.flush()] == ["same"]
