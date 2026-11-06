from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from support.fakes import FakeReader, patch_module_ref, raising
from support.prometheus_metrics import registry_metric_value
from support.settings import make_settings
from support.waits import WAIT_SECONDS
from test_serve import _recs_df

from cicerone.config import ConfigError, EventsSettings, IOSettings
from cicerone.events.webhook import WebhookEventSource
from cicerone.locks import LockLostError, WriterLockBusyError
from cicerone.serve import app as serve_app
from cicerone.serve import create_app
from cicerone.serve_schemas import ValidationErrorDetail
from cicerone.track.store import TrackStore


def _settings(tmp_path, **overrides):
    return make_settings(
        **{
            "mode": "serve",
            "serve_auth_token": "secret",
            "track": {"enabled": True},
            "output": IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
            **overrides,
        }
    )


def _impression(**overrides):
    payload = {
        "kind": "impression",
        "user_id": "u1",
        "item_id": "i1",
        "rank": 1,
        "occurred_at": "2026-08-28T12:00:00Z",
        "event_id": "imp-1",
    }
    payload.update(overrides)
    return payload


def test_post_track_single_and_batch(tmp_path):
    app = create_app(_settings(tmp_path), FakeReader(_recs_df()))
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    single = client.post("/track", headers=headers, json=_impression())
    assert single.status_code == 202
    body = single.json()
    assert body["accepted"] == 1
    assert body["event_ids"] == ["imp-1"]
    batch = client.post(
        "/track",
        headers=headers,
        json={"events": [_impression(item_id="i2", event_id="imp-2", rank=2)]},
    )
    assert batch.status_code == 202
    assert batch.json()["accepted"] == 1
    listed = client.post(
        "/track",
        headers=headers,
        json=[_impression(item_id="i3", event_id="imp-3", rank=3)],
    )
    assert listed.status_code == 202
    store = TrackStore(_settings(tmp_path).output)
    assert {row["event_id"] for row in store.read_rows()} == {"imp-1", "imp-2", "imp-3"}


def test_post_track_increments_prometheus(tmp_path):
    from prometheus_client import generate_latest
    from support.prometheus_metrics import metric_value

    from cicerone.metrics import record_track_ingest

    before = metric_value(
        generate_latest().decode(),
        "cicerone_track_ingest_total",
        {"kind": "impression", "status": "accepted"},
    )
    app = create_app(_settings(tmp_path), FakeReader(_recs_df()))
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    assert client.post("/track", headers=headers, json=_impression()).status_code == 202
    after = metric_value(
        generate_latest().decode(),
        "cicerone_track_ingest_total",
        {"kind": "impression", "status": "accepted"},
    )
    assert after == before + 1
    record_track_ingest(kind="click", status="accepted")
    assert (
        metric_value(
            generate_latest().decode(),
            "cicerone_track_ingest_total",
            {"kind": "click", "status": "accepted"},
        )
        >= 1
    )


def test_post_track_auth_and_validation(tmp_path):
    app = create_app(_settings(tmp_path), FakeReader(_recs_df()))
    client = TestClient(app)
    assert client.post("/track", json=_impression()).status_code == 401
    bad = client.post(
        "/track",
        headers={"Authorization": "Bearer secret"},
        json={"user_id": "u1"},
    )
    assert bad.status_code == 400
    detail = bad.json()["detail"]
    assert isinstance(detail, list)
    ValidationErrorDetail.model_validate({"detail": detail})


def test_post_track_counts_malformed_body_as_error_once(tmp_path):
    from support.prometheus_metrics import registry_metric_value

    labels = {"kind": "other", "status": "error"}
    before = registry_metric_value("cicerone_track_ingest_total", labels)
    app = create_app(_settings(tmp_path), FakeReader(_recs_df()))
    response = TestClient(app).post("/track", headers={"Authorization": "Bearer secret"}, json="nope")
    assert response.status_code == 400
    assert registry_metric_value("cicerone_track_ingest_total", labels) == before + 1


def test_post_track_normalization_error_returns_400(tmp_path):
    app = create_app(_settings(tmp_path), FakeReader(_recs_df()))
    response = TestClient(app).post(
        "/track",
        headers={"Authorization": "Bearer secret"},
        json=[_impression(occurred_at="2026-08-28T12:00:00")],
    )
    assert response.status_code == 400
    assert "occurred_at" in str(response.json()["detail"])


_TOO_LONG_FOR_MYSQL = [
    pytest.param({"event_id": "x" * 256}, "event_id", id="event_id"),
    pytest.param({"event_id": None, "idempotency_key": "x" * 256}, "event_id", id="idempotency_key"),
    pytest.param({"user_id": "x" * 65536}, "user_id", id="user_id"),
    pytest.param({"variant": "é" * 32768}, "variant", id="variant-multibyte"),
]
_TOO_LONG_VALUES = [pytest.param(param.values[0], id=param.id) for param in _TOO_LONG_FOR_MYSQL]


def _post_track(tmp_path, output, *events):
    app = create_app(_settings(tmp_path, output=output), FakeReader(_recs_df()))
    # Stdlib JSON, because httpx's encoder can't send a lone surrogate.
    response = TestClient(app).post(
        "/track",
        headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
        content=json.dumps(list(events)),
    )
    return app.state.track_store, response


@pytest.mark.parametrize(("overrides", "column"), _TOO_LONG_FOR_MYSQL)
def test_post_track_too_long_value_returns_400_on_mysql(tmp_path, mysql_track_output, overrides, column):
    store, response = _post_track(
        tmp_path, mysql_track_output, _impression(), _impression(**{"event_id": "too-long", **overrides})
    )
    assert response.status_code == 400
    assert column in response.json()["detail"]
    assert store.read_rows() == []


@pytest.mark.parametrize("overrides", _TOO_LONG_VALUES)
def test_post_track_too_long_value_is_accepted_on_other_backends(tmp_path, non_mysql_track_output, overrides):
    store, response = _post_track(
        tmp_path, non_mysql_track_output, _impression(), _impression(**{"event_id": "too-long", **overrides})
    )
    assert response.status_code == 202
    assert response.json()["accepted"] == 2
    assert len(store.read_rows()) == 2


def test_post_track_lone_surrogate_returns_400_on_mysql(tmp_path, mysql_track_output):
    store, response = _post_track(
        tmp_path, mysql_track_output, _impression(), _impression(event_id="bad", user_id="\ud800")
    )
    assert response.status_code == 400
    assert store.read_rows() == []


_BAD_DUPLICATES = [
    pytest.param({"user_id": "x" * 65536}, id="too-long"),
    pytest.param({"user_id": "\ud800"}, id="lone-surrogate"),
]


@pytest.mark.parametrize("overrides", _BAD_DUPLICATES)
def test_post_track_bad_duplicate_row_returns_400_on_mysql(tmp_path, mysql_track_output, overrides):
    store, response = _post_track(tmp_path, mysql_track_output, _impression(), _impression(**overrides))
    assert response.status_code == 400
    assert store.read_rows() == []


@pytest.mark.parametrize("overrides", _BAD_DUPLICATES)
def test_post_track_bad_duplicate_row_is_dropped_on_other_backends(
    tmp_path, non_mysql_track_output, overrides
):
    store, response = _post_track(tmp_path, non_mysql_track_output, _impression(), _impression(**overrides))
    assert response.status_code == 202
    assert response.json()["accepted"] == 1
    assert [row["user_id"] for row in store.read_rows()] == ["u1"]


def test_post_track_longest_values_are_accepted_on_every_backend(tmp_path, track_db_output):
    _store, response = _post_track(
        tmp_path, track_db_output, _impression(event_id="x" * 255, user_id="é" * 32767 + "x")
    )
    assert response.status_code == 202
    assert response.json()["accepted"] == 1


def _db_output(database_url):
    return IOSettings(kind="db", options={"database_url": database_url})


@pytest.mark.parametrize(
    ("database_url", "backend"),
    [("oracle+oracledb://u:p@host/db", "oracle"), ("mariadb+pymysql://u:p@host/db", "mariadb")],
)
def test_serve_refuses_to_start_on_an_unsupported_track_database(tmp_path, database_url, backend):
    with pytest.raises(ConfigError, match=f"mysql, postgresql, sqlite, not {backend}"):
        create_app(_settings(tmp_path, output=_db_output(database_url)), FakeReader(_recs_df()))


def test_track_store_for_reads_and_eval_is_still_built_on_an_unsupported_database():
    assert TrackStore(_db_output("oracle+oracledb://u:p@host/db")).backend is not None


def test_post_track_idempotent(tmp_path):
    app = create_app(_settings(tmp_path), FakeReader(_recs_df()))
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    from prometheus_client import generate_latest
    from support.prometheus_metrics import metric_value

    first = client.post("/track", headers=headers, json=_impression())
    after_first = metric_value(
        generate_latest().decode(),
        "cicerone_track_ingest_total",
        {"kind": "impression", "status": "accepted"},
    )
    second = client.post("/track", headers=headers, json=_impression())
    after_second = metric_value(
        generate_latest().decode(),
        "cicerone_track_ingest_total",
        {"kind": "impression", "status": "accepted"},
    )
    assert first.status_code == 202
    assert first.json()["accepted"] == 1
    assert first.json()["event_ids"] == ["imp-1"]
    assert second.json()["accepted"] == 0
    assert second.json()["event_ids"] == []
    assert after_second == after_first


def test_post_track_trims_supplied_event_ids(tmp_path):
    app = create_app(_settings(tmp_path), FakeReader(_recs_df()))
    client = TestClient(app)
    rows = [
        _impression(item_id="i1", rank=1, event_id=" abc "),
        _impression(item_id="i2", rank=2, event_id=" ", idempotency_key="k"),
        _impression(item_id="i3", rank=3, event_id=" "),
    ]
    response = client.post("/track", headers={"Authorization": "Bearer secret"}, json=rows)
    assert response.status_code == 202
    trimmed, keyed, generated = response.json()["event_ids"]
    assert (trimmed, keyed) == ("abc", "k")
    assert generated.strip()
    assert generated not in {"abc", "k"}
    stored = {row["item_id"]: row["event_id"] for row in TrackStore(_settings(tmp_path).output).read_rows()}
    assert stored == {"i1": "abc", "i2": "k", "i3": generated}


def test_post_track_counts_accepted_kinds_in_mixed_partial_batch(tmp_path):
    from prometheus_client import generate_latest
    from support.prometheus_metrics import metric_value

    app = create_app(_settings(tmp_path), FakeReader(_recs_df()))
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    client.post("/track", headers=headers, json=_impression())
    before_imp = metric_value(
        generate_latest().decode(),
        "cicerone_track_ingest_total",
        {"kind": "impression", "status": "accepted"},
    )
    before_click = metric_value(
        generate_latest().decode(),
        "cicerone_track_ingest_total",
        {"kind": "click", "status": "accepted"},
    )
    before_other = metric_value(
        generate_latest().decode(),
        "cicerone_track_ingest_total",
        {"kind": "other", "status": "accepted"},
    )
    mixed = client.post(
        "/track",
        headers=headers,
        json=[
            _impression(),
            _impression(kind="click", event_id="clk-1", rank=None),
        ],
    )
    assert mixed.status_code == 202
    assert mixed.json()["accepted"] == 1
    assert mixed.json()["event_ids"] == ["clk-1"]
    after_imp = metric_value(
        generate_latest().decode(),
        "cicerone_track_ingest_total",
        {"kind": "impression", "status": "accepted"},
    )
    after_click = metric_value(
        generate_latest().decode(),
        "cicerone_track_ingest_total",
        {"kind": "click", "status": "accepted"},
    )
    after_other = metric_value(
        generate_latest().decode(),
        "cicerone_track_ingest_total",
        {"kind": "other", "status": "accepted"},
    )
    assert after_imp == before_imp
    assert after_click == before_click + 1
    assert after_other == before_other


def test_post_track_records_error_when_append_fails(tmp_path, monkeypatch):
    from prometheus_client import generate_latest
    from support.prometheus_metrics import metric_value

    monkeypatch.setattr(
        "cicerone.track.store.TrackStore.append_accepted_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("store")),
    )
    before = metric_value(
        generate_latest().decode(),
        "cicerone_track_ingest_total",
        {"kind": "other", "status": "error"},
    )
    app = create_app(_settings(tmp_path), FakeReader(_recs_df()))
    response = TestClient(app, raise_server_exceptions=False).post(
        "/track",
        headers={"Authorization": "Bearer secret"},
        json=_impression(),
    )
    after = metric_value(
        generate_latest().decode(),
        "cicerone_track_ingest_total",
        {"kind": "other", "status": "error"},
    )
    assert response.status_code == 500
    assert after == before + 1


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (WriterLockBusyError("dataset writer lock busy"), "Writer lock is busy"),
        (LockLostError("dataset writer lock lost before write", kind="writer"), "Writer lock was lost"),
        (OSError("disk full"), "Track store unavailable"),
        (OperationalError("INSERT", {}, Exception("db down")), "Track store unavailable"),
    ],
)
def test_post_track_returns_503_when_store_unavailable(tmp_path, monkeypatch, error, detail):
    from prometheus_client import generate_latest
    from support.prometheus_metrics import metric_value

    monkeypatch.setattr("cicerone.track.store.TrackStore.append_accepted_rows", raising(error))
    labels = {"kind": "other", "status": "error"}
    before = metric_value(generate_latest().decode(), "cicerone_track_ingest_total", labels)
    app = create_app(_settings(tmp_path), FakeReader(_recs_df()))
    response = TestClient(app, raise_server_exceptions=False).post(
        "/track",
        headers={"Authorization": "Bearer secret"},
        json=_impression(),
    )
    assert response.status_code == 503
    assert response.json()["detail"] == detail
    assert metric_value(generate_latest().decode(), "cicerone_track_ingest_total", labels) == before + 1


def test_post_track_append_does_not_block_event_loop(tmp_path, monkeypatch):
    import asyncio

    import httpx

    entered = threading.Event()
    release = threading.Event()

    def _blocking_append(self, rows):
        entered.set()
        assert release.wait(WAIT_SECONDS)
        return [dict(row) for row in rows]

    monkeypatch.setattr("cicerone.track.store.TrackStore.append_accepted_rows", _blocking_append)
    app = create_app(_settings(tmp_path), FakeReader(_recs_df()))

    @app.get("/__probe")
    async def _probe() -> dict[str, bool]:
        return {"ok": True}

    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": "Bearer secret"}
            track = asyncio.create_task(client.post("/track", headers=headers, json=_impression()))
            assert await asyncio.to_thread(entered.wait, WAIT_SECONDS)
            probe = await client.get("/__probe")
            assert probe.status_code == 200
            assert not track.done()
            release.set()
            track_response = await track
            assert track_response.status_code == 202

    asyncio.run(_run())


def test_track_route_absent_when_disabled(tmp_path):
    app = create_app(
        make_settings(
            mode="serve",
            serve_auth_token="secret",
            output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)}),
        ),
        FakeReader(_recs_df()),
    )
    client = TestClient(app)
    denied = client.post("/track", headers={"Authorization": "Bearer secret"}, json=_impression())
    assert denied.status_code == 404


def test_track_rows_never_enter_event_source(tmp_path):
    source = WebhookEventSource({})
    app = create_app(
        _settings(tmp_path, events=EventsSettings(enabled=True, kind="webhook")),
        FakeReader(_recs_df()),
        event_source=source,
    )
    response = TestClient(app).post(
        "/track",
        headers={"Authorization": "Bearer secret"},
        json=_impression(),
    )
    assert response.status_code == 202
    assert source.health().lag == 0
    assert len(TrackStore(_settings(tmp_path).output).read_rows()) == 1


def test_track_openapi_documents_structured_validation_400(tmp_path):
    app = create_app(_settings(tmp_path), FakeReader(_recs_df()))
    schema = TestClient(app).get("/openapi.json").json()
    assert "TrackEvent" in schema["components"]["schemas"]
    assert "TrackIngestRequest" in schema["components"]["schemas"]
    request_body = schema["paths"]["/track"]["post"]["requestBody"]
    body_schema = request_body["content"]["application/json"]["schema"]
    assert {"$ref": "#/components/schemas/TrackEvent"} in body_schema["oneOf"]
    track_400 = schema["paths"]["/track"]["post"]["responses"]["400"]
    content_schema = track_400["content"]["application/json"]["schema"]
    refs = {item.get("$ref") for item in content_schema.get("anyOf", [])}
    assert "#/components/schemas/ErrorDetail" in refs
    assert "#/components/schemas/ValidationErrorDetail" in refs
    track_413 = schema["paths"]["/track"]["post"]["responses"]["413"]
    assert track_413["content"]["application/json"]["schema"]["$ref"] == ("#/components/schemas/ErrorDetail")
    again = TestClient(app).get("/openapi.json").json()
    assert again["info"]["title"]


def test_post_track_non_json_payload(tmp_path):
    app = create_app(_settings(tmp_path), FakeReader(_recs_df()))
    resp = TestClient(app).post(
        "/track",
        headers={"Authorization": "Bearer secret", "Content-Type": "text/plain"},
        content="not json at all",
    )
    assert resp.status_code == 400
    assert resp.json() == {"detail": "Request body must be JSON"}


def test_post_track_rejects_oversized_body(tmp_path):
    app = create_app(
        _settings(tmp_path, events=EventsSettings(options={"max_body_bytes": 64})),
        FakeReader(_recs_df()),
    )
    response = TestClient(app).post(
        "/track",
        headers={"Authorization": "Bearer secret", "content-type": "application/json"},
        content=b'{"kind":"impression","user_id":"' + b"u" * 200 + b'","item_id":"i1","rank":1}',
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}


def test_post_track_rejects_deeply_nested_json(tmp_path):
    app = create_app(_settings(tmp_path), FakeReader(_recs_df()))
    nested = b"[" * 3000 + b"0" + b"]" * 3000
    response = TestClient(app).post(
        "/track",
        headers={"Authorization": "Bearer secret", "content-type": "application/json"},
        content=nested,
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Request body must be JSON"}


def test_post_track_rejects_oversized_content_length(tmp_path):
    app = create_app(
        _settings(tmp_path, events=EventsSettings(options={"max_body_bytes": 64})),
        FakeReader(_recs_df()),
    )
    response = TestClient(app).post(
        "/track",
        headers={
            "Authorization": "Bearer secret",
            "content-type": "application/json",
            "content-length": "9999",
        },
        content=b"{}",
    )
    assert response.status_code == 413


def test_get_does_not_log_impressions_by_default(tmp_path):
    app = create_app(_settings(tmp_path), FakeReader(_recs_df()))
    TestClient(app).get("/recommendations/u1", headers={"Authorization": "Bearer secret"})
    assert TrackStore(_settings(tmp_path).output).read_rows() == []


def test_get_log_impressions_when_enabled(tmp_path):
    app = create_app(_settings(tmp_path, serve_log_impressions=True), FakeReader(_recs_df()))
    with TestClient(app) as client:
        body = client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"}).json()
    rows = TrackStore(_settings(tmp_path).output).read_rows()
    assert {row["kind"] for row in rows} == {"impression"}
    assert {row["item_id"] for row in rows} == {item["item_id"] for item in body["items"]}
    assert all(row["user_id"] == "u1" for row in rows)
    assert [row["rank"] for row in rows] == [item["rank"] for item in body["items"]]


def test_get_log_impressions_retries_writer_lock_busy(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        del args, kwargs
        calls["n"] += 1
        if calls["n"] < 3:
            raise WriterLockBusyError("dataset writer lock busy")
        return 1

    monkeypatch.setattr("cicerone.track.store.TrackStore.append_rows", flaky)
    patch_module_ref(monkeypatch, "cicerone.serve.app.time", sleep=lambda _s: None)
    app = create_app(_settings(tmp_path, serve_log_impressions=True), FakeReader(_recs_df()))
    with TestClient(app) as client:
        response = client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert calls["n"] == 3


def test_get_log_impressions_swallows_store_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cicerone.track.store.TrackStore.append_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("store")),
    )
    app = create_app(_settings(tmp_path, serve_log_impressions=True), FakeReader(_recs_df()))
    response = TestClient(app).get("/recommendations/u1", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert response.json()["items"]


def test_get_log_impressions_append_runs_on_dedicated_thread_and_drains(tmp_path, monkeypatch):
    release = threading.Event()
    appended: list[str] = []

    def slow_append(*args, **kwargs):
        del args, kwargs
        release.wait(timeout=5)
        appended.append(threading.current_thread().name)
        return 1

    monkeypatch.setattr("cicerone.track.store.TrackStore.append_rows", slow_append)
    app = create_app(_settings(tmp_path, serve_log_impressions=True), FakeReader(_recs_df()))
    with TestClient(app) as client:
        response = client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"})
        assert response.status_code == 200
        assert appended == []
        threading.Timer(0.1, release.set).start()
    assert len(appended) == 1
    assert appended[0].startswith("cicerone-serve-append")


def test_get_log_impressions_drops_the_append_when_the_queue_is_full(tmp_path, monkeypatch, caplog):
    release = threading.Event()
    appended: list[int] = []

    def slow_append(*args, **kwargs):
        del args, kwargs
        release.wait(timeout=5)
        appended.append(1)
        return 1

    slot_freed = threading.Event()

    class _RecordingSlots(threading.BoundedSemaphore):
        def release(self, n: int = 1) -> None:
            super().release(n)
            slot_freed.set()

    monkeypatch.setattr(
        serve_app, "threading", SimpleNamespace(**{**vars(threading), "BoundedSemaphore": _RecordingSlots})
    )
    monkeypatch.setattr("cicerone.serve.app._MAX_PENDING_APPENDS", 1)
    monkeypatch.setattr("cicerone.track.store.TrackStore.append_rows", slow_append)
    dropped_metric = ("cicerone_serve_appends_dropped_total", {"kind": "impression"})
    dropped = registry_metric_value(*dropped_metric)
    app = create_app(_settings(tmp_path, serve_log_impressions=True), FakeReader(_recs_df()))
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        assert client.get("/recommendations/u1", headers=headers).status_code == 200
        with caplog.at_level(logging.WARNING, logger="cicerone.serve.app"):
            assert client.get("/recommendations/u1", headers=headers).status_code == 200
        release.set()
        assert slot_freed.wait(timeout=5)
        assert client.get("/recommendations/u1", headers=headers).status_code == 200
    assert appended == [1, 1]
    assert registry_metric_value(*dropped_metric) == dropped + 1
    assert "Dropped impression append for user_id='u1': append queue is full" in caplog.text


def test_attach_track_ingest_openapi_noop_without_post():
    from cicerone.serve.events_routes import attach_ingest_openapi
    from cicerone.track.routes import TRACK_INGEST

    schema: dict = {"paths": {}}
    attach_ingest_openapi(schema, TRACK_INGEST)
    assert schema == {"paths": {}}
