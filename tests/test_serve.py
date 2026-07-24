from __future__ import annotations

import threading

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cicerone.config import IOSettings, Settings
from cicerone.serve import _start_refresh_loop, create_app, main


def _settings(**overrides) -> Settings:
    base = dict(
        input=IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/in"}),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/out"}),
        feature_config_path="/app/config/features.toml",
        top_k=10,
        half_life_days=90,
        cron_schedule="0 3 * * *",
        models=None,
        model_weights=None,
        rrf_k=None,
        automl_enabled=False,
        automl_n_splits=2,
        automl_test_days=14,
        automl_primary_metric="MAP",
        automl_candidates=None,
        mode="serve",
        serve_host="0.0.0.0",
        serve_port=8000,
        serve_auth_token="secret",
        serve_default_k=10,
        serve_refresh_interval_seconds=60,
        trigger_enabled=False,
        trigger_host="0.0.0.0",
        trigger_port=8080,
        trigger_auth_token=None,
        trigger_debounce_seconds=60,
        trigger_poll_input_bucket=False,
        trigger_poll_interval_seconds=300,
    )
    base.update(overrides)
    return Settings(**base)


class _FakeReader:
    def __init__(self, recs: pd.DataFrame):
        self._recs = recs
        self.refresh_calls = 0

    def refresh(self) -> None:
        self.refresh_calls += 1

    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame:
        rows = self._recs[self._recs["user_id"] == user_id].sort_values("rank")
        return rows.head(k).reset_index(drop=True)


def _recs_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"},
            {"user_id": "u1", "item_id": "i2", "rank": 2, "score": 0.5, "source": "personalized"},
        ]
    )


def test_health_requires_no_auth():
    app = create_app(_settings(), _FakeReader(_recs_df()))
    client = TestClient(app)

    assert client.get("/health").status_code == 200


def test_recommendations_requires_auth():
    app = create_app(_settings(), _FakeReader(_recs_df()))
    client = TestClient(app)

    response = client.get("/recommendations/u1")

    assert response.status_code == 401


def test_recommendations_rejects_wrong_token():
    app = create_app(_settings(), _FakeReader(_recs_df()))
    client = TestClient(app)

    response = client.get("/recommendations/u1", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401


def test_recommendations_returns_records_with_valid_token():
    app = create_app(_settings(), _FakeReader(_recs_df()))
    client = TestClient(app)

    response = client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    body = response.json()
    assert [row["item_id"] for row in body] == ["i1", "i2"]


def test_recommendations_respects_k_query_param():
    app = create_app(_settings(), _FakeReader(_recs_df()))
    client = TestClient(app)

    response = client.get("/recommendations/u1?k=1", headers={"Authorization": "Bearer secret"})

    assert len(response.json()) == 1


def test_recommendations_unknown_user_returns_404():
    app = create_app(_settings(), _FakeReader(_recs_df()))
    client = TestClient(app)

    response = client.get("/recommendations/nobody", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 404


def test_recommendations_without_auth_token_configured_is_open():
    app = create_app(_settings(serve_auth_token=None), _FakeReader(_recs_df()))
    client = TestClient(app)

    response = client.get("/recommendations/u1")

    assert response.status_code == 200


def test_start_refresh_loop_calls_refresh_periodically(monkeypatch):
    reader = _FakeReader(_recs_df())
    calls = {"sleep": 0}

    def fake_sleep(_seconds):
        calls["sleep"] += 1
        if calls["sleep"] >= 3:
            raise SystemExit("stop after three ticks")

    monkeypatch.setattr("cicerone.serve.time.sleep", fake_sleep)
    monkeypatch.setattr(threading.Thread, "start", lambda self: self.run())

    with pytest.raises(SystemExit):
        _start_refresh_loop(reader, interval_seconds=0.01)

    assert reader.refresh_calls >= 2


def test_main_requires_serve_mode(tmp_path, monkeypatch):
    config_path = tmp_path / "cicerone.toml"
    config_path.write_text(
        f"""
        [job]
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "{tmp_path}"
        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "{tmp_path}"
        """
    )
    monkeypatch.setenv("CICERONE_CONFIG_PATH", str(config_path))

    with pytest.raises(RuntimeError, match="requires mode = 'serve'"):
        main()


def test_main_starts_serve_app_in_serve_mode(tmp_path, monkeypatch):
    config_path = tmp_path / "cicerone.toml"
    config_path.write_text(
        f"""
        [job]
        mode = "serve"

        [serve]
        auth_token = "secret"

        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "{tmp_path}"
        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "{tmp_path}"
        """
    )
    monkeypatch.setenv("CICERONE_CONFIG_PATH", str(config_path))

    refresh_calls = []
    uvicorn_calls = {}

    import cicerone.serve as serve_module

    def fake_start_refresh_loop(reader, interval):
        refresh_calls.append(reader)

    def fake_uvicorn_run(app, host, port):
        uvicorn_calls.update(host=host, port=port)

    monkeypatch.setattr(serve_module, "_start_refresh_loop", fake_start_refresh_loop)
    monkeypatch.setattr(serve_module, "uvicorn", type("_U", (), {"run": staticmethod(fake_uvicorn_run)}))

    main()

    assert len(refresh_calls) == 1
    assert uvicorn_calls == {"host": "0.0.0.0", "port": 8000}
