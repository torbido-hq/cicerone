from __future__ import annotations

from dataclasses import dataclass

from conftest import make_settings
from fastapi.testclient import TestClient

from cicerone.config import EventsSettings, IOSettings
from cicerone.dashboard import ROBOTS_TAG, create_app
from cicerone.dashboard_config import MISSING, REDACTED, _normalize, config_display


class _FakeReader:
    def read_latest(self) -> dict | None:
        return None

    def read_recent(self, limit: int) -> list[dict]:
        return []


def _users_with(username: str, password: str) -> dict[str, str]:
    import bcrypt

    return {username: bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")}


def _section(display: dict, section_id: str) -> dict:
    return next(item for item in display["sections"] if item["id"] == section_id)


def _secret_settings(**overrides):
    return make_settings(
        dashboard_enabled=True,
        serve_auth_token="super-secret-serve",
        trigger_auth_token="super-secret-trigger",
        trigger_postgres_url="postgresql://lock:hunter2@localhost/locks",
        input=IOSettings(
            kind="dataset",
            options={
                "storage_backend": "s3",
                "access_key_id": "AKIATEST",
                "secret_access_key": "s3secretVALUE",
                "api_key": "options-api-key",
                "bucket": "recs",
                "endpoint_url": "https://minio.example:9000",
                "webhook": "https://hook:leaked@example.com/hook",
            },
        ),
        output=IOSettings(
            kind="db",
            options={"database_url": "postgresql://user:hunter2@localhost/cicerone"},
        ),
        events=EventsSettings(
            enabled=True,
            kind="webhook",
            options={
                "auth_token": "event-secret",
                "queue_url": "https://sqs.example/123/cicerone",
            },
        ),
        **overrides,
    )


def test_config_display_redacts_secrets_and_keeps_safe_values():
    display = config_display(
        _secret_settings(top_k=25, cron_schedule="0 4 * * *"),
        config_path="/app/config/cicerone.dashboard.toml",
        usernames=("alice", "bob"),
    )
    meta = _section(display, "meta")
    job = _section(display, "job")
    serve = _section(display, "serve")
    trigger = _section(display, "trigger")
    incoming = _section(display, "input")
    outgoing = _section(display, "output")
    events = _section(display, "events")
    dashboard = _section(display, "dashboard")

    assert meta["fields"]["config_path"] == "/app/config/cicerone.dashboard.toml"
    assert meta["fields"]["mode"] == "batch"
    assert job["fields"]["top_k"] == 25
    assert job["fields"]["cron_schedule"] == "0 4 * * *"
    assert serve["fields"]["auth_token"] == REDACTED
    assert serve["fields"]["metrics_token"] == MISSING
    assert trigger["fields"]["auth_token"] == REDACTED
    assert trigger["fields"]["postgres_url"] == REDACTED
    assert incoming["fields"]["kind"] == "dataset"
    assert incoming["fields"]["options"]["bucket"] == "recs"
    assert incoming["fields"]["options"]["access_key_id"] == "AKIATEST"
    assert incoming["fields"]["options"]["secret_access_key"] == REDACTED
    assert incoming["fields"]["options"]["api_key"] == REDACTED
    assert incoming["fields"]["options"]["endpoint_url"] == REDACTED
    assert incoming["fields"]["options"]["webhook"] == REDACTED
    assert outgoing["fields"]["kind"] == "db"
    assert outgoing["fields"]["options"]["database_url"] == REDACTED
    assert events["fields"]["options"]["auth_token"] == REDACTED
    assert events["fields"]["options"]["queue_url"] == REDACTED
    assert isinstance(job["fields"]["automl"], dict)
    assert "enabled" in job["fields"]["automl"]
    assert isinstance(job["fields"]["explain"], dict)
    assert isinstance(_section(display, "experiment")["fields"], dict)
    assert dashboard["fields"]["users"] == ["alice", "bob"]
    assert events["badge"] == "on"
    assert events["kind"] == "webhook"
    assert incoming["kind"] == "dataset"
    assert incoming["toml"] == "[input]"
    assert "split" in display["split_note"]
    assert display["hints"]["job.top_k"]["text"]
    assert display["hints"]["job.top_k"]["docs"].startswith("https://cicerone.dev/")


def test_config_display_missing_feature_file(tmp_path):
    settings = make_settings(feature_config_path=str(tmp_path / "missing.toml"))
    features = _section(config_display(settings), "features")
    assert features["fields"] is None
    assert "No feature config file" in features["message"]


def test_config_display_unreadable_feature_file(tmp_path):
    path = tmp_path / "features.toml"
    path.write_text("not = valid = toml {{")
    settings = make_settings(feature_config_path=str(path))
    features = _section(config_display(settings), "features")
    assert features["fields"] is None
    assert features["message"] == "Feature config could not be loaded."


def test_config_display_loads_feature_file(tmp_path):
    path = tmp_path / "features.toml"
    path.write_text("[event_weights]\npurchase = 4.0\n")
    settings = make_settings(feature_config_path=str(path))
    features = _section(config_display(settings), "features")
    assert features["message"] is None
    assert features["fields"]["event_weights"]["purchase"] == 4.0


def test_normalize_dataclass_instance():
    @dataclass
    class _Probe:
        enabled: bool = True

    assert _normalize(_Probe()) == {"enabled": "true"}


def test_config_display_empty_model_configs_wraps_missing():
    models = _section(config_display(make_settings(model_configs={})), "model_configs")
    assert models["fields"] == {"value": MISSING}


def test_config_page_requires_auth():
    app = create_app(make_settings(dashboard_enabled=True), _FakeReader(), _users_with("alice", "s3cret"))
    response = TestClient(app).get("/dashboard/config")
    assert response.status_code == 401


def test_config_page_renders_redacted_html(tmp_path):
    users = _users_with("alice", "s3cret")
    hash_value = users["alice"]
    features = tmp_path / "features.toml"
    features.write_text("[event_weights]\npurchase = 4.0\n")
    app = create_app(
        _secret_settings(top_k=25, feature_config_path=str(features)),
        _FakeReader(),
        users,
        config_path="/app/config/cicerone.dashboard.toml",
    )
    response = TestClient(app).get("/dashboard/config", auth=("alice", "s3cret"))

    assert response.status_code == 200
    html = response.text
    assert "<title>Cicerone — Configuration</title>" in html
    assert 'href="/dashboard/config"' in html
    assert 'data-config-section="job"' in html
    assert ">top_k<" in html
    assert "25" in html
    assert "cron_schedule" in html
    assert "dataset" in html
    assert "AKIATEST" in html
    assert "recs" in html
    assert "[redacted]" in html
    assert "super-secret-serve" not in html
    assert "super-secret-trigger" not in html
    assert "s3secretVALUE" not in html
    assert "options-api-key" not in html
    assert "hunter2" not in html
    assert "event-secret" not in html
    assert "minio.example" not in html
    assert "sqs.example" not in html
    assert "leaked" not in html
    assert hash_value not in html
    assert ">alice<" in html or "alice" in html
    assert "purchase" in html
    assert "This is the config this dashboard process loaded" in html
    assert ">Metrics_token<" not in html
    assert ">metrics_token<" in html
    assert 'data-config-jump="job"' in html
    assert 'href="#config-job"' in html
    assert 'id="config-job"' in html
    assert 'aria-current="page"' in html
    assert 'aria-label="Config sections"' in html
    assert 'aria-label="Main"' in html
    assert 'name="robots"' in html
    assert f'content="{ROBOTS_TAG}"' in html
    assert "noindex" in html
    assert "[job]" in html
    assert 'data-config-hint-button="job.top_k"' in html
    assert 'aria-label="About top_k"' in html
    assert "How many items the job writes for each user." in html
    assert "https://cicerone.dev/how-it-works/" in html
    assert 'popovertarget="config-hint-job-top_k"' in html


def test_config_page_unset_token_is_em_dash(tmp_path):
    app = create_app(
        make_settings(
            dashboard_enabled=True,
            feature_config_path=str(tmp_path / "missing.toml"),
        ),
        _FakeReader(),
        _users_with("alice", "s3cret"),
    )
    html = TestClient(app).get("/dashboard/config", auth=("alice", "s3cret")).text
    assert "super-secret" not in html
    assert MISSING in html
    assert "No feature config file" in html
