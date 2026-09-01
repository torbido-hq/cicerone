from __future__ import annotations

import logging

import pytest
from support.toml_config import write_toml

from cicerone.config import ConfigError, EventsSettings, load_settings, make_settings
from cicerone.config.events import coerce_events_settings, load_events_settings


def test_make_settings_events_defaults():
    settings = make_settings()
    assert settings.events.enabled is False
    assert settings.events.kind == "webhook"
    assert settings.events.incremental.batch_size == 100
    assert settings.events.incremental.poll_interval_seconds == 1.0
    assert settings.events.online.enabled is False
    assert settings.events.online.fit_partial_epochs == 1
    assert settings.events.online.fit_min_events == 100
    assert settings.events.online.max_extra_interactions == 50_000
    assert settings.events_enabled is False
    assert settings.events_kind == "webhook"


def test_coerce_events_settings_errors():
    with pytest.raises(TypeError):
        coerce_events_settings("bad")
    with pytest.raises(TypeError):
        coerce_events_settings({"incremental": "bad"})
    with pytest.raises(ValueError):
        coerce_events_settings({"incremental": {"batch_size": 0}})
    with pytest.raises(TypeError):
        coerce_events_settings({"online": "bad"})
    with pytest.raises(ValueError):
        coerce_events_settings({"online": {"fit_partial_epochs": -1}})
    with pytest.raises(ValueError):
        coerce_events_settings({"online": {"fit_min_events": 0}})
    with pytest.raises(ValueError):
        coerce_events_settings({"online": {"max_extra_interactions": 0}})
    with pytest.raises(ConfigError, match="events.online.enabled requires events.enabled"):
        coerce_events_settings({"enabled": False, "online": {"enabled": True}})


def test_load_events_section(tmp_path):
    path = write_toml(
        tmp_path,
        """
        [job]
        mode = "serve"
        [serve]
        auth_token = "tok"
        [events]
        enabled = true
        kind = "webhook"
        [events.options]
        auth_token = "events-tok"
        [events.incremental]
        batch_size = 5
        batch_window_seconds = 12
        poll_interval_seconds = 0.5
        [events.online]
        enabled = true
        fit_partial_epochs = 2
        fit_min_events = 7
        max_extra_interactions = 12
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )
    settings = load_settings(path)
    assert settings.events.enabled is True
    assert settings.events.options["auth_token"] == "events-tok"
    assert settings.events.incremental.batch_size == 5
    assert settings.events.incremental.batch_window_seconds == 12.0
    assert settings.events.incremental.poll_interval_seconds == 0.5
    assert settings.events.online.enabled is True
    assert settings.events.online.fit_partial_epochs == 2
    assert settings.events.online.fit_min_events == 7
    assert settings.events.online.max_extra_interactions == 12


def test_load_online_and_experiment_warns(tmp_path, caplog):
    path = write_toml(
        tmp_path,
        """
        [job]
        mode = "serve"
        [serve]
        auth_token = "tok"
        [events]
        enabled = true
        kind = "webhook"
        [events.online]
        enabled = true
        [experiment]
        enabled = true
        id = "exp"
        [[experiment.variants]]
        name = "control"
        traffic = 0.5
        [[experiment.variants]]
        name = "treatment"
        traffic = 0.5
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )
    with caplog.at_level(logging.WARNING, logger="cicerone.config.load"):
        settings = load_settings(path)
    assert settings.events.online.enabled is True
    assert settings.experiment.enabled is True
    assert any("online collaborative refresh will be skipped" in record.message for record in caplog.records)


def test_make_settings_online_sequential_without_torch_warns(monkeypatch, caplog):
    from cicerone.config import EventsOnlineSettings

    monkeypatch.setattr("cicerone.model_config.sequential_extra_available", lambda: False)
    with caplog.at_level(logging.WARNING, logger="cicerone.config.load"):
        make_settings(
            models=["collaborative", "sequential"],
            events=EventsSettings(
                enabled=True,
                kind="webhook",
                online=EventsOnlineSettings(enabled=True),
            ),
        )
    assert any("torch extra is not installed" in record.message for record in caplog.records)


def test_load_online_rejects_s3_output(tmp_path):
    path = write_toml(
        tmp_path,
        """
        [job]
        mode = "serve"
        [serve]
        auth_token = "tok"
        [events]
        enabled = true
        kind = "webhook"
        [events.online]
        enabled = true
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "s3"
        bucket = "recs"
        access_key_id = "id"
        secret_access_key = "secret"
        """,
    )
    with pytest.raises(ConfigError, match="compare-and-swap"):
        load_settings(path)


def test_load_events_unknown_kind(tmp_path):
    path = write_toml(
        tmp_path,
        """
        [job]
        [events]
        enabled = true
        kind = "kafka"
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )
    with pytest.raises(ConfigError, match="events.kind"):
        load_settings(path)


def test_load_events_db_requires_database_url():
    with pytest.raises(ConfigError, match="database_url"):
        load_events_settings(
            {"enabled": True, "kind": "db", "options": {}},
            mode="serve",
            serve_auth_token="tok",
            resolve_env=lambda value, _path: value,
        )


def test_load_events_db_ok():
    settings = load_events_settings(
        {"enabled": True, "kind": "db", "options": {"database_url": "sqlite+pysqlite:///:memory:"}},
        mode="serve",
        serve_auth_token="tok",
        resolve_env=lambda value, _path: value,
    )
    assert settings.kind == "db"
    assert settings.options["database_url"].startswith("sqlite")


def test_load_events_s3_requires_bucket():
    with pytest.raises(ConfigError, match="bucket"):
        load_events_settings(
            {
                "enabled": True,
                "kind": "s3",
                "options": {"access_key_id": "a", "secret_access_key": "b"},
            },
            mode="serve",
            serve_auth_token="tok",
            resolve_env=lambda value, _path: value,
        )


def test_load_events_s3_sqs_requires_queue_url():
    with pytest.raises(ConfigError, match="queue_url"):
        load_events_settings(
            {
                "enabled": True,
                "kind": "s3",
                "options": {
                    "access_key_id": "a",
                    "secret_access_key": "b",
                    "bucket": "events",
                    "mode": "sqs",
                },
            },
            mode="serve",
            serve_auth_token="tok",
            resolve_env=lambda value, _path: value,
        )


def test_load_events_s3_list_ok():
    settings = load_events_settings(
        {
            "enabled": True,
            "kind": "s3",
            "options": {
                "access_key_id": "a",
                "secret_access_key": "b",
                "bucket": "events",
                "mode": "list",
                "endpoint_url": "https://abc.r2.cloudflarestorage.com",
            },
        },
        mode="serve",
        serve_auth_token="tok",
        resolve_env=lambda value, _path: value,
    )
    assert settings.kind == "s3"


def test_load_events_s3_sqs_rejects_endpoint_url():
    with pytest.raises(ConfigError, match="AWS-only"):
        load_events_settings(
            {
                "enabled": True,
                "kind": "s3",
                "options": {
                    "access_key_id": "a",
                    "secret_access_key": "b",
                    "bucket": "events",
                    "mode": "sqs",
                    "queue_url": "https://sqs.example/q",
                    "endpoint_url": "https://abc.r2.cloudflarestorage.com",
                },
            },
            mode="serve",
            serve_auth_token="tok",
            resolve_env=lambda value, _path: value,
        )


def test_load_events_redis_streams_requires_stream():
    with pytest.raises(ConfigError, match="stream"):
        load_events_settings(
            {
                "enabled": True,
                "kind": "redis_streams",
                "options": {"redis_url": "redis://localhost:6379/0", "consumer_group": "cicerone"},
            },
            mode="serve",
            serve_auth_token="tok",
            resolve_env=lambda value, _path: value,
        )


def test_load_events_redis_streams_ok():
    settings = load_events_settings(
        {
            "enabled": True,
            "kind": "redis_streams",
            "options": {
                "redis_url": "redis://localhost:6379/0",
                "stream": "cicerone:events",
                "consumer_group": "cicerone",
            },
        },
        mode="serve",
        serve_auth_token="tok",
        resolve_env=lambda value, _path: value,
    )
    assert settings.kind == "redis_streams"


def test_load_events_online_must_be_table():
    with pytest.raises(ConfigError, match="events.online must be a table"):
        load_events_settings(
            {"enabled": True, "kind": "webhook", "online": "nope"},
            mode="batch",
            serve_auth_token="tok",
            resolve_env=lambda value, _path: value,
        )


def test_load_events_online_requires_events_enabled():
    with pytest.raises(ConfigError, match="events.online.enabled requires events.enabled"):
        load_events_settings(
            {"enabled": False, "online": {"enabled": True}},
            mode="batch",
            serve_auth_token=None,
            resolve_env=lambda value, _path: value,
        )


def test_load_events_incremental_must_be_table():
    with pytest.raises(ConfigError, match="events.incremental must be a table"):
        load_events_settings(
            {"enabled": False, "incremental": "nope"},
            mode="batch",
            serve_auth_token=None,
            resolve_env=lambda value, _path: value,
        )


def test_load_events_disabled_allows_unknown_kind():
    settings = load_events_settings(
        {"enabled": False, "kind": "kafka"},
        mode="batch",
        serve_auth_token=None,
        resolve_env=lambda value, _path: value,
    )
    assert settings.enabled is False
    assert settings.kind == "kafka"


def test_coerce_pass_through_events_settings():
    original = EventsSettings(enabled=True, kind="webhook")
    coerced = coerce_events_settings(original)
    assert coerced.enabled is True
    assert coerced.incremental.batch_size == 100
    assert coerced.ha is False
    assert coerced.online.enabled is False
    ha = coerce_events_settings(EventsSettings(enabled=True, kind="webhook", ha=True))
    assert ha.ha is True
