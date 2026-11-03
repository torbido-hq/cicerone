from __future__ import annotations

import logging

import pytest
from support.toml_config import MINIMAL_IO_TOML, write_toml

from cicerone.config import (
    ConfigError,
    EventsIncrementalSettings,
    EventsOnlineSettings,
    EventsSettings,
    IOSettings,
    TriggerSettings,
    load_settings,
    make_settings,
)
from cicerone.config.events import load_events_settings


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
    assert settings.events.enabled is False
    assert settings.events.kind == "webhook"


@pytest.mark.parametrize(
    ("events", "error"),
    [
        ("bad", TypeError),
        ({"incremental": {"batch_size": 0}}, ValueError),
        ({"online": {"fit_partial_epochs": -1}}, ValueError),
        ({"online": {"fit_min_events": 0}}, ValueError),
        ({"online": {"max_extra_interactions": 0}}, ValueError),
        (EventsSettings(incremental=EventsIncrementalSettings(batch_size=0)), ValueError),
    ],
)
def test_make_settings_events_errors(events, error):
    with pytest.raises(error):
        make_settings(events=events)


@pytest.mark.parametrize(
    ("events", "match"),
    [
        (EventsSettings(enabled=True, kind="carrier-pigeon"), "events.kind"),
        (EventsSettings(enabled=True, kind="db"), "database_url"),
        (
            EventsSettings(online=EventsOnlineSettings(enabled=True)),
            "events.online.enabled requires events.enabled",
        ),
        (EventsSettings(enabled=True, online=EventsOnlineSettings(enabled=True)), "output.artifact_hmac_key"),
    ],
)
def test_make_settings_validates_events_like_load_settings(events, match):
    with pytest.raises(ConfigError, match=match):
        make_settings(events=events)


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
        artifact_hmac_key = "0123456789abcdef"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )
    settings = load_settings(path)
    assert settings.events.enabled is True
    assert settings.output.artifact_hmac_key == "0123456789abcdef"
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
        artifact_hmac_key = "0123456789abcdef"
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
    monkeypatch.setattr("cicerone.model_config.sequential_extra_available", lambda: False)
    with caplog.at_level(logging.WARNING, logger="cicerone.config.load"):
        make_settings(
            models=["collaborative", "sequential"],
            output=IOSettings(
                kind="dataset",
                options={"storage_backend": "local", "path": "/tmp/out"},
                artifact_hmac_key="0123456789abcdef",
            ),
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
        artifact_hmac_key = "0123456789abcdef"
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
        kind = "carrier-pigeon"
        """
        + MINIMAL_IO_TOML,
    )
    with pytest.raises(ConfigError, match="events.kind"):
        load_settings(path)


_S3_KEYS = {"access_key_id": "a", "secret_access_key": "b"}
_R2_ENDPOINT = "https://abc.r2.cloudflarestorage.com"


@pytest.mark.parametrize(
    ("kind", "options", "match"),
    [
        ("db", {}, "database_url"),
        ("s3", _S3_KEYS, "bucket"),
        ("s3", {**_S3_KEYS, "bucket": "events", "mode": "sqs"}, "queue_url"),
        (
            "s3",
            {
                **_S3_KEYS,
                "bucket": "events",
                "mode": "sqs",
                "queue_url": "https://sqs.example/q",
                "endpoint_url": _R2_ENDPOINT,
            },
            "AWS-only",
        ),
        ("redis_streams", {"redis_url": "redis://localhost:6379/0", "consumer_group": "cicerone"}, "stream"),
        ("kafka", {"topic": "t", "group_id": "g"}, "bootstrap_servers"),
        ("rabbitmq", {"amqp_url": "amqp://localhost/"}, "queue"),
    ],
)
def test_load_events_rejects_incomplete_options(kind, options, match):
    with pytest.raises(ConfigError, match=match):
        load_events_settings(
            {"enabled": True, "kind": kind, "options": options}, mode="serve", serve_auth_token="tok"
        )


@pytest.mark.parametrize(
    ("kind", "options"),
    [
        ("db", {"database_url": "sqlite+pysqlite:///:memory:"}),
        ("s3", {**_S3_KEYS, "bucket": "events", "mode": "list", "endpoint_url": _R2_ENDPOINT}),
        (
            "redis_streams",
            {
                "redis_url": "redis://localhost:6379/0",
                "stream": "cicerone:events",
                "consumer_group": "cicerone",
            },
        ),
        (
            "kafka",
            {"bootstrap_servers": "localhost:9092", "topic": "cicerone.events", "group_id": "cicerone"},
        ),
        ("rabbitmq", {"amqp_url": "amqp://localhost/", "queue": "cicerone.events"}),
    ],
)
def test_load_events_accepts_complete_options(kind, options):
    settings = load_events_settings(
        {"enabled": True, "kind": kind, "options": options}, mode="serve", serve_auth_token="tok"
    )
    assert settings.kind == kind
    assert settings.options == options


def test_load_events_online_must_be_table():
    with pytest.raises(ConfigError, match="events.online must be a table"):
        load_events_settings(
            {"enabled": True, "kind": "webhook", "online": "nope"},
            mode="batch",
            serve_auth_token="tok",
        )


def test_load_events_online_requires_events_enabled():
    with pytest.raises(ConfigError, match="events.online.enabled requires events.enabled"):
        load_events_settings(
            {"enabled": False, "online": {"enabled": True}},
            mode="batch",
            serve_auth_token=None,
        )


def test_load_events_incremental_must_be_table():
    with pytest.raises(ConfigError, match="events.incremental must be a table"):
        load_events_settings(
            {"enabled": False, "incremental": "nope"},
            mode="batch",
            serve_auth_token=None,
        )


def test_load_events_disabled_allows_unknown_kind():
    settings = load_events_settings(
        {"enabled": False, "kind": "kafka"},
        mode="batch",
        serve_auth_token=None,
    )
    assert settings.enabled is False
    assert settings.kind == "kafka"


def test_make_settings_events_dataclass_round_trips():
    events = EventsSettings(enabled=True, kind="webhook", ha=True, options={"auth_token": "t"})
    assert (
        make_settings(
            events=events, trigger=TriggerSettings(lock_backend="redis", redis_url="redis://r")
        ).events
        == events
    )


def test_load_online_requires_artifact_hmac_key(tmp_path):
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
        """
        + MINIMAL_IO_TOML,
    )
    with pytest.raises(ConfigError, match="output.artifact_hmac_key"):
        load_settings(path)


def test_load_artifact_hmac_key_rejects_table_before_env_resolve(tmp_path):
    path = write_toml(
        tmp_path,
        """
        [job]
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        [output]
        kind = "dataset"
        artifact_hmac_key = { x = "${MISSING}" }
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )
    with pytest.raises(ConfigError, match="must be a string"):
        load_settings(path)


def test_load_artifact_hmac_key_rejects_non_string(tmp_path):
    path = write_toml(
        tmp_path,
        """
        [job]
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        [output]
        kind = "dataset"
        artifact_hmac_key = 123
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )
    with pytest.raises(ConfigError, match="must be a string"):
        load_settings(path)


def test_load_artifact_hmac_key_too_short(tmp_path):
    path = write_toml(
        tmp_path,
        """
        [job]
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"
        [output]
        kind = "dataset"
        artifact_hmac_key = "short"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
        """,
    )
    with pytest.raises(ConfigError, match="at least 16 bytes"):
        load_settings(path)


def test_input_rejects_artifact_hmac_key(tmp_path):
    path = write_toml(
        tmp_path,
        """
        [job]
        [input]
        kind = "dataset"
        artifact_hmac_key = "0123456789abcdef"
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
    with pytest.raises(ConfigError, match="only valid on \\[output\\]"):
        load_settings(path)


@pytest.mark.parametrize(
    ("token", "expected"), [("${EVENTS_TOKEN}", "secret"), ("$${EVENTS_TOKEN}", "${EVENTS_TOKEN}")]
)
def test_load_events_resolves_auth_token_placeholders_once(monkeypatch, token, expected):
    monkeypatch.setenv("EVENTS_TOKEN", "secret")
    settings = load_events_settings(
        {"enabled": True, "kind": "webhook", "options": {"auth_token": token}},
        mode="serve",
        serve_auth_token=None,
    )
    assert settings.options["auth_token"] == expected
