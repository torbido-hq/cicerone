from __future__ import annotations

import pytest
from support.toml_config import write_toml

from cicerone.config import ConfigError, PublishSettings, load_settings, make_settings
from cicerone.config.publish import coerce_publish_settings, load_publish_settings


def test_make_settings_publish_defaults():
    settings = make_settings()
    assert settings.publish.enabled is False
    assert settings.publish.kind == "kafka"
    assert settings.publish.options == {}


def test_coerce_publish_settings():
    original = PublishSettings(enabled=True, kind="rabbitmq", options={"queue": "q"})
    coerced = coerce_publish_settings(original)
    assert coerced.enabled is True
    assert coerced.kind == "rabbitmq"
    assert coerce_publish_settings(None).enabled is False
    from_dict = coerce_publish_settings({"enabled": True, "kind": "Kafka", "options": {"topic": "t"}})
    assert from_dict.kind == "kafka"
    assert from_dict.enabled is True
    with pytest.raises(TypeError):
        coerce_publish_settings("bad")


def test_load_publish_disabled_allows_unknown_kind():
    settings = load_publish_settings(
        {"enabled": False, "kind": "sns"},
        resolve_env=lambda value, _path: value,
    )
    assert settings.enabled is False
    assert settings.kind == "sns"


def test_load_publish_unknown_kind_when_enabled():
    with pytest.raises(ConfigError, match="publish.kind"):
        load_publish_settings(
            {"enabled": True, "kind": "sns"},
            resolve_env=lambda value, _path: value,
        )


def test_load_publish_kafka_requires_bootstrap():
    with pytest.raises(ConfigError, match="bootstrap_servers"):
        load_publish_settings(
            {"enabled": True, "kind": "kafka", "options": {"topic": "recs"}},
            resolve_env=lambda value, _path: value,
        )


def test_load_publish_kafka_ok():
    settings = load_publish_settings(
        {
            "enabled": True,
            "kind": "kafka",
            "options": {"bootstrap_servers": "localhost:9092", "topic": "recs"},
        },
        resolve_env=lambda value, _path: value,
    )
    assert settings.kind == "kafka"
    settings = load_publish_settings(
        {
            "enabled": True,
            "kind": "rabbitmq",
            "options": {"amqp_url": "amqp://localhost/", "queue": "recs"},
        },
        resolve_env=lambda value, _path: value,
    )
    assert settings.kind == "rabbitmq"


def test_load_publish_options_must_be_table():
    with pytest.raises(ConfigError, match="publish.options must be a table"):
        load_publish_settings(
            {"enabled": False, "options": "nope"},
            resolve_env=lambda value, _path: value,
        )


def test_load_settings_publish_section(tmp_path):
    path = write_toml(
        tmp_path,
        """
        [job]
        [publish]
        enabled = true
        kind = "kafka"
        [publish.options]
        bootstrap_servers = "localhost:9092"
        topic = "cicerone.recommendations"
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
    assert settings.publish.enabled is True
    assert settings.publish.options["topic"] == "cicerone.recommendations"
