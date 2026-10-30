from __future__ import annotations

import pytest
from support.toml_config import MINIMAL_IO_TOML, write_toml

from cicerone.config import ConfigError, PublishSettings, load_settings, make_settings
from cicerone.config.publish import load_publish_settings


def test_make_settings_publish_defaults():
    settings = make_settings()
    assert settings.publish.enabled is False
    assert settings.publish.kind == "kafka"
    assert settings.publish.options == {}


def test_make_settings_validates_publish_like_load_settings():
    original = PublishSettings(
        enabled=True, kind="rabbitmq", options={"amqp_url": "amqp://localhost/", "queue": "q"}
    )
    assert make_settings(publish=original).publish == original
    assert (
        make_settings(
            publish={"enabled": True, "kind": "Kafka", "options": {"bootstrap_servers": "b", "topic": "t"}}
        ).publish.kind
        == "kafka"
    )
    with pytest.raises(ConfigError, match="bootstrap_servers"):
        make_settings(publish=PublishSettings(enabled=True, kind="kafka", options={"topic": "t"}))
    with pytest.raises(TypeError):
        make_settings(publish="bad")


def test_load_publish_disabled_allows_unknown_kind():
    settings = load_publish_settings(
        {"enabled": False, "kind": "sns"},
    )
    assert settings.enabled is False
    assert settings.kind == "sns"


def test_load_publish_unknown_kind_when_enabled():
    with pytest.raises(ConfigError, match="publish.kind"):
        load_publish_settings(
            {"enabled": True, "kind": "sns"},
        )


def test_load_publish_kafka_requires_bootstrap():
    with pytest.raises(ConfigError, match="bootstrap_servers"):
        load_publish_settings(
            {"enabled": True, "kind": "kafka", "options": {"topic": "recs"}},
        )


def test_load_publish_kafka_ok():
    settings = load_publish_settings(
        {
            "enabled": True,
            "kind": "kafka",
            "options": {"bootstrap_servers": "localhost:9092", "topic": "recs"},
        },
    )
    assert settings.kind == "kafka"
    settings = load_publish_settings(
        {
            "enabled": True,
            "kind": "rabbitmq",
            "options": {"amqp_url": "amqp://localhost/", "queue": "recs"},
        },
    )
    assert settings.kind == "rabbitmq"


def test_load_publish_options_must_be_table():
    with pytest.raises(ConfigError, match="publish.options must be a table"):
        load_publish_settings(
            {"enabled": False, "options": "nope"},
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
        """
        + MINIMAL_IO_TOML,
    )
    settings = load_settings(path)
    assert settings.publish.enabled is True
    assert settings.publish.options["topic"] == "cicerone.recommendations"
