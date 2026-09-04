from __future__ import annotations

import json

import pandas as pd
import pytest
from support.events import event_payload
from support.fake_kafka import install_fake_kafka
from support.fake_rabbitmq import install_fake_rabbitmq

from cicerone.config import ConfigError, IOSettings, PublishSettings, make_settings
from cicerone.events.normalize import normalize_event
from cicerone.events.updater import IncrementalUpdater
from cicerone.feature_config import FeatureConfig
from cicerone.io.factory import build_output_sink
from cicerone.publish import build_publisher, registered_publish_kinds
from cicerone.publish.factory import build_publisher_from_kind
from cicerone.publish.kafka import KafkaPublisher, validate_kafka_publish_options
from cicerone.publish.payload import user_recommendation_messages
from cicerone.publish.rabbitmq import RabbitMQPublisher, validate_rabbitmq_publish_options


def _recs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "popular"},
            {"user_id": "u1", "item_id": "i2", "rank": 2, "score": 0.8, "source": "latest"},
            {"user_id": "u2", "item_id": "i1", "rank": 1, "score": 0.7, "source": "popular"},
        ]
    )


def test_user_recommendation_messages_one_per_user():
    messages = user_recommendation_messages(_recs_frame())
    assert [user_id for user_id, _body in messages] == ["u1", "u2"]
    first = json.loads(messages[0][1])
    assert first["user_id"] == "u1"
    assert len(first["recommendations"]) == 2
    assert first["recommendations"][0]["item_id"] == "i1"


def test_user_recommendation_messages_empty():
    assert user_recommendation_messages(pd.DataFrame()) == []
    assert user_recommendation_messages(pd.DataFrame({"item_id": ["i1"]})) == []


def test_user_recommendation_messages_keeps_reasons_and_variant():
    frame = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "rank": 1,
                "score": 0.9,
                "source": "popular",
                "reasons": "popular",
                "variant": "control",
            }
        ]
    )
    _user_id, body = user_recommendation_messages(frame)[0]
    rec = json.loads(body)["recommendations"][0]
    assert rec["reasons"] == "popular"
    assert rec["variant"] == "control"


def test_registered_publish_kinds():
    assert registered_publish_kinds() == ("kafka", "rabbitmq")


def test_validate_kafka_publish_options():
    with pytest.raises(ConfigError, match="bootstrap_servers"):
        validate_kafka_publish_options({"topic": "t"})
    with pytest.raises(ConfigError, match="topic"):
        validate_kafka_publish_options({"bootstrap_servers": "h:9092"})


def test_validate_rabbitmq_publish_options():
    with pytest.raises(ConfigError, match="amqp_url"):
        validate_rabbitmq_publish_options({"queue": "q"})
    with pytest.raises(ConfigError, match="queue"):
        validate_rabbitmq_publish_options({"amqp_url": "amqp://localhost/"})
    validate_rabbitmq_publish_options({"amqp_url": "amqp://localhost/", "exchange": "recs"})


def test_kafka_publisher_emits_per_user(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    publisher = KafkaPublisher({"bootstrap_servers": "localhost:9092", "topic": "cicerone.recs"})
    publisher.connect()
    publisher.publish(_recs_frame())
    keys = [key.decode() if isinstance(key, bytes) else key for _, key, _ in broker.produced]
    assert keys == ["u1", "u2"]
    body = json.loads(broker.produced[0][2])
    assert body["user_id"] == "u1"
    publisher.close()


def test_kafka_publisher_flush_timeout(monkeypatch):
    install_fake_kafka(monkeypatch)
    publisher = KafkaPublisher({"bootstrap_servers": "localhost:9092", "topic": "cicerone.recs"})
    publisher.connect()
    publisher._producer.flush_remaining = 2
    with pytest.raises(RuntimeError, match="timed out"):
        publisher.publish(_recs_frame())


def test_rabbitmq_publisher_close_tolerates_failure(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    publisher = RabbitMQPublisher({"amqp_url": "amqp://localhost/", "queue": "q"})
    publisher.connect()

    def _boom() -> None:
        raise RuntimeError("close fail")

    broker.connection.channel_obj.close = _boom  # type: ignore[method-assign]
    broker.connection.close = _boom  # type: ignore[method-assign]
    publisher.close()


def test_build_publisher_disabled():
    settings = make_settings()
    assert settings.publish.enabled is False
    assert build_publisher(settings) is None
    assert build_publisher(PublishSettings()) is None


def test_build_publisher_kafka(monkeypatch):
    install_fake_kafka(monkeypatch)
    settings = make_settings(
        publish=PublishSettings(
            enabled=True,
            kind="kafka",
            options={"bootstrap_servers": "localhost:9092", "topic": "recs"},
        )
    )
    publisher = build_publisher(settings)
    assert publisher is not None
    publisher.close()


def test_missing_kafka_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "confluent_kafka":
            raise ImportError("no kafka")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(ConfigError, match=r"cicerone-recommender\[kafka\]"):
        build_publisher_from_kind("kafka", {"bootstrap_servers": "h:9092", "topic": "t"})


def test_missing_rabbitmq_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "pika":
            raise ImportError("no pika")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    with pytest.raises(ConfigError, match=r"cicerone-recommender\[rabbitmq\]"):
        build_publisher_from_kind("rabbitmq", {"amqp_url": "amqp://localhost/", "queue": "q"})


def test_updater_publishes_after_replace(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )
    captured: list[pd.DataFrame] = []

    class _Pub:
        def publish(self, df: pd.DataFrame) -> None:
            captured.append(df.copy())

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
    assert len(captured) == 1
    assert "u1" in set(captured[0]["user_id"].astype(str))


def test_kafka_publisher_not_connected():
    publisher = KafkaPublisher({"bootstrap_servers": "localhost:9092", "topic": "t"})
    with pytest.raises(RuntimeError, match="not connected"):
        publisher.publish(_recs_frame())
    publisher.close()


def test_rabbitmq_publisher_not_connected():
    publisher = RabbitMQPublisher({"amqp_url": "amqp://localhost/", "queue": "q"})
    with pytest.raises(RuntimeError, match="not connected"):
        publisher.publish(_recs_frame())


def test_rabbitmq_publisher_exchange(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    publisher = RabbitMQPublisher(
        {
            "amqp_url": "amqp://guest:guest@localhost:5672/",
            "exchange": "recs",
            "routing_key": "recommendations",
        }
    )
    publisher.connect()
    publisher.publish(_recs_frame())
    assert broker.published[0][0] == "recs"
    assert broker.published[0][1] == "recommendations"
    publisher.close()


def test_rabbitmq_publisher_queue_uses_queue_as_routing_key(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    publisher = RabbitMQPublisher({"amqp_url": "amqp://localhost/", "queue": "recs"})
    publisher.connect()
    publisher.publish(_recs_frame())
    assert [key for _exchange, key, _body in broker.published] == ["recs", "recs"]
    publisher.close()


def test_rabbitmq_publisher_exchange_empty_routing_key(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    publisher = RabbitMQPublisher({"amqp_url": "amqp://localhost/", "exchange": "recs"})
    publisher.connect()
    publisher.publish(_recs_frame())
    assert [(exchange, key) for exchange, key, _body in broker.published] == [
        ("recs", ""),
        ("recs", ""),
    ]
    publisher.close()


def test_kafka_publisher_connect_failure(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    broker.list_topics_error = RuntimeError("down")
    publisher = KafkaPublisher({"bootstrap_servers": "localhost:9092", "topic": "t"})
    with pytest.raises(ConfigError, match="unreachable"):
        publisher.connect()
    assert broker.flush_calls == [1]


def test_rabbitmq_publisher_connect_failure(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    broker.connect_error = RuntimeError("down")
    publisher = RabbitMQPublisher({"amqp_url": "amqp://localhost/", "queue": "q"})
    with pytest.raises(ConfigError, match="unreachable"):
        publisher.connect()


def test_rabbitmq_publisher_closes_connection_when_declare_fails(monkeypatch):
    broker = install_fake_rabbitmq(monkeypatch)
    broker.queue_declare_error = RuntimeError("no queue")
    publisher = RabbitMQPublisher({"amqp_url": "amqp://localhost/", "queue": "q"})
    with pytest.raises(ConfigError, match="unreachable"):
        publisher.connect()
    assert broker.connection.closed is True


def test_build_publisher_unknown_kind():
    with pytest.raises(ConfigError, match="publish.kind"):
        build_publisher_from_kind("sns", {})
    with pytest.raises(ConfigError, match="publish.kind"):
        build_publisher(PublishSettings(enabled=True, kind="sns"))


def test_kafka_publisher_close_flush_failure(monkeypatch):
    install_fake_kafka(monkeypatch)
    publisher = KafkaPublisher({"bootstrap_servers": "localhost:9092", "topic": "t"})
    publisher.connect()

    def _boom(_timeout=None):
        raise RuntimeError("flush fail")

    publisher._producer.flush = _boom  # type: ignore[method-assign]
    publisher.close()


def test_publish_empty_frame_is_noop(monkeypatch):
    broker = install_fake_kafka(monkeypatch)
    publisher = KafkaPublisher({"bootstrap_servers": "localhost:9092", "topic": "t"})
    publisher.connect()
    publisher.publish(pd.DataFrame())
    assert broker.produced == []
    publisher.close()


def test_updater_publish_failure_does_not_stop_apply(tmp_path, feature_config: FeatureConfig):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "old", "rank": 1, "score": 1.0, "source": "personalized"}]
    ).to_parquet(out / "recommendations.parquet", index=False)
    settings = make_settings(
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(out)}),
        top_k=5,
    )

    class _Boom:
        def publish(self, df: pd.DataFrame) -> None:
            raise RuntimeError("broker down")

    updater = IncrementalUpdater(
        sink=build_output_sink(settings.output),
        output_settings=settings.output,
        feature_config=feature_config,
        top_k=5,
        publisher=_Boom(),
    )
    events = [normalize_event(event_payload(user_id="u1", item_id="i9", event_id="n1"))]
    assert updater.apply(events) == 1
    assert updater.events_applied == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "success"
