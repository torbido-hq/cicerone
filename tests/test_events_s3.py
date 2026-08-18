from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws
from support.events import event_payload

from cicerone.config import ConfigError
from cicerone.events.registry import build_event_source, registered_event_source_kinds
from cicerone.events.s3 import (
    _SQS_NACK_VISIBILITY_TIMEOUT_SECONDS,
    S3EventSource,
    _events_from_body,
    _s3_records_from_sqs_body,
)


def _creds(**extra):
    return {
        "access_key_id": "testing",
        "secret_access_key": "testing",
        "bucket": "events-bucket",
        "region_name": "us-east-1",
        **extra,
    }


def _put_event(client, key: str, payload: dict | list) -> None:
    client.put_object(Bucket="events-bucket", Key=key, Body=json.dumps(payload).encode())


def _s3_notification(key: str, bucket: str = "events-bucket") -> str:
    return json.dumps(
        {
            "Records": [
                {
                    "eventName": "ObjectCreated:Put",
                    "s3": {"bucket": {"name": bucket}, "object": {"key": key, "eTag": "abc"}},
                }
            ]
        }
    )


@mock_aws
def test_s3_registered_and_build_list_mode():
    assert "s3" in registered_event_source_kinds()
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="events-bucket")
    source = build_event_source("s3", _creds(mode="list"))
    assert isinstance(source, S3EventSource)


@mock_aws
def test_s3_list_poll_ack_advances_marker(tmp_path):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="events-bucket")
    _put_event(client, "events/a.json", event_payload(event_id="e1", item_id="i1"))
    _put_event(client, "events/b.json", event_payload(event_id="e2", item_id="i2"))
    marker = tmp_path / "marker.json"
    source = S3EventSource(_creds(mode="list", prefix="events/", marker_path=str(marker)))
    source.connect()
    first = list(source.poll(1))
    assert [event.event_id for event in first] == ["e1"]
    assert source.health().lag is not None and source.health().lag >= 1
    source.ack([first[0].event_id])
    assert json.loads(marker.read_text())["key"] == "events/a.json"
    second = list(source.poll(10))
    assert [event.event_id for event in second] == ["e2"]
    source.ack([second[0].event_id])
    assert list(source.poll(10)) == []
    assert json.loads(marker.read_text())["key"] == "events/b.json"


@mock_aws
def test_s3_list_nack_allows_repoll():
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="events-bucket")
    _put_event(client, "events/a.json", event_payload(event_id="e1"))
    source = S3EventSource(_creds(mode="list", prefix="events/"))
    source.connect()
    first = list(source.poll(10))
    assert len(first) == 1
    source.nack(first)
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["e1"]


@mock_aws
def test_s3_list_array_payload_and_stable_ids():
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="events-bucket")
    client.put_object(
        Bucket="events-bucket",
        Key="batch.json",
        Body=json.dumps(
            [
                {
                    "user_id": "u1",
                    "item_id": "i1",
                    "event_type": "purchase",
                    "quantity": 1,
                    "occurred_at": "2026-08-14T12:00:00Z",
                },
                {
                    "user_id": "u1",
                    "item_id": "i2",
                    "event_type": "view",
                    "quantity": 1,
                    "occurred_at": "2026-08-14T12:01:00Z",
                },
            ]
        ),
    )
    source = S3EventSource(_creds(mode="list"))
    source.connect()
    events = list(source.poll(10))
    assert len(events) == 2
    assert events[0].item_id == "i1"
    assert events[1].item_id == "i2"
    assert events[0].event_id.startswith("events-bucket/batch.json|")
    source.ack([event.event_id for event in events])
    assert list(source.poll(10)) == []


@mock_aws
def test_s3_sqs_poll_ack_deletes_message():
    s3 = boto3.client("s3", region_name="us-east-1")
    sqs = boto3.client("sqs", region_name="us-east-1")
    s3.create_bucket(Bucket="events-bucket")
    queue_url = sqs.create_queue(QueueName="events")["QueueUrl"]
    _put_event(s3, "events/e1.json", event_payload(event_id="sqs-1", item_id="i9"))
    sqs.send_message(QueueUrl=queue_url, MessageBody=_s3_notification("events/e1.json"))
    source = S3EventSource(_creds(mode="sqs", queue_url=queue_url))
    source.connect()
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["sqs-1"]
    source.ack([events[0].event_id])
    assert list(source.poll(10)) == []
    attrs = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["ApproximateNumberOfMessages"])[
        "Attributes"
    ]
    assert int(attrs["ApproximateNumberOfMessages"]) == 0


@mock_aws
def test_s3_sqs_nack_requeues_and_ack_still_deletes():
    s3 = boto3.client("s3", region_name="us-east-1")
    sqs = boto3.client("sqs", region_name="us-east-1")
    s3.create_bucket(Bucket="events-bucket")
    queue_url = sqs.create_queue(QueueName="events-nack")["QueueUrl"]
    _put_event(s3, "events/e1.json", event_payload(event_id="sqs-nack", item_id="i9"))
    sqs.send_message(QueueUrl=queue_url, MessageBody=_s3_notification("events/e1.json"))
    source = S3EventSource(_creds(mode="sqs", queue_url=queue_url))
    source.connect()
    first = list(source.poll(10))
    assert [event.event_id for event in first] == ["sqs-nack"]
    source.nack(first)
    source.nack(first)
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["sqs-nack"]
    source.ack([again[0].event_id])
    assert list(source.poll(10)) == []
    attrs = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["ApproximateNumberOfMessages"])[
        "Attributes"
    ]
    assert int(attrs["ApproximateNumberOfMessages"]) == 0


@mock_aws
def test_s3_sqs_nack_extends_visibility(monkeypatch):
    s3 = boto3.client("s3", region_name="us-east-1")
    sqs = boto3.client("sqs", region_name="us-east-1")
    s3.create_bucket(Bucket="events-bucket")
    queue_url = sqs.create_queue(QueueName="events-vis")["QueueUrl"]
    _put_event(s3, "events/e1.json", event_payload(event_id="sqs-vis"))
    sqs.send_message(QueueUrl=queue_url, MessageBody=_s3_notification("events/e1.json"))
    source = S3EventSource(_creds(mode="sqs", queue_url=queue_url))
    source.connect()
    assert source._sqs is not None
    calls = {"n": 0}
    real = source._sqs.change_message_visibility

    def counting(**kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        assert kwargs["VisibilityTimeout"] == _SQS_NACK_VISIBILITY_TIMEOUT_SECONDS
        return real(**kwargs)

    monkeypatch.setattr(source._sqs, "change_message_visibility", counting)
    first = list(source.poll(10))
    source.nack(first)
    assert calls["n"] == 1
    source._adopt_sqs_receipt("adopted-receipt", {first[0].event_id})
    assert source._event_batch[first[0].event_id].receipt_handle == "adopted-receipt"


@mock_aws
def test_s3_sqs_redelivery_adopts_receipt_instead_of_deleting():
    s3 = boto3.client("s3", region_name="us-east-1")
    sqs = boto3.client("sqs", region_name="us-east-1")
    s3.create_bucket(Bucket="events-bucket")
    queue_url = sqs.create_queue(QueueName="events-adopt")["QueueUrl"]
    _put_event(s3, "events/e1.json", event_payload(event_id="sqs-adopt"))
    sqs.send_message(QueueUrl=queue_url, MessageBody=_s3_notification("events/e1.json"))
    source = S3EventSource(_creds(mode="sqs", queue_url=queue_url))
    source.connect()
    first = list(source.poll(10))
    old_receipt = source._event_batch[first[0].event_id].receipt_handle
    assert old_receipt is not None
    sqs.change_message_visibility(QueueUrl=queue_url, ReceiptHandle=old_receipt, VisibilityTimeout=0)
    source._fetch_sqs(10)
    assert list(source._pending) == []
    new_receipt = source._event_batch[first[0].event_id].receipt_handle
    assert new_receipt is not None
    assert new_receipt != old_receipt
    source.ack([first[0].event_id])
    assert list(source.poll(10)) == []


@mock_aws
def test_s3_list_partial_nack_keeps_sibling_events():
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="events-bucket")
    client.put_object(
        Bucket="events-bucket",
        Key="batch.json",
        Body=json.dumps(
            [
                event_payload(event_id="keep", item_id="i1"),
                event_payload(event_id="retry", item_id="i2"),
            ]
        ),
    )
    source = S3EventSource(_creds(mode="list"))
    source.connect()
    events = list(source.poll(10))
    assert {event.event_id for event in events} == {"keep", "retry"}
    retry = [event for event in events if event.event_id == "retry"]
    source.nack(retry)
    source.ack(["keep"])
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["retry"]
    source.ack(["retry"])
    assert list(source.poll(10)) == []


@mock_aws
def test_s3_sqs_supports_sns_envelope():
    s3 = boto3.client("s3", region_name="us-east-1")
    sqs = boto3.client("sqs", region_name="us-east-1")
    s3.create_bucket(Bucket="events-bucket")
    queue_url = sqs.create_queue(QueueName="events-sns")["QueueUrl"]
    _put_event(s3, "x.json", event_payload(event_id="via-sns"))
    envelope = json.dumps(
        {
            "Type": "Notification",
            "TopicArn": "arn:aws:sns:us-east-1:123:topic",
            "Message": _s3_notification("x.json"),
        }
    )
    sqs.send_message(QueueUrl=queue_url, MessageBody=envelope)
    source = S3EventSource(_creds(queue_url=queue_url))  # mode inferred
    source.connect()
    events = list(source.poll(5))
    assert [event.event_id for event in events] == ["via-sns"]


def test_s3_requires_queue_url_for_sqs_mode():
    with pytest.raises(ConfigError, match="queue_url"):
        S3EventSource(_creds(mode="sqs"))


def test_s3_rejects_unknown_mode():
    with pytest.raises(ConfigError, match="mode"):
        S3EventSource(_creds(mode="kafka"))


@mock_aws
def test_s3_poll_after_close_raises_controlled_error():
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="events-bucket")
    _put_event(client, "events/a.json", event_payload(event_id="e1"))
    source = S3EventSource(_creds(mode="list", prefix="events/"))
    source.connect()
    source.close()
    with pytest.raises(RuntimeError, match="connect"):
        source.poll(10)


def test_s3_tuning_options_and_rejects_non_positive():
    source = S3EventSource(
        _creds(
            mode="list",
            list_page_size=50,
            sqs_lag_cache_ttl_seconds=1.5,
            sqs_client_timeout_seconds=3,
        )
    )
    assert source._list_page_size == 50
    assert source._sqs_lag_cache_ttl_seconds == 1.5
    assert source._sqs_client_timeout_seconds == 3.0
    with pytest.raises(ConfigError, match="list_page_size"):
        S3EventSource(_creds(mode="list", list_page_size=0))
    with pytest.raises(ConfigError, match="list_page_size"):
        S3EventSource(_creds(mode="list", list_page_size="abc"))
    with pytest.raises(ConfigError, match="sqs_lag_cache_ttl_seconds"):
        S3EventSource(_creds(mode="list", sqs_lag_cache_ttl_seconds="nope"))
    with pytest.raises(ConfigError, match="wait_time_seconds"):
        S3EventSource(_creds(mode="list", wait_time_seconds="slow"))
    with pytest.raises(ConfigError, match="max_messages"):
        S3EventSource(_creds(mode="list", max_messages="many"))


@mock_aws
def test_s3_sqs_health_caches_visible_lag(monkeypatch):
    s3 = boto3.client("s3", region_name="us-east-1")
    sqs = boto3.client("sqs", region_name="us-east-1")
    s3.create_bucket(Bucket="events-bucket")
    queue_url = sqs.create_queue(QueueName="events-lag-cache")["QueueUrl"]
    source = S3EventSource(_creds(mode="sqs", queue_url=queue_url))
    source.connect()
    calls = {"n": 0}
    real = source._sqs.get_queue_attributes

    def counting_attrs(**kwargs):
        calls["n"] += 1
        return real(**kwargs)

    assert source._sqs is not None
    monkeypatch.setattr(source._sqs, "get_queue_attributes", counting_attrs)
    source.health()
    source.health()
    assert calls["n"] == 1


def test_s3_rejects_sqs_with_endpoint_url():
    with pytest.raises(ConfigError, match="AWS-only"):
        S3EventSource(
            _creds(
                mode="sqs",
                queue_url="https://sqs.us-east-1.amazonaws.com/123/q",
                endpoint_url="https://abc.r2.cloudflarestorage.com",
            )
        )


def test_events_from_body_validation():
    assert _events_from_body(b"  ", bucket="b", key="k", etag="e") == []
    with pytest.raises(ValueError, match="invalid JSON"):
        _events_from_body(b"{", bucket="b", key="k", etag="e")
    with pytest.raises(ValueError, match="object or array"):
        _events_from_body(b"1", bucket="b", key="k", etag="e")
    # Non-dict / invalid elements are skipped; valid siblings are kept.
    kept = _events_from_body(
        json.dumps(
            [
                1,
                {
                    "user_id": "u1",
                    "item_id": "i1",
                    "event_type": "view",
                    "quantity": 1,
                    "occurred_at": "2026-08-14T12:00:00Z",
                    "event_id": "ok",
                },
                {"user_id": "u1"},  # missing required fields
            ]
        ).encode(),
        bucket="b",
        key="k",
        etag="e",
    )
    assert [event.event_id for event in kept] == ["ok"]
    with pytest.raises(ValueError, match="JSON object"):
        _s3_records_from_sqs_body("[]")
    with pytest.raises(ValueError, match="Records"):
        _s3_records_from_sqs_body("{}")
    assert _s3_records_from_sqs_body(
        json.dumps(
            {
                "Records": [
                    "skip",
                    {
                        "eventName": "ObjectRemoved:Delete",
                        "s3": {"bucket": {"name": "b"}, "object": {"key": "x"}},
                    },
                    {"eventName": "ObjectCreated:Put", "s3": "bad"},
                    {
                        "eventName": "ObjectCreated:Put",
                        "s3": {"bucket": {"name": "b"}, "object": {"key": "a%2Fb.json"}},
                    },
                ]
            }
        )
    ) == [("b", "a/b.json")]


@mock_aws
def test_s3_list_skips_bad_object_and_empty_payload():
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="events-bucket")
    client.put_object(Bucket="events-bucket", Key="events/bad.json", Body=b"not-json")
    client.put_object(Bucket="events-bucket", Key="events/empty.json", Body=b"")
    _put_event(client, "events/ok.json", event_payload(event_id="ok"))
    source = S3EventSource(_creds(mode="list", prefix="events/"))
    source.connect()
    assert source.poll(0) == []
    with pytest.raises(RuntimeError, match="connect"):
        S3EventSource(_creds(mode="list")).poll(1)
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["ok"]
    source.close()
    assert source.health().connected is False


@mock_aws
def test_s3_list_loads_marker_and_ignores_corrupt(tmp_path):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="events-bucket")
    _put_event(client, "events/a.json", event_payload(event_id="a"))
    _put_event(client, "events/b.json", event_payload(event_id="b"))
    marker = tmp_path / "marker.json"
    marker.write_text(json.dumps({"key": "events/a.json"}))
    source = S3EventSource(_creds(mode="list", prefix="events/", marker_path=str(marker)))
    source.connect()
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["b"]
    source.close()

    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    source2 = S3EventSource(
        _creds(mode="list", prefix="events/", marker_path=str(bad), initial_marker="events/a.json")
    )
    source2.connect()
    assert list(source2.poll(10))[0].event_id == "b"


@mock_aws
def test_s3_sqs_poison_and_missing_object_and_health():
    s3 = boto3.client("s3", region_name="us-east-1")
    sqs = boto3.client("sqs", region_name="us-east-1")
    s3.create_bucket(Bucket="events-bucket")
    queue_url = sqs.create_queue(QueueName="events-poison")["QueueUrl"]
    sqs.send_message(QueueUrl=queue_url, MessageBody="not-json")
    sqs.send_message(QueueUrl=queue_url, MessageBody=_s3_notification("missing.json"))
    _put_event(s3, "ok.json", event_payload(event_id="ok-sqs"))
    sqs.send_message(QueueUrl=queue_url, MessageBody=_s3_notification("ok.json"))
    source = S3EventSource(_creds(mode="sqs", queue_url=queue_url))
    source.connect()
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["ok-sqs"]
    assert source.health().lag is not None and source.health().lag >= 1
    source.ack(["unknown-id", events[0].event_id])


@mock_aws
def test_s3_sqs_filters_bucket_and_prefix():
    s3 = boto3.client("s3", region_name="us-east-1")
    sqs = boto3.client("sqs", region_name="us-east-1")
    s3.create_bucket(Bucket="events-bucket")
    s3.create_bucket(Bucket="other-bucket")
    queue_url = sqs.create_queue(QueueName="events-filter")["QueueUrl"]
    _put_event(s3, "events/keep.json", event_payload(event_id="keep"))
    _put_event(s3, "other/skip.json", event_payload(event_id="skip-prefix"))
    s3.put_object(
        Bucket="other-bucket",
        Key="events/foreign.json",
        Body=json.dumps(event_payload(event_id="skip-bucket")).encode(),
    )
    sqs.send_message(QueueUrl=queue_url, MessageBody=_s3_notification("other/skip.json"))
    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=_s3_notification("events/foreign.json", bucket="other-bucket"),
    )
    sqs.send_message(QueueUrl=queue_url, MessageBody=_s3_notification("events/keep.json"))
    source = S3EventSource(_creds(mode="sqs", queue_url=queue_url, prefix="events/"))
    source.connect()
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["keep"]


def test_matching_sqs_records_helper():
    source = S3EventSource(_creds(mode="list", prefix="events/"))
    assert source._matching_sqs_records(
        [
            ("events-bucket", "events/a.json"),
            ("events-bucket", "other/b.json"),
            ("nope", "events/c.json"),
        ]
    ) == [("events-bucket", "events/a.json")]


@mock_aws
def test_s3_list_duplicate_object_advances_marker(tmp_path):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="events-bucket")
    payload = event_payload(event_id="same")
    _put_event(client, "events/a.json", payload)
    marker = tmp_path / "m.json"
    source = S3EventSource(_creds(mode="list", prefix="events/", marker_path=str(marker)))
    source.connect()
    first = list(source.poll(10))
    assert first[0].event_id == "same"
    _put_event(client, "events/b.json", payload)
    assert list(source.poll(10)) == []
    source.ack([first[0].event_id])
    assert json.loads(marker.read_text())["key"] in {"events/a.json", "events/b.json"}


@mock_aws
def test_s3_list_respects_max_events_with_pending():
    """Large object payloads must not over-deliver past max_events (buffer safety)."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="events-bucket")
    client.put_object(
        Bucket="events-bucket",
        Key="events/big.json",
        Body=json.dumps(
            [
                {
                    "user_id": "u1",
                    "item_id": f"i{i}",
                    "event_type": "view",
                    "quantity": 1,
                    "occurred_at": "2026-08-14T12:00:00Z",
                    "event_id": f"e{i}",
                }
                for i in range(5)
            ]
        ),
    )
    source = S3EventSource(_creds(mode="list", prefix="events/"))
    source.connect()
    first = list(source.poll(2))
    assert [event.event_id for event in first] == ["e0", "e1"]
    assert source.health().lag == 5  # 2 in_flight delivered + 3 still pending
    second = list(source.poll(10))
    assert [event.event_id for event in second] == ["e2", "e3", "e4"]
    source.ack([event.event_id for event in first + second])
    assert list(source.poll(10)) == []


@mock_aws
def test_s3_list_prefix_is_directory_scoped():
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="events-bucket")
    _put_event(client, "events/a.json", event_payload(event_id="keep"))
    _put_event(client, "events_other/b.json", event_payload(event_id="skip"))
    source = S3EventSource(_creds(mode="list", prefix="events"))  # no trailing slash
    source.connect()
    events = list(source.poll(10))
    assert [event.event_id for event in events] == ["keep"]


@mock_aws
def test_count_list_lag_truncated(monkeypatch):
    # retained name for history; list lag is local pending+in_flight (no R2 scan)
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="events-bucket")
    _put_event(client, "a.json", event_payload(event_id="x"))
    source = S3EventSource(_creds(mode="list"))
    source.connect()
    events = list(source.poll(1))
    assert source.health().lag == 1
    source.ack([events[0].event_id])
    assert source.health().lag == 0
