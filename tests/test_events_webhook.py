from __future__ import annotations

import pytest
from support.events import event_payload

from cicerone.events.normalize import EventNormalizeError
from cicerone.events.webhook import WebhookEventSource


def test_webhook_ingest_poll_ack_health():
    source = WebhookEventSource({})
    source.connect()
    source.ingest(event_payload(event_id="a"))
    source.ingest([event_payload(event_id="b", item_id="i2")])
    health = source.health()
    assert health.connected is True
    assert health.lag == 2
    assert health.detail == "webhook"
    polled = list(source.poll(1))
    assert len(polled) == 1
    assert source.health().lag == 2
    source.ack([polled[0].event_id])
    assert source.health().lag == 1


def test_webhook_dedupes_event_id_and_nack():
    source = WebhookEventSource({})
    first = source.ingest(event_payload(event_id="same"))
    second = source.ingest(event_payload(event_id="same", item_id="other"))
    assert len(first) == 1
    assert second == []
    polled = list(source.poll(10))
    assert len(polled) == 1
    source.nack(polled)
    assert source.health().lag == 1
    again = list(source.poll(10))
    assert [event.event_id for event in again] == ["same"]


def test_webhook_poll_zero_and_reject_scalar():
    source = WebhookEventSource({})
    assert source.poll(0) == []
    with pytest.raises(EventNormalizeError):
        source.ingest(123)


def test_webhook_ack_unknown_id_is_noop():
    source = WebhookEventSource({})
    source.ingest(event_payload(event_id="x"))
    source.poll(1)
    source.ack(["missing"])
    assert source.health().lag == 1


def test_webhook_max_pending_rejects_when_full():
    source = WebhookEventSource({"max_pending": 1})
    source.ingest(event_payload(event_id="a"))
    with pytest.raises(EventNormalizeError, match="backlog full"):
        source.ingest(event_payload(event_id="b"))
    with pytest.raises(ValueError, match="max_pending"):
        WebhookEventSource({"max_pending": 0})


def test_webhook_max_pending_ignores_duplicates():
    source = WebhookEventSource({"max_pending": 1})
    source.ingest(event_payload(event_id="a"))
    assert source.ingest(event_payload(event_id="a", item_id="other")) == []
    assert source.health().lag == 1
