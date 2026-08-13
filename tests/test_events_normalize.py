from __future__ import annotations

import pytest
from support.events import event_payload

from cicerone.events.normalize import (
    EventNormalizeError,
    events_to_dataframe,
    normalize_event,
    normalize_events,
)


def test_normalize_event_and_errors():
    event = normalize_event(event_payload())
    assert event.user_id == "u1"
    assert event.quantity == 1
    with pytest.raises(EventNormalizeError, match="missing"):
        normalize_event({"user_id": "u1"})
    with pytest.raises(EventNormalizeError, match="quantity"):
        normalize_event(event_payload(quantity=0))


def test_normalize_occurred_at_epoch_and_z():
    from datetime import UTC, datetime

    epoch = normalize_event(event_payload(occurred_at=1_724_000_000))
    assert epoch.occurred_at == datetime.fromtimestamp(1_724_000_000, tz=UTC)
    zulu = normalize_event(event_payload(occurred_at="2026-08-13T12:00:00Z"))
    assert zulu.occurred_at.tzinfo is not None
    assert zulu.occurred_at.hour == 12


def test_normalize_more_edge_cases():
    with pytest.raises(EventNormalizeError, match="JSON object"):
        normalize_event("not-a-dict")
    with pytest.raises(EventNormalizeError, match="quantity"):
        normalize_event(event_payload(quantity="x"))
    with pytest.raises(EventNormalizeError, match="non-empty"):
        normalize_event(event_payload(user_id="  "))
    with pytest.raises(EventNormalizeError, match="occurred_at"):
        normalize_event(event_payload(occurred_at="not-a-date"))
    bare = normalize_event(
        {
            "user_id": "u1",
            "item_id": "i1",
            "event_type": "view",
            "occurred_at": "2026-08-13T12:00:00",
            "idempotency_key": "idem-1",
        }
    )
    assert bare.event_id == "idem-1"
    assert bare.quantity == 1
    assert events_to_dataframe([]).empty
    assert len(normalize_events([event_payload(event_id="z")])) == 1
