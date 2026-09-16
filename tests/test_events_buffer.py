from __future__ import annotations

import pytest
from support.events import event_payload

from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.normalize import event_fingerprint, normalize_event


def test_micro_batch_buffer_count_and_dedupe():
    buffer = MicroBatchBuffer(batch_size=2, batch_window_seconds=60.0)
    e1 = normalize_event(event_payload(event_id="1"))
    e2 = normalize_event(event_payload(event_id="dup"))  # same fingerprint as e1
    e3 = normalize_event(event_payload(event_id="3", item_id="i2"))
    first = buffer.extend([e1, e2])
    assert first.kept_count == 1
    assert len(first.duplicates) == 1
    assert buffer.ready() is False
    second = buffer.extend([e3])
    assert second.kept_count == 1
    flushed = buffer.flush_if_ready()
    assert len(flushed) == 2


def test_micro_batch_buffer_window():
    buffer = MicroBatchBuffer(batch_size=100, batch_window_seconds=10.0)
    buffer.extend([normalize_event(event_payload())])
    assert buffer.ready(now=0.0) is False
    # Force window start for deterministic ready() without sleeping.
    buffer._window_started_at = 0.0
    assert buffer.ready(now=10.0) is True


def test_buffer_validation_and_len():
    with pytest.raises(ValueError, match="batch_size"):
        MicroBatchBuffer(batch_size=0, batch_window_seconds=1.0)
    with pytest.raises(ValueError, match="batch_window"):
        MicroBatchBuffer(batch_size=1, batch_window_seconds=0)
    buffer = MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0, dedupe=False)
    assert len(buffer) == 0
    assert buffer.flush_if_ready() == []
    buffer.extend([normalize_event(event_payload()), normalize_event(event_payload(event_id="same-fp"))])
    assert len(buffer) == 2
    assert buffer.ready() is False


def test_buffer_contains_fingerprint():
    buffer = MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0)
    event = normalize_event(event_payload(event_id="fp-1", item_id="a"))
    assert buffer.contains_fingerprint(event_fingerprint(event)) is False
    buffer.extend([event])
    assert buffer.contains_event_id("fp-1") is True
    assert buffer.contains_fingerprint(event_fingerprint(event)) is True


def test_buffer_dedupes_by_event_id():
    buffer = MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0)
    first = normalize_event(event_payload(event_id="id-1", item_id="a"))
    second = normalize_event(event_payload(event_id="id-1", item_id="b"))
    result = buffer.extend([first, second])
    assert result.kept_count == 1
    assert len(result.duplicates) == 1


def test_buffer_caps_events_and_clears_dedupe_on_flush():
    buffer = MicroBatchBuffer(batch_size=2, batch_window_seconds=60.0, max_events=2)
    events = [normalize_event(event_payload(event_id=f"e{i}", item_id=f"i{i}")) for i in range(4)]
    result = buffer.extend(events)
    assert result.kept_count == 2
    assert len(result.overflow) == 2
    assert buffer.remaining_capacity == 0
    assert len(buffer._event_ids) == 2
    flushed = buffer.flush()
    assert len(flushed) == 2
    assert buffer._event_ids == set()
    assert buffer._fingerprints == set()
    assert buffer.remaining_capacity == 2


def test_buffer_rejects_max_events_below_batch_size():
    with pytest.raises(ValueError, match="max_events"):
        MicroBatchBuffer(batch_size=5, batch_window_seconds=1.0, max_events=2)


def test_buffer_fingerprint_dedupe_can_be_disabled():
    buffer = MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0)
    buffer.configure_fingerprint_dedupe(False, generated_only=True)
    first = normalize_event(event_payload(event_id="a", item_id="same"))
    second = normalize_event(event_payload(event_id="b", item_id="same"))
    result = buffer.extend([first, second])
    assert result.kept_count == 2
    assert result.duplicates == ()


def test_buffer_fingerprint_dedupe_generated_only():
    buffer = MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0)
    buffer.configure_fingerprint_dedupe(True, generated_only=True)
    payload = event_payload(item_id="same")
    payload.pop("event_id")
    generated = normalize_event(payload)
    explicit = normalize_event(event_payload(event_id="explicit", item_id="same"))
    generated_dup = normalize_event(payload)
    result = buffer.extend([generated, explicit, generated_dup])
    assert [event.event_id for event in result.kept] == [generated.event_id, "explicit"]
    assert [event.event_id for event in result.duplicates] == [generated_dup.event_id]
