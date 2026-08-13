from __future__ import annotations

import pytest
from support.events import event_payload

from cicerone.events.buffer import MicroBatchBuffer
from cicerone.events.normalize import normalize_event


def test_micro_batch_buffer_count_and_dedupe():
    buffer = MicroBatchBuffer(batch_size=2, batch_window_seconds=60.0)
    e1 = normalize_event(event_payload(event_id="1"))
    e2 = normalize_event(event_payload(event_id="dup"))  # same fingerprint as e1
    e3 = normalize_event(event_payload(event_id="3", item_id="i2"))
    assert buffer.extend([e1, e2]) == 1
    assert buffer.ready() is False
    assert buffer.extend([e3]) == 1
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


def test_buffer_dedupes_by_event_id():
    buffer = MicroBatchBuffer(batch_size=10, batch_window_seconds=60.0)
    first = normalize_event(event_payload(event_id="id-1", item_id="a"))
    second = normalize_event(event_payload(event_id="id-1", item_id="b"))
    assert buffer.extend([first, second]) == 1


def test_buffer_caps_events_and_clears_dedupe_on_flush():
    buffer = MicroBatchBuffer(batch_size=2, batch_window_seconds=60.0, max_events=2)
    events = [normalize_event(event_payload(event_id=f"e{i}", item_id=f"i{i}")) for i in range(4)]
    assert buffer.extend(events) == 2
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
