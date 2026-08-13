"""Shared helpers for events package tests."""

from __future__ import annotations


def event_payload(**overrides) -> dict:
    base = {
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "purchase",
        "quantity": 1,
        "occurred_at": "2026-08-13T12:00:00Z",
        "event_id": "e1",
    }
    base.update(overrides)
    return base
