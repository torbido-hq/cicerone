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


def record_durable_write_steps(monkeypatch) -> list[str]:
    """Record fsync(file) / replace / fsync(dir) order of state-file writes."""
    import os
    import stat
    from pathlib import Path

    steps: list[str] = []
    real_fsync = os.fsync
    real_replace = Path.replace

    def _fsync(fd: int) -> None:
        steps.append("fsync_dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "fsync_file")
        real_fsync(fd)

    def _replace(self: Path, target):
        steps.append("replace")
        return real_replace(self, target)

    monkeypatch.setattr(os, "fsync", _fsync)
    monkeypatch.setattr(Path, "replace", _replace)
    return steps
