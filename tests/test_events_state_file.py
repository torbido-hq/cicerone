import json
import os
import stat

import pytest

from cicerone.events.state_file import write_json_durably


def test_write_json_durably_raises_when_directory_open_fails(tmp_path, monkeypatch):
    target = tmp_path / "state.json"

    def fail_open(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "open", fail_open)
    with pytest.raises(PermissionError):
        write_json_durably(target, {"key": "a"})
    assert json.loads(target.read_text()) == {"key": "a"}


def test_write_json_durably_raises_when_directory_fsync_fails(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    real_fsync = os.fsync

    def fail_dir_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_dir_fsync)
    with pytest.raises(OSError, match="fsync failed"):
        write_json_durably(target, {"key": "a"})
    assert json.loads(target.read_text()) == {"key": "a"}


def test_write_json_durably_keeps_umask_permissions(tmp_path):
    target = tmp_path / "state.json"
    previous = os.umask(0o027)
    try:
        write_json_durably(target, {"key": "a"})
    finally:
        os.umask(previous)

    assert json.loads(target.read_text()) == {"key": "a"}
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
