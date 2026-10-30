from __future__ import annotations

import os
from types import SimpleNamespace

from cicerone.events import state_file
from cicerone.events.state_file import read_json_object_file, write_json_object_file


def test_write_json_object_file_syncs_the_directory_after_the_rename(tmp_path, monkeypatch) -> None:
    target = tmp_path / "positions" / "marker.json"
    target_existed_at_sync: list[bool] = []

    def fsync(_fd: int) -> None:
        target_existed_at_sync.append(target.exists())

    monkeypatch.setattr(
        state_file, "os", SimpleNamespace(fsync=fsync, open=os.open, close=os.close, O_RDONLY=os.O_RDONLY)
    )
    write_json_object_file(target, {"key": "a"})
    assert target_existed_at_sync == [False, True]
    assert read_json_object_file(target) == {"key": "a"}
    assert not (target.parent / ".marker.json.tmp").exists()
