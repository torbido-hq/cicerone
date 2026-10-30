from __future__ import annotations

import os
import stat
from types import SimpleNamespace

from cicerone.events import state_file
from cicerone.events.state_file import read_json_object_file, write_json_object_file


def test_write_json_object_file_syncs_the_directory_after_the_rename(tmp_path, monkeypatch) -> None:
    target = tmp_path / "positions" / "marker.json"
    synced: list[tuple[bool, bool]] = []

    def fsync(fd: int) -> None:
        synced.append((stat.S_ISDIR(os.fstat(fd).st_mode), target.exists()))

    monkeypatch.setattr(
        state_file, "os", SimpleNamespace(fsync=fsync, open=os.open, close=os.close, O_RDONLY=os.O_RDONLY)
    )
    write_json_object_file(target, {"key": "a"})
    # The temp file before the rename, then the directory after it.
    assert synced == [(False, False), (True, True)]
    assert read_json_object_file(target) == {"key": "a"}
    assert not (target.parent / ".marker.json.tmp").exists()
