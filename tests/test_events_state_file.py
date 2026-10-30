from __future__ import annotations

import os
import stat
from types import SimpleNamespace

from cicerone.events import state_file
from cicerone.events.state_file import read_json_object_file, write_json_object_file


def test_write_json_object_file_syncs_the_directory_after_the_rename(tmp_path, monkeypatch) -> None:
    target = tmp_path / "positions" / "marker.json"
    synced: list[tuple[bool, bool, int]] = []

    def fsync(fd: int) -> None:
        info = os.fstat(fd)
        synced.append((stat.S_ISDIR(info.st_mode), target.exists(), info.st_ino))

    monkeypatch.setattr(
        state_file, "os", SimpleNamespace(fsync=fsync, open=os.open, close=os.close, O_RDONLY=os.O_RDONLY)
    )
    write_json_object_file(target, {"key": "a"})
    # The temp file before the rename, then its directory, then the new directory's parent.
    assert [entry[:2] for entry in synced] == [(False, False), (True, True), (True, True)]
    assert [entry[2] for entry in synced[1:]] == [target.parent.stat().st_ino, tmp_path.stat().st_ino]
    assert read_json_object_file(target) == {"key": "a"}
    assert not (target.parent / ".marker.json.tmp").exists()
