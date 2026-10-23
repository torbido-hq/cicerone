import json
import os

from cicerone.events.state_file import write_json_durably


def test_write_json_durably_overlapping_writers_do_not_share_temp_file(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    real_fsync = os.fsync
    nested = []

    def fsync_then_write_again(fd: int) -> None:
        real_fsync(fd)
        if not nested:
            nested.append(True)
            write_json_durably(target, {"writer": 2})

    monkeypatch.setattr(os, "fsync", fsync_then_write_again)
    write_json_durably(target, {"writer": 1})

    assert json.loads(target.read_text()) == {"writer": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
