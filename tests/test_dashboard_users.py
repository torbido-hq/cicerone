from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from cicerone.dashboard_users import (
    _restrict_owner_only,
    _restrict_windows_acl,
    load_users,
    save_users,
)


def test_load_users_missing_file_returns_empty_dict(tmp_path):
    assert load_users(tmp_path / "nope.toml") == {}


def test_save_then_load_users_round_trips(tmp_path):
    path = tmp_path / "dashboard_users.toml"
    users = {"alice": "hash-a", "bob": "hash-b"}

    save_users(path, users)

    assert load_users(path) == users


def test_save_users_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dashboard_users.toml"

    save_users(path, {"alice": "hash-a"})

    assert load_users(path) == {"alice": "hash-a"}


def test_save_users_sets_owner_only_mode(tmp_path):
    path = tmp_path / "dashboard_users.toml"
    path.write_text("stale\n")
    path.chmod(0o644)

    save_users(path, {"alice": "hash-a"})

    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_users(path) == {"alice": "hash-a"}


def test_save_users_keeps_0600_when_umask_is_zero(tmp_path):
    path = tmp_path / "dashboard_users.toml"
    previous = os.umask(0)
    try:
        save_users(path, {"alice": "hash-a"})
    finally:
        os.umask(previous)
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_save_users_does_not_truncate_destination(tmp_path, monkeypatch):
    path = tmp_path / "dashboard_users.toml"
    path.write_text("stale-secret\n")
    original_open = os.open

    def guarded(name: str | os.PathLike[str], flags: int, *args: object, **kwargs: object) -> int:
        if Path(name).resolve() == path.resolve() and flags & os.O_TRUNC:
            raise AssertionError("truncated destination")
        return original_open(name, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", guarded)
    save_users(path, {"alice": "hash-a"})
    assert "stale-secret" not in path.read_text()
    assert load_users(path) == {"alice": "hash-a"}


def test_save_users_closes_fd_if_fdopen_fails(tmp_path, monkeypatch):
    path = tmp_path / "dashboard_users.toml"
    opened: list[int] = []
    closed: list[int] = []
    real_open = os.open
    real_close = os.close

    def tracked_open(name: str | os.PathLike[str], flags: int, *args: object, **kwargs: object) -> int:
        fd = real_open(name, flags, *args, **kwargs)  # type: ignore[arg-type]
        opened.append(fd)
        return fd

    def failing_fdopen(_fd: int, *_args: object, **_kwargs: object) -> object:
        raise OSError("fdopen failed")

    def tracked_close(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "fdopen", failing_fdopen)
    monkeypatch.setattr(os, "close", tracked_close)

    with pytest.raises(OSError, match="fdopen failed"):
        save_users(path, {"alice": "hash-a"})
    assert opened
    assert closed == opened
    assert not path.exists()


def test_restrict_owner_only_dispatches_to_windows_acl(tmp_path, monkeypatch):
    path = tmp_path / "users.toml"
    path.write_text("x")
    seen: list[Path] = []
    monkeypatch.setattr("cicerone.dashboard_users.os", type("O", (), {"name": "nt"})())
    monkeypatch.setattr("cicerone.dashboard_users._restrict_windows_acl", seen.append)
    _restrict_owner_only(path)
    assert seen == [path]


class _FakeKernel32:
    def GetCurrentProcess(self) -> int:
        return 1

    def CloseHandle(self, _token: object) -> int:
        return 1


class _FakeAdvapi32:
    def __init__(
        self,
        *,
        open_ok: bool = True,
        info_ok: bool = True,
        init_ok: bool = True,
        ace_ok: bool = True,
        set_status: int = 0,
    ) -> None:
        self.open_ok = open_ok
        self.info_ok = info_ok
        self.init_ok = init_ok
        self.ace_ok = ace_ok
        self.set_status = set_status
        self.set_info: int | None = None
        self.set_path: str | None = None

    def OpenProcessToken(self, _proc: object, _access: int, _token: object) -> bool:
        return self.open_ok

    def GetTokenInformation(
        self, _token: object, _cls: int, buf: object, _buflen: object, needed: object
    ) -> bool:
        needed._obj.value = 64
        return True if buf is None else self.info_ok

    def GetLengthSid(self, _sid: object) -> int:
        return 12

    def InitializeAcl(self, _acl: object, _size: int, _rev: int) -> bool:
        return self.init_ok

    def AddAccessAllowedAce(self, _acl: object, _rev: int, _mask: int, _sid: object) -> bool:
        return self.ace_ok

    def SetNamedSecurityInfoW(
        self,
        name: str,
        _kind: int,
        info: int,
        _owner: object,
        _group: object,
        _dacl: object,
        _sacl: object,
    ) -> int:
        self.set_path = name
        self.set_info = info
        return self.set_status


def test_restrict_windows_acl_sets_protected_owner_dacl(tmp_path):
    path = tmp_path / "users.toml"
    path.write_text("x")
    advapi32 = _FakeAdvapi32()
    _restrict_windows_acl(path, kernel32=_FakeKernel32(), advapi32=advapi32)
    assert advapi32.set_path == str(path)
    assert advapi32.set_info == 0x00000004 | 0x80000000


def test_restrict_windows_acl_raises_on_token_or_set_failure(tmp_path):
    path = tmp_path / "users.toml"
    path.write_text("x")
    with pytest.raises(OSError, match="OpenProcessToken"):
        _restrict_windows_acl(path, kernel32=_FakeKernel32(), advapi32=_FakeAdvapi32(open_ok=False))
    with pytest.raises(OSError, match="GetTokenInformation"):
        _restrict_windows_acl(path, kernel32=_FakeKernel32(), advapi32=_FakeAdvapi32(info_ok=False))
    with pytest.raises(OSError, match="SetNamedSecurityInfoW"):
        _restrict_windows_acl(path, kernel32=_FakeKernel32(), advapi32=_FakeAdvapi32(set_status=5))
    with pytest.raises(OSError, match="InitializeAcl"):
        _restrict_windows_acl(path, kernel32=_FakeKernel32(), advapi32=_FakeAdvapi32(init_ok=False))
    with pytest.raises(OSError, match="AddAccessAllowedAce"):
        _restrict_windows_acl(path, kernel32=_FakeKernel32(), advapi32=_FakeAdvapi32(ace_ok=False))


def test_load_users_missing_users_table_returns_empty_dict(tmp_path):
    path = tmp_path / "dashboard_users.toml"
    path.write_text("# no [users] table here\n")

    assert load_users(path) == {}
