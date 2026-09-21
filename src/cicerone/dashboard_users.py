"""Dashboard Basic Auth users file (username → bcrypt hash)."""

from __future__ import annotations

import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any

_OWNER_ONLY = 0o600


def load_users(path: str | Path) -> dict[str, str]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open("rb") as f:
        raw = tomllib.load(f)
    return dict(raw.get("users", {}))


def save_users(path: str | Path, users: dict[str, str]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Managed by `cicerone users` -- do not edit by hand.",
        "[users]",
    ]
    for username in sorted(users):
        lines.append(f'{username} = "{users[username]}"')
    text = "\n".join(lines) + "\n"
    with tempfile.TemporaryDirectory(dir=file_path.parent) as temp_dir:
        temp_path = Path(temp_dir) / file_path.name
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _OWNER_ONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        _restrict_owner_only(temp_path)
        temp_path.replace(file_path)
    _restrict_owner_only(file_path)


def _restrict_owner_only(path: Path) -> None:
    if os.name == "nt":
        _restrict_windows_acl(path)
        return
    path.chmod(_OWNER_ONLY)


def _restrict_windows_acl(
    path: Path,
    *,
    kernel32: Any | None = None,
    advapi32: Any | None = None,
) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = kernel32 or ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    advapi32 = advapi32 or ctypes.WinDLL("advapi32", use_last_error=True)  # type: ignore[attr-defined]
    last_error = getattr(ctypes, "get_last_error", lambda: 1)

    class SID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

    class TOKEN_USER(ctypes.Structure):
        _fields_ = [("User", SID_AND_ATTRIBUTES)]

    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        raise OSError(last_error() or 1, "OpenProcessToken")
    try:
        needed = wintypes.DWORD(0)
        advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
        buf = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetTokenInformation(token, 1, buf, needed, ctypes.byref(needed)):
            raise OSError(last_error() or 1, "GetTokenInformation")
        sid = TOKEN_USER.from_buffer(buf).User.Sid
        ace = 8 + int(advapi32.GetLengthSid(sid))
        acl = ctypes.create_string_buffer(8 + ace)
        if not advapi32.InitializeAcl(acl, 8 + ace, 2):
            raise OSError(last_error() or 1, "InitializeAcl")
        if not advapi32.AddAccessAllowedAce(acl, 2, 0x1F01FF, sid):
            raise OSError(last_error() or 1, "AddAccessAllowedAce")
        status = advapi32.SetNamedSecurityInfoW(
            str(path),
            1,
            0x00000004 | 0x80000000,
            None,
            None,
            acl,
            None,
        )
        if status:
            raise OSError(status, "SetNamedSecurityInfoW")
    finally:
        kernel32.CloseHandle(token)
