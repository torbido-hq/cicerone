from __future__ import annotations

import bcrypt
import pytest

from cicerone.dashboard_users import load_users
from cicerone.manage_dashboard_users import main


def test_add_user_prompts_for_password_and_stores_bcrypt_hash(tmp_path, monkeypatch):
    users_path = tmp_path / "dashboard_users.toml"
    passwords = iter(["s3cret", "s3cret"])
    monkeypatch.setattr("getpass.getpass", lambda *_args, **_kwargs: next(passwords))

    main(["--users-path", str(users_path), "add", "alice"])

    users = load_users(users_path)
    assert set(users) == {"alice"}
    assert bcrypt.checkpw(b"s3cret", users["alice"].encode("ascii"))


def test_add_user_rejects_mismatched_confirmation(tmp_path, monkeypatch):
    users_path = tmp_path / "dashboard_users.toml"
    passwords = iter(["s3cret", "different"])
    monkeypatch.setattr("getpass.getpass", lambda *_args, **_kwargs: next(passwords))

    with pytest.raises(SystemExit, match="do not match"):
        main(["--users-path", str(users_path), "add", "alice"])


def test_add_user_rejects_empty_password(tmp_path, monkeypatch):
    users_path = tmp_path / "dashboard_users.toml"
    monkeypatch.setattr("getpass.getpass", lambda *_args, **_kwargs: "")

    with pytest.raises(SystemExit, match="must not be empty"):
        main(["--users-path", str(users_path), "add", "alice"])


def test_add_user_rejects_invalid_username(tmp_path):
    with pytest.raises(SystemExit, match="Invalid username"):
        main(["--users-path", str(tmp_path / "u.toml"), "add", "not a valid name"])


def test_add_user_updates_existing_user(tmp_path, monkeypatch, capsys):
    users_path = tmp_path / "dashboard_users.toml"
    passwords = iter(["first-pw", "first-pw", "second-pw", "second-pw"])
    monkeypatch.setattr("getpass.getpass", lambda *_args, **_kwargs: next(passwords))

    main(["--users-path", str(users_path), "add", "alice"])
    capsys.readouterr()
    main(["--users-path", str(users_path), "add", "alice"])

    assert "Updated user 'alice'" in capsys.readouterr().out
    users = load_users(users_path)
    assert bcrypt.checkpw(b"second-pw", users["alice"].encode("ascii"))


def test_remove_user_removes_from_file(tmp_path, monkeypatch, capsys):
    users_path = tmp_path / "dashboard_users.toml"
    monkeypatch.setattr("getpass.getpass", lambda *_args, **_kwargs: "s3cret")
    main(["--users-path", str(users_path), "add", "alice"])

    main(["--users-path", str(users_path), "remove", "alice"])

    assert "Removed user 'alice'" in capsys.readouterr().out
    assert load_users(users_path) == {}


def test_remove_unknown_user_raises(tmp_path):
    with pytest.raises(SystemExit, match="No such user"):
        main(["--users-path", str(tmp_path / "u.toml"), "remove", "ghost"])


def test_list_users_prints_usernames_sorted(tmp_path, monkeypatch, capsys):
    users_path = tmp_path / "dashboard_users.toml"
    monkeypatch.setattr("getpass.getpass", lambda *_args, **_kwargs: "s3cret")
    main(["--users-path", str(users_path), "add", "bob"])
    main(["--users-path", str(users_path), "add", "alice"])
    capsys.readouterr()

    main(["--users-path", str(users_path), "list"])

    assert capsys.readouterr().out.splitlines() == ["alice", "bob"]


def test_list_users_empty_prints_a_message(tmp_path, capsys):
    users_path = tmp_path / "dashboard_users.toml"

    main(["--users-path", str(users_path), "list"])

    assert "No users configured" in capsys.readouterr().out
