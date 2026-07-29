from __future__ import annotations

from cicerone.dashboard_users import load_users, save_users


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


def test_load_users_missing_users_table_returns_empty_dict(tmp_path):
    path = tmp_path / "dashboard_users.toml"
    path.write_text("# no [users] table here\n")

    assert load_users(path) == {}
