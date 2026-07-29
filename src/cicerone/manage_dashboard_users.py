"""CLI to manage the dashboard's HTTP Basic Auth users (see
cicerone.dashboard_users for the file format/loader cicerone.dashboard
reads at startup). Intended for a small, fixed set of named people (a
handful of maintainers) -- not a general user management system, so it
deliberately does nothing more than add/remove/list against that one file.

Usage:
  python -m cicerone.manage_dashboard_users add <username> [--users-path PATH]
  python -m cicerone.manage_dashboard_users remove <username> [--users-path PATH]
  python -m cicerone.manage_dashboard_users list [--users-path PATH]

Passwords are always read interactively via getpass (never as a CLI
argument), so they never end up in shell history or a `ps` listing.
"""

from __future__ import annotations

import argparse
import getpass
import re
import sys

import bcrypt

from cicerone.dashboard_users import load_users, save_users

DEFAULT_USERS_PATH = "/app/config/dashboard_users.toml"

# Usernames are stored as bare TOML keys, restricted to what TOML allows there.
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_username(username: str) -> None:
    if not _USERNAME_RE.match(username):
        raise SystemExit(f"Invalid username {username!r}: only letters, digits, '_' and '-' are allowed.")


def _cmd_add(args: argparse.Namespace) -> None:
    _validate_username(args.username)
    password = getpass.getpass("Password: ")
    if not password:
        raise SystemExit("Password must not be empty.")
    if getpass.getpass("Confirm password: ") != password:
        raise SystemExit("Passwords do not match.")

    users = load_users(args.users_path)
    action = "Updated" if args.username in users else "Added"
    users[args.username] = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")
    save_users(args.users_path, users)
    print(f"{action} user {args.username!r} in {args.users_path}")


def _cmd_remove(args: argparse.Namespace) -> None:
    users = load_users(args.users_path)
    if args.username not in users:
        raise SystemExit(f"No such user: {args.username!r}")
    del users[args.username]
    save_users(args.users_path, users)
    print(f"Removed user {args.username!r} from {args.users_path}")


def _cmd_list(args: argparse.Namespace) -> None:
    users = load_users(args.users_path)
    if not users:
        print(f"No users configured in {args.users_path}")
        return
    for username in sorted(users):
        print(username)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manage cicerone dashboard Basic Auth users")
    parser.add_argument(
        "--users-path",
        default=DEFAULT_USERS_PATH,
        help=f"Path to the users TOML file (default: {DEFAULT_USERS_PATH})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add or update a user (prompts for password)")
    add_parser.add_argument("username")
    add_parser.set_defaults(func=_cmd_add)

    remove_parser = subparsers.add_parser("remove", help="Remove a user")
    remove_parser.add_argument("username")
    remove_parser.set_defaults(func=_cmd_remove)

    list_parser = subparsers.add_parser("list", help="List configured usernames")
    list_parser.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
