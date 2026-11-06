"""Canonical CI MySQL defaults for tests, read from ``docker/mysql/defaults.env``.

Set ``MYSQL_TEST_HOST`` to run the MySQL side of the track store tests.
"""

from __future__ import annotations

import os
from pathlib import Path

from support.postgres_defaults import load_env_defaults, require_test_database_url

_DEFAULTS_PATH = Path(__file__).resolve().parents[2] / "docker" / "mysql" / "defaults.env"
_REQUIRED_KEYS = ("MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE", "MYSQL_PORT")


def resolve_mysql_test_database_url() -> str | None:
    host = os.environ.get("MYSQL_TEST_HOST")
    if not host:
        return None
    env = load_env_defaults(_DEFAULTS_PATH, _REQUIRED_KEYS)
    return require_test_database_url(
        f"mysql+pymysql://{env['MYSQL_USER']}:{env['MYSQL_PASSWORD']}"
        f"@{host}:{env['MYSQL_PORT']}/{env['MYSQL_DATABASE']}"
    )
