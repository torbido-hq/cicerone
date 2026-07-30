"""Canonical local/CI Postgres defaults for tests.

Reads ``docker/postgres/defaults.env`` (same file compose/CI use) so the
pytest DB name and ``TEST_DATABASE_URL`` assembly cannot drift from that
source. Prefer setting ``POSTGRES_TEST_HOST`` (and optionally the
``POSTGRES_*`` vars) rather than hand-building the URL:

- host / venv: ``POSTGRES_TEST_HOST=localhost`` (uses ``POSTGRES_HOST_PORT``)
- compose CI:  ``POSTGRES_TEST_HOST=db-test`` (uses container ``POSTGRES_PORT``)

``TEST_DATABASE_URL``, when set, still wins (explicit override).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "docker" / "postgres" / "defaults.env"

_REQUIRED_KEYS = (
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_TEST_DB",
    "POSTGRES_PORT",
    "POSTGRES_HOST_PORT",
)

# Hostnames that reach Postgres via the published host port map.
_HOST_SIDE_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@lru_cache(maxsize=1)
def load_postgres_defaults() -> dict[str, str]:
    """Parse docker/postgres/defaults.env into a dict (no shell expansion)."""
    if not _DEFAULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing Postgres defaults file at {_DEFAULTS_PATH}. "
            "It is the canonical source for local/CI DB credentials."
        )
    values: dict[str, str] = {}
    for raw_line in _DEFAULTS_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    missing = [key for key in _REQUIRED_KEYS if key not in values]
    if missing:
        raise ValueError(f"{_DEFAULTS_PATH} missing required keys: {missing}")
    return values


def _default(key: str) -> str:
    return os.environ.get(key) or load_postgres_defaults()[key]


def looks_like_test_database(db_name: str | None) -> bool:
    """True when ``db_name`` follows the dedicated-test naming convention.

    Schema-reset guardrails use this pattern only — never an env-overridable
    exact name — so ``POSTGRES_TEST_DB=cicerone`` cannot authorize wiping the
    app database.
    """
    if not db_name:
        return False
    return db_name.endswith("_test") or db_name.startswith("test_")


def canonical_postgres_test_db() -> str:
    """``POSTGRES_TEST_DB`` from defaults.env only (ignores process env)."""
    return load_postgres_defaults()["POSTGRES_TEST_DB"]


def postgres_test_db() -> str:
    """Pytest database name (``POSTGRES_TEST_DB``, env override allowed).

    The resolved name must still look like a test database so an override
    cannot silently target the app DB (see ``looks_like_test_database``).
    """
    name = _default("POSTGRES_TEST_DB")
    if not looks_like_test_database(name):
        raise ValueError(
            f"POSTGRES_TEST_DB must look like a dedicated test database "
            f"(start with 'test_' or end with '_test'), got {name!r}"
        )
    return name


def postgres_port_for_host(host: str) -> str:
    """Port for ``host``: published map for localhost, container port otherwise."""
    if host in _HOST_SIDE_HOSTS:
        return _default("POSTGRES_HOST_PORT")
    return _default("POSTGRES_PORT")


def build_test_database_url(host: str) -> str:
    """Assemble ``TEST_DATABASE_URL`` for ``host`` from canonical defaults."""
    user = _default("POSTGRES_USER")
    password = _default("POSTGRES_PASSWORD")
    port = postgres_port_for_host(host)
    database = postgres_test_db()
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"


def resolve_test_database_url() -> str | None:
    """Resolve the DB URL for pytest: explicit URL, else host + defaults."""
    if url := os.environ.get("TEST_DATABASE_URL"):
        return url
    if host := os.environ.get("POSTGRES_TEST_HOST"):
        return build_test_database_url(host)
    return None
