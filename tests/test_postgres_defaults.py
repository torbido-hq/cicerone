"""Unit tests for tests.postgres_defaults (no live Postgres required)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from postgres_defaults import (
    build_test_database_url,
    load_postgres_defaults,
    postgres_port_for_host,
    postgres_test_db,
    resolve_test_database_url,
)


def test_load_postgres_defaults_matches_env_file_keys():
    defaults = load_postgres_defaults()
    assert defaults["POSTGRES_TEST_DB"]
    assert defaults["POSTGRES_USER"]
    assert defaults["POSTGRES_PASSWORD"]
    assert defaults["POSTGRES_PORT"]
    assert defaults["POSTGRES_HOST_PORT"]


def test_postgres_test_db_reads_defaults_file(monkeypatch):
    monkeypatch.delenv("POSTGRES_TEST_DB", raising=False)
    assert postgres_test_db() == load_postgres_defaults()["POSTGRES_TEST_DB"]


def test_postgres_test_db_prefers_env_override(monkeypatch):
    monkeypatch.setenv("POSTGRES_TEST_DB", "custom_test")
    assert postgres_test_db() == "custom_test"


def test_build_test_database_url_uses_host_port_for_localhost(monkeypatch):
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.delenv("POSTGRES_HOST_PORT", raising=False)
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    monkeypatch.delenv("POSTGRES_TEST_DB", raising=False)
    defaults = load_postgres_defaults()
    url = build_test_database_url("localhost")
    assert url == (
        f"postgresql+psycopg://{defaults['POSTGRES_USER']}:"
        f"{defaults['POSTGRES_PASSWORD']}@localhost:"
        f"{defaults['POSTGRES_HOST_PORT']}/{defaults['POSTGRES_TEST_DB']}"
    )


def test_build_test_database_url_uses_container_port_for_compose_hosts(monkeypatch):
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.delenv("POSTGRES_HOST_PORT", raising=False)
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    monkeypatch.delenv("POSTGRES_TEST_DB", raising=False)
    monkeypatch.setenv("POSTGRES_HOST_PORT", "15432")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    defaults = load_postgres_defaults()
    url = build_test_database_url("db-test")
    assert url == (
        f"postgresql+psycopg://{defaults['POSTGRES_USER']}:"
        f"{defaults['POSTGRES_PASSWORD']}@db-test:5432/"
        f"{defaults['POSTGRES_TEST_DB']}"
    )
    assert postgres_port_for_host("localhost") == "15432"
    assert postgres_port_for_host("postgres") == "5432"


def test_resolve_prefers_explicit_test_database_url(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+psycopg://u:p@h:1/db")
    monkeypatch.setenv("POSTGRES_TEST_HOST", "ignored")
    assert resolve_test_database_url() == "postgresql+psycopg://u:p@h:1/db"


def test_resolve_builds_from_postgres_test_host(monkeypatch):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_TEST_HOST", "db-test")
    assert resolve_test_database_url() == build_test_database_url("db-test")


def test_resolve_returns_none_without_url_or_host(monkeypatch):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_TEST_HOST", raising=False)
    assert resolve_test_database_url() is None


@pytest.mark.parametrize("host", ["localhost", "db-test", "postgres"])
def test_shell_helper_matches_python_builder(host):
    """Keep docker/postgres/test-database-url.sh in sync with the Python helper."""
    script = Path(__file__).resolve().parents[1] / "docker" / "postgres" / "test-database-url.sh"
    if not os.access(script, os.X_OK):
        pytest.skip(f"{script} is not executable")
    result = subprocess.run(
        [str(script), host],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == build_test_database_url(host)
