"""Unit tests for support.system_db helpers (no live Postgres required)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from support.postgres_defaults import postgres_test_db
from support.system_db import is_dedicated_test_database, postgres_ready, reset_schema

from cicerone.io.db_store import DEFAULT_DB_TABLES


@pytest.mark.parametrize(
    ("db_name", "expected"),
    [
        ("test_db", True),
        ("foo_test", True),
        ("test_", True),
        ("_test", True),
        (None, False),
        ("", False),
        ("production", False),
        ("staging", False),
        ("dev", False),
        ("foo_test_backup", False),
        ("pretest_db", False),
    ],
)
def test_is_dedicated_test_database_classification(db_name: str | None, expected: bool) -> None:
    assert is_dedicated_test_database(db_name) is expected


def test_is_dedicated_test_database_accepts_canonical_postgres_test_db() -> None:
    assert is_dedicated_test_database(postgres_test_db()) is True


def test_is_dedicated_test_database_ignores_postgres_test_db_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POSTGRES_TEST_DB=cicerone must not authorize schema reset on the app DB."""
    monkeypatch.setenv("POSTGRES_TEST_DB", "cicerone")
    assert is_dedicated_test_database("cicerone") is False
    with pytest.raises(ValueError, match="dedicated test database"):
        postgres_test_db()


def test_reset_schema_rejects_non_test_database_names() -> None:
    for db_name in ("prod", "analytics"):
        fake_engine = SimpleNamespace(url=SimpleNamespace(database=db_name))
        with pytest.raises(RuntimeError, match="Refusing to reset schema for non-test database"):
            reset_schema(fake_engine)  # type: ignore[arg-type]


def test_reset_schema_refusal_not_masked_by_bad_postgres_test_db_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hostile POSTGRES_TEST_DB must not replace the non-test-DB RuntimeError."""
    monkeypatch.setenv("POSTGRES_TEST_DB", "cicerone")
    fake_engine = SimpleNamespace(url=SimpleNamespace(database="cicerone"))
    with pytest.raises(RuntimeError, match="Refusing to reset schema for non-test database"):
        reset_schema(fake_engine)  # type: ignore[arg-type]


def test_reset_schema_requires_allow_schema_reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_engine = SimpleNamespace(url=SimpleNamespace(database=postgres_test_db()))

    monkeypatch.delenv("ALLOW_SCHEMA_RESET_FOR_TESTS", raising=False)
    with pytest.raises(RuntimeError, match="ALLOW_SCHEMA_RESET_FOR_TESTS"):
        reset_schema(fake_engine)  # type: ignore[arg-type]

    monkeypatch.setenv("ALLOW_SCHEMA_RESET_FOR_TESTS", "0")
    with pytest.raises(RuntimeError, match="ALLOW_SCHEMA_RESET_FOR_TESTS"):
        reset_schema(fake_engine)  # type: ignore[arg-type]


def test_reset_schema_drops_only_cicerone_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTable:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeMetaData:
        def __init__(self) -> None:
            self.tables = {name: FakeTable(name) for name in DEFAULT_DB_TABLES}
            self.tables["unrelated_table"] = FakeTable("unrelated_table")
            self.reflected = False
            self.dropped_names: list[str] = []

        def reflect(self, bind=None) -> None:
            self.reflected = True

        def remove(self, table: FakeTable) -> None:
            self.tables.pop(table.name, None)

        def drop_all(self, bind=None) -> None:
            self.dropped_names = list(self.tables)

    fake_metadata = FakeMetaData()
    monkeypatch.setenv("ALLOW_SCHEMA_RESET_FOR_TESTS", "1")
    monkeypatch.setattr("support.system_db.MetaData", lambda: fake_metadata)

    fake_engine = SimpleNamespace(url=SimpleNamespace(database=postgres_test_db()))
    reset_schema(fake_engine)  # type: ignore[arg-type]

    assert fake_metadata.reflected is True
    assert set(fake_metadata.dropped_names) == set(DEFAULT_DB_TABLES)
    assert "unrelated_table" not in fake_metadata.dropped_names


def test_postgres_ready_normalizes_arrays_tuples_and_scalars() -> None:
    fixture = pd.DataFrame(
        {
            "array_col": [np.array([1, 2]), np.array([3, 4])],
            "tuple_col": [(5, 6), (7, 8)],
            "scalar_col": [9, 10],
        }
    )
    assert isinstance(fixture.loc[0, "array_col"], np.ndarray)
    assert isinstance(fixture.loc[0, "tuple_col"], tuple)

    ready = postgres_ready(fixture)

    assert ready.loc[0, "array_col"] == [1, 2]
    assert ready.loc[1, "array_col"] == [3, 4]
    assert isinstance(ready.loc[0, "array_col"], list)
    assert ready.loc[0, "tuple_col"] == [5, 6]
    assert ready.loc[1, "tuple_col"] == [7, 8]
    assert isinstance(ready.loc[0, "tuple_col"], list)
    assert ready.loc[0, "scalar_col"] == 9
    assert ready.loc[1, "scalar_col"] == 10
    assert ready["array_col"].dtype == object
    assert ready["tuple_col"].dtype == object
