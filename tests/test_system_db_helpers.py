"""Unit tests for support.system_db helpers (no live Postgres required)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from support.postgres_defaults import postgres_test_db
from support.system_db import (
    REPO_FEATURES_CONFIG,
    SYSTEM_SERVE_TOKEN,
    available_recommendation_ids,
    dashboard_users,
    is_dedicated_test_database,
    postgres_ready,
    reset_schema,
    sample_system_catalog,
    write_system_config,
)

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


def test_sample_system_catalog_has_filter_columns() -> None:
    events, users, items = sample_system_catalog()
    assert set(events["user_id"]) == {"u1", "u2", "u3"}
    assert "u4" in set(users["user_id"])
    assert set(items["item_id"]) == {"i1", "i2", "i3", "i4"}
    wine = items.loc[items["item_id"] == "i3"].iloc[0]
    assert wine["category"] == "wine"
    assert bool(wine["in_stock"]) is False
    unpublished = items.loc[items["item_id"] == "i4"].iloc[0]
    assert bool(unpublished["published"]) is False


def test_write_system_config_enables_serve_dashboard_track_eval(tmp_path) -> None:
    import tomllib

    path = write_system_config(tmp_path / "cicerone.toml", database_url="postgresql://example/cicerone_test")
    raw = tomllib.loads(path.read_text())
    assert raw["job"]["save_model_artifact"] is True
    assert raw["job"]["eval"]["enabled"] is True
    assert raw["job"]["feature_config_path"] == str(REPO_FEATURES_CONFIG)
    assert raw["input"]["kind"] == "db"
    assert raw["output"]["options"]["database_url"] == "postgresql://example/cicerone_test"
    assert raw["serve"]["auth_token"] == SYSTEM_SERVE_TOKEN
    assert raw["serve"]["category_column"] == "category"
    assert raw["dashboard"]["enabled"] is True
    assert raw["track"]["enabled"] is True


def test_write_system_config_mixes_db_input_and_dataset_output(tmp_path) -> None:
    import tomllib

    output_path = tmp_path / "out"
    path = write_system_config(
        tmp_path / "cicerone.toml",
        input_kind="db",
        output_kind="dataset",
        database_url="postgresql://example/cicerone_test",
        output_path=output_path,
    )
    raw = tomllib.loads(path.read_text())
    assert raw["input"]["kind"] == "db"
    assert raw["input"]["options"]["database_url"] == "postgresql://example/cicerone_test"
    assert "path" not in raw["input"]["options"]
    assert raw["output"]["kind"] == "dataset"
    assert raw["output"]["options"]["storage_backend"] == "local"
    assert raw["output"]["options"]["path"] == str(output_path)
    assert "database_url" not in raw["output"]["options"]


def test_write_system_config_dataset_keeps_input_and_output_trees_apart(tmp_path) -> None:
    import tomllib

    input_path = tmp_path / "in"
    output_path = tmp_path / "out"
    path = write_system_config(
        tmp_path / "cicerone.toml",
        kind="dataset",
        input_path=input_path,
        output_path=output_path,
    )
    raw = tomllib.loads(path.read_text())
    assert raw["input"]["kind"] == "dataset"
    assert raw["output"]["kind"] == "dataset"
    assert raw["input"]["options"]["storage_backend"] == "local"
    assert raw["output"]["options"]["storage_backend"] == "local"
    assert raw["input"]["options"]["path"] == str(input_path)
    assert raw["output"]["options"]["path"] == str(output_path)
    assert raw["input"]["options"]["path"] != raw["output"]["options"]["path"]


def test_available_recommendation_ids_drops_unavailable_items() -> None:
    recs = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i3", "rank": 1, "score": 0.9, "source": "popular_fallback"},
            {"user_id": "u1", "item_id": "i1", "rank": 2, "score": 0.8, "source": "personalized"},
            {"user_id": "u1", "item_id": "i4", "rank": 3, "score": 0.7, "source": "popular_fallback"},
            {"user_id": "u1", "item_id": "i2", "rank": 4, "score": 0.6, "source": "personalized"},
        ]
    )
    _events, _users, items = sample_system_catalog()
    assert available_recommendation_ids(
        recs,
        items,
        availability_filters=["published", "in_stock"],
        k=3,
    ) == ["i1", "i2"]


def test_dashboard_users_hashes_password() -> None:
    import bcrypt

    users = dashboard_users("alice", "s3cret")
    assert set(users) == {"alice"}
    assert bcrypt.checkpw(b"s3cret", users["alice"].encode("ascii"))
