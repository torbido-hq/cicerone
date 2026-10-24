from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import inspect
from support.contract import load_contract
from support.system_db import seed_catalog
from support.system_spec import (
    run_system_job,
    sample_system_catalog,
    seed_dataset_catalog,
    write_system_config,
)

# Generic SQLAlchemy types pandas.to_sql creates for these dtypes; its columns are nullable.
TO_SQL_TYPES = {"object": "Text", "bool": "Boolean", "int64": "BigInteger", "float32": "Float"}


def _to_sql_columns(dtypes: dict[str, str]) -> dict[str, list[Any]]:
    return {column: [TO_SQL_TYPES[dtype], True] for column, dtype in dtypes.items()}


def _to_sql_value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "Boolean"
    return "BigInteger" if isinstance(value, int) else "Text"


def _run_dataset_job(tmp_path: Path, **config: Any) -> Path:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    seed_dataset_catalog(input_dir, *sample_system_catalog())
    config_path = write_system_config(
        tmp_path / "cicerone.toml", kind="dataset", input_path=input_dir, output_path=output_dir, **config
    )
    run_system_job(config_path, triggered_by="contract")
    return output_dir


def _dataset_outputs(output_dir: Path) -> dict[str, Any]:
    files: dict[str, dict[str, str] | None] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_dir() or path.name.startswith("."):
            continue
        name = str(path.relative_to(output_dir))
        if path.parent.name == "recommendation_history":
            name = "recommendation_history/<generated_at>.parquet"
        assert name not in files, f"{path} collides with another output under {name}"
        dtypes = None
        if path.suffix == ".parquet":
            dtypes = {column: str(dtype) for column, dtype in pd.read_parquet(path).dtypes.items()}
        files[name] = dtypes
    manifest = json.loads((output_dir / "manifest.json").read_text())
    datetime.fromisoformat(manifest.pop("generated_at"))
    return {"files": files, "manifest": manifest}


def test_dataset_job_outputs(tmp_path):
    assert _dataset_outputs(_run_dataset_job(tmp_path)) == load_contract("job_dataset_outputs.json")


@pytest.mark.usefixtures("clean_schema")
def test_db_job_tables(tmp_path, db_engine, monkeypatch):
    seed_catalog(db_engine, *sample_system_catalog())
    before = set(inspect(db_engine).get_table_names())
    # The config names an env placeholder, so the database password is never written to disk.
    monkeypatch.setenv("CONTRACT_DATABASE_URL", db_engine.url.render_as_string(hide_password=False))
    config = write_system_config(tmp_path / "cicerone.toml", database_url="${CONTRACT_DATABASE_URL}")
    run_system_job(config, triggered_by="contract")

    inspector = inspect(db_engine)
    created = set(inspector.get_table_names()) - before
    schema = {
        table: {
            col["name"]: [type(col["type"].as_generic()).__name__, col["nullable"]]
            for col in inspector.get_columns(table)
        }
        for table in created
    }
    expected = load_contract("job_dataset_outputs.json")
    files = expected["files"]
    manifest = {**expected["manifest"], "generated_at": "<iso timestamp>"}
    assert schema == {
        "model_artifacts": {"payload": ["LargeBinary", False], "written_at": ["DateTime", False]},
        "recommendation_eval": {"payload": ["Text", False], "written_at": ["Text", False]},
        "recommendation_history": _to_sql_columns(files["recommendation_history/<generated_at>.parquet"]),
        "recommendation_items": _to_sql_columns(files["items_snapshot.parquet"]),
        "recommendation_runs": {name: [_to_sql_value_type(value), True] for name, value in manifest.items()},
        "recommendations": _to_sql_columns(files["recommendations.parquet"]),
    }


def test_legacy_item_based_k_neighbors_changes_recommendations(tmp_path):
    def recommendations(name: str, extra_toml: str = "") -> pd.DataFrame:
        run_dir = tmp_path / name
        run_dir.mkdir()
        output_dir = _run_dataset_job(run_dir, models=["item_based"], extra_toml=extra_toml)
        return pd.read_parquet(output_dir / "recommendations.parquet")

    default = recommendations("default")
    narrowed = recommendations("k1", "[job.item_based]\nk_neighbors = 1")
    assert len(default) > len(narrowed)
