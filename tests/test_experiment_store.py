from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import text

from cicerone.config import ConfigError, IOSettings
from cicerone.experiment.store import ExperimentStore, experiment_state


def test_experiment_store_roundtrip_dataset(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    assert store.read_state() is None
    assert store.read_exposures() == []

    store.write_state(experiment_state("exp", promoted_variant="treatment"))
    state = store.read_state()
    assert state is not None
    assert state["experiment_id"] == "exp"
    assert state["promoted_variant"] == "treatment"

    other = ExperimentStore(output)
    seen = other.read_state()
    assert seen is not None
    assert seen["promoted_variant"] == "treatment"

    store.append_exposures(
        [
            {
                "user_id": "u1",
                "experiment_id": "exp",
                "variant": "control",
                "generated_at": None,
                "exposed_at": "2026-08-25T00:00:00+00:00",
            }
        ]
    )
    store.append_exposures(
        [
            {
                "user_id": "u2",
                "experiment_id": "exp",
                "variant": "treatment",
                "generated_at": "t",
                "exposed_at": "2026-08-25T00:01:00+00:00",
            }
        ]
    )
    rows = store.read_exposures()
    assert [row["user_id"] for row in rows] == ["u1", "u2"]
    store.append_exposures(
        [
            {
                "user_id": "u9",
                "experiment_id": "other",
                "variant": "control",
                "generated_at": None,
                "exposed_at": "2026-08-25T00:02:00+00:00",
            }
        ]
    )
    assert [row["user_id"] for row in store.read_exposures(experiment_id="exp")] == ["u1", "u2"]


def test_experiment_store_roundtrip_sqlite(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    assert store.read_state() is None
    assert store.read_exposures() == []
    store.write_state(experiment_state("exp", promoted_variant="control"))
    state = store.read_state()
    assert state is not None
    assert state["promoted_variant"] == "control"
    store.append_exposures(
        [
            {
                "user_id": "u1",
                "experiment_id": "exp",
                "variant": "control",
                "generated_at": None,
                "exposed_at": "t",
            }
        ]
    )
    assert store.read_exposures()[0]["user_id"] == "u1"
    engine = store._db_engine()
    store.read_state()
    store.append_exposures(
        [
            {
                "user_id": "u2",
                "experiment_id": "exp",
                "variant": "treatment",
                "generated_at": None,
                "exposed_at": "t2",
            }
        ]
    )
    assert store._db_engine() is engine
    assert [row["user_id"] for row in store.read_exposures()] == ["u1", "u2"]


def test_experiment_store_sqlite_replace_clears_other_experiment_ids(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp-a", promoted_variant="control"))
    store.write_state(experiment_state("exp-b", promoted_variant="treatment"))
    frame = pd.read_sql(text("SELECT * FROM experiment_state"), store._db_engine())
    assert len(frame) == 1
    state = store.read_state()
    assert state is not None
    assert state["experiment_id"] == "exp-b"
    assert state["promoted_variant"] == "treatment"


def test_experiment_store_reads_legacy_table_without_promoted_at(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    engine = store._db_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE experiment_state (experiment_id TEXT, promoted_variant TEXT)"))
        conn.execute(
            text("INSERT INTO experiment_state (experiment_id, promoted_variant) VALUES ('exp', 'treatment')")
        )
    state = store.read_state()
    assert state is not None
    assert state["experiment_id"] == "exp"
    assert state["promoted_variant"] == "treatment"


def test_experiment_store_sqlite_replaces_same_experiment_id(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp", promoted_variant="control"))
    store.write_state(experiment_state("exp", promoted_variant="treatment"))
    frame = pd.read_sql(text("SELECT * FROM experiment_state"), store._db_engine())
    assert len(frame) == 1
    state = store.read_state()
    assert state is not None
    assert state["promoted_variant"] == "treatment"


def test_experiment_store_ignores_invalid_dataset_state(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    (tmp_path / "experiment_state.json").write_text("not-json", encoding="utf-8")
    store = ExperimentStore(output)
    assert store.read_state() is None


def test_append_exposures_rejects_object_store() -> None:
    output = IOSettings(
        kind="dataset",
        options={
            "storage_backend": "s3",
            "bucket": "recs",
            "access_key_id": "id",
            "secret_access_key": "secret",
        },
    )
    store = ExperimentStore(output)
    with pytest.raises(ConfigError, match="not atomic"):
        store.append_exposures(
            [
                {
                    "user_id": "u1",
                    "experiment_id": "exp",
                    "variant": "control",
                    "generated_at": None,
                    "exposed_at": "t",
                }
            ]
        )


def test_require_appendable_exposure_log_allows_db_and_local(tmp_path) -> None:
    from cicerone.experiment.store import require_appendable_exposure_log

    require_appendable_exposure_log(IOSettings(kind="db", options={"database_url": "sqlite://"}))
    require_appendable_exposure_log(
        IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    )


def test_append_exposures_empty_is_noop(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    ExperimentStore(output).append_exposures([])
    assert not (tmp_path / "exposures.jsonl").exists()


def test_experiment_store_state_roundtrip_s3() -> None:
    import boto3
    from moto import mock_aws

    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="recs")
        output = IOSettings(
            kind="dataset",
            options={
                "storage_backend": "s3",
                "bucket": "recs",
                "access_key_id": "test",
                "secret_access_key": "test",
                "prefix": "out",
            },
        )
        store = ExperimentStore(output)
        assert store.read_state() is None
        store.write_state(experiment_state("exp", promoted_variant="treatment"))
        state = store.read_state()
        assert state is not None
        assert state["promoted_variant"] == "treatment"


def test_jsonish_numpy_scalar_and_na() -> None:
    import numpy as np
    import pandas as pd

    from cicerone.experiment.store import _jsonish

    assert _jsonish(np.int64(3)) == 3
    assert _jsonish(pd.NA) is None
