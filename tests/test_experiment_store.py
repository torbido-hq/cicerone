from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

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


def test_promoted_variant_reuses_cache_when_read_fails(tmp_path, monkeypatch) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp", promoted_variant="treatment"))
    assert store.promoted_variant("exp") == "treatment"

    def boom() -> None:
        raise RuntimeError("store down")

    monkeypatch.setattr(store, "read_state", boom)
    assert store.promoted_variant("exp") == "treatment"
    assert store.promoted_variant("other") is None


def test_promoted_variant_reuses_cache_when_db_read_raises(tmp_path, monkeypatch) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp", promoted_variant="treatment"))
    assert store.promoted_variant("exp") == "treatment"

    def boom() -> None:
        raise RuntimeError("store down")

    monkeypatch.setattr(store, "_read_state_db", boom)
    assert store.promoted_variant("exp") == "treatment"


def test_experiment_store_prefers_timestamped_promote_over_null(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp", promoted_variant="control"))
    engine = store._db_engine()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE experiment_state"))
        conn.execute(
            text(
                "CREATE TABLE experiment_state (experiment_id TEXT, promoted_variant TEXT, promoted_at TEXT)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO experiment_state (experiment_id, promoted_variant, promoted_at) "
                "VALUES ('exp', 'legacy', NULL)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO experiment_state (experiment_id, promoted_variant, promoted_at) "
                "VALUES ('exp', 'winner', '2026-09-01T00:00:00+00:00')"
            )
        )
    state = store.read_state()
    assert state is not None
    assert state["promoted_variant"] == "winner"


def test_experiment_store_last_state_reuses_write_cache(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    store.write_state(
        experiment_state("exp", promoted_variant="treatment", promoted_at="2026-09-01T00:00:00Z")
    )
    cached = store.last_state("exp")
    assert cached is not None
    assert cached["promoted_variant"] == "treatment"
    assert cached["promoted_at"] == "2026-09-01T00:00:00Z"
    assert store.last_state("other") is None


def test_last_state_ignores_cache_when_promoted_variant_queries_other_id(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    store.write_state(
        experiment_state("exp", promoted_variant="treatment", promoted_at="2026-09-02T00:00:00Z")
    )
    assert store.promoted_variant("other") is None
    assert store.last_state("other") is None
    cached = store.last_state("exp")
    assert cached is not None
    assert cached["promoted_variant"] == "treatment"


def test_last_state_matches_non_string_experiment_id(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    store.write_state(
        {
            "experiment_id": 7,
            "promoted_variant": "treatment",
            "promoted_at": "2026-09-02T00:00:00Z",
        }
    )
    cached = store.last_state("7")
    assert cached is not None
    assert cached["promoted_variant"] == "treatment"


def test_read_exposures_missing_experiment_id_column_returns_empty(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE recommendation_exposures (user_id TEXT, variant TEXT)"))
        conn.execute(text("INSERT INTO recommendation_exposures (user_id, variant) VALUES ('u1', 'control')"))
    assert ExperimentStore(output).read_exposures(experiment_id="exp") == []


def test_jsonish_numpy_scalar_and_na() -> None:
    import numpy as np
    import pandas as pd

    from cicerone.experiment.store import _jsonish

    assert _jsonish(np.int64(3)) == 3
    assert _jsonish(pd.NA) is None


def test_experiment_state_extra_keys_roundtrip_dataset(tmp_path) -> None:
    output = IOSettings(kind="dataset", options={"storage_backend": "local", "path": str(tmp_path)})
    store = ExperimentStore(output)
    store.write_state(
        experiment_state(
            "exp",
            promoted_variant=None,
            champion="control",
            challenger="blend",
            arms={"control": {"successes": 3, "failures": 1}},
            p_best={"control": 0.8, "blend": 0.2},
            pair_impressions=12,
        )
    )
    state = store.read_state()
    assert state is not None
    assert state["champion"] == "control"
    assert state["challenger"] == "blend"
    assert state["arms"]["control"]["successes"] == 3
    promoted, pair = store.assignment_overlay("exp")
    assert promoted is None
    assert pair == ("control", "blend")


def test_experiment_state_payload_roundtrip_sqlite(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'exp.db'}"
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    store.write_state(
        experiment_state("exp", promoted_variant="control", champion="control", challenger="treatment")
    )
    state = store.read_state()
    assert state is not None
    assert state["promoted_variant"] == "control"
    assert state["champion"] == "control"
    assert state["challenger"] == "treatment"
    assert "payload" not in state
    promoted, pair = store.assignment_overlay("exp")
    assert promoted == "control"
    assert pair == ("control", "treatment")


def test_experiment_state_alters_legacy_sqlite_table(tmp_path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'legacy.db'}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE experiment_state (experiment_id TEXT, promoted_variant TEXT, promoted_at TEXT)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO experiment_state (experiment_id, promoted_variant, promoted_at) "
                "VALUES ('exp', 'control', '2026-09-01T00:00:00Z')"
            )
        )
    output = IOSettings(kind="db", options={"database_url": url})
    store = ExperimentStore(output)
    store.write_state(experiment_state("exp", promoted_variant=None, champion="a", challenger="b"))
    state = store.read_state()
    assert state is not None
    assert state["champion"] == "a"
    assert state["challenger"] == "b"


def test_hydrate_state_row_ignores_invalid_payload() -> None:
    from cicerone.experiment.store import (
        _hydrate_state_row,
        active_pair_from_state,
        merge_experiment_state,
    )

    assert _hydrate_state_row({"experiment_id": "exp", "payload": "{not-json"}) == {"experiment_id": "exp"}
    assert _hydrate_state_row({"experiment_id": "exp", "payload": '["list"]'}) == {"experiment_id": "exp"}
    hydrated = _hydrate_state_row(
        {"experiment_id": "exp", "promoted_variant": "control", "payload": {"champion": "a", "payload": "x"}}
    )
    assert hydrated["champion"] == "a"
    assert "payload" not in hydrated
    assert active_pair_from_state(None) is None
    assert active_pair_from_state({"champion": "a"}) is None
    merged = merge_experiment_state(
        {"champion": "old", "extra": 1},
        experiment_id="exp",
        promoted_variant=None,
        challenger="new",
    )
    assert merged["champion"] == "old"
    assert merged["challenger"] == "new"
    assert merged["extra"] == 1
