from __future__ import annotations

from cicerone.config import IOSettings
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
