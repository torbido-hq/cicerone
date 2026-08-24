"""Sequential strategy: SASRec / BERT4Rec config, AutoML skip, optional torch extra."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from rectools import Columns
from support.toml_config import write_toml

from cicerone.automl import (
    Candidate,
    exclude_sequential_from_candidates,
    median_distinct_items_per_user,
    sequential_automl_skip_reason,
)
from cicerone.config import ConfigError, load_settings
from cicerone.dataset import build_dataset
from cicerone.model_config import (
    DEFAULT_SEQUENTIAL_CONFIG,
    SEQUENTIAL_ARCHITECTURES,
    SEQUENTIAL_EXTRA_HINT,
    apply_sequential_architecture,
    default_model_configs,
    rectools_model_config,
    resolve_model_configs,
    sequential_extra_available,
)

requires_sequential_extra = pytest.mark.skipif(
    not sequential_extra_available(),
    reason="requires cicerone-recommender[sequential] (rectools[torch])",
)


def _minimal_io_toml() -> str:
    return """
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "/tmp/in"

        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "/tmp/out"
    """


def _sequence_events(n_items: int = 6, n_users: int = 4) -> pd.DataFrame:
    now = pd.Timestamp.utcnow()
    rows = []
    for user_i in range(n_users):
        for item_i in range(n_items):
            rows.append(
                {
                    "user_id": f"u{user_i}",
                    "item_id": f"i{item_i}",
                    "event_type": "purchase",
                    "quantity": 1,
                    "occurred_at": now - pd.Timedelta(hours=n_items - item_i),
                }
            )
    return pd.DataFrame(rows)


def test_default_sequential_config_is_sasrec():
    assert DEFAULT_SEQUENTIAL_CONFIG["cls"] == "SASRecModel"
    assert DEFAULT_SEQUENTIAL_CONFIG["architecture"] == "sasrec"
    assert DEFAULT_SEQUENTIAL_CONFIG["n_factors"] == 64
    assert DEFAULT_SEQUENTIAL_CONFIG["epochs"] == 3
    assert DEFAULT_SEQUENTIAL_CONFIG["loss"] == "sampled_softmax"
    assert DEFAULT_SEQUENTIAL_CONFIG["n_negatives"] == 256
    assert DEFAULT_SEQUENTIAL_CONFIG["transformer_layers_type"] == "LiGRLayers"
    assert DEFAULT_SEQUENTIAL_CONFIG["session_max_len"] == 50
    assert DEFAULT_SEQUENTIAL_CONFIG["train_min_user_interactions"] == 2
    configs = default_model_configs()
    assert configs["sequential"]["cls"] == "SASRecModel"


def test_architecture_sasrec_and_bert4rec_map_to_cls():
    sasrec = apply_sequential_architecture({"architecture": "sasrec", "n_factors": 32})
    assert sasrec["cls"] == "SASRecModel"
    assert sasrec["architecture"] == "sasrec"
    bert = apply_sequential_architecture({"architecture": "BERT4Rec", "epochs": 2})
    assert bert["cls"] == "BERT4RecModel"
    assert bert["architecture"] == "bert4rec"
    hstu = apply_sequential_architecture({"architecture": "hstu"})
    assert hstu["cls"] == "HSTUModel"
    assert hstu["architecture"] == "hstu"


def test_architecture_defaults_when_missing():
    cfg = apply_sequential_architecture({})
    assert cfg["cls"] == "SASRecModel"
    assert cfg["architecture"] == "sasrec"


def test_architecture_rejects_non_string():
    with pytest.raises(ConfigError, match="must be a string"):
        apply_sequential_architecture({"architecture": 1})


def test_architecture_inferred_from_cls():
    cfg = apply_sequential_architecture({"cls": "BERT4RecModel"})
    assert cfg["architecture"] == "bert4rec"


def test_architecture_cls_conflict_raises():
    with pytest.raises(ConfigError, match="Conflicting sequential"):
        apply_sequential_architecture(
            {"architecture": "sasrec", "cls": "BERT4RecModel"},
            architecture_explicit=True,
            cls_explicit=True,
        )


def test_architecture_override_wins_over_default_cls():
    cfg = apply_sequential_architecture(
        {"architecture": "bert4rec", "cls": "SASRecModel"},
        architecture_explicit=True,
        cls_explicit=False,
    )
    assert cfg["cls"] == "BERT4RecModel"
    assert cfg["architecture"] == "bert4rec"


def test_unknown_architecture_raises():
    with pytest.raises(ConfigError, match="architecture must be one of"):
        apply_sequential_architecture({"architecture": "gru4rec"})


def test_unknown_sequential_cls_raises():
    with pytest.raises(ConfigError, match="HSTUModel"):
        apply_sequential_architecture({"cls": "PopularModel"})


def test_hstu_drops_ligr_transformer_layers():
    cfg = apply_sequential_architecture(
        {**DEFAULT_SEQUENTIAL_CONFIG, "architecture": "hstu"},
        architecture_explicit=True,
    )
    assert cfg["cls"] == "HSTUModel"
    assert "transformer_layers_type" not in cfg


def test_rectools_model_config_expands_ligr_layers():
    stripped = rectools_model_config({**DEFAULT_SEQUENTIAL_CONFIG})
    assert stripped["transformer_layers_type"].endswith("LiGRLayers")
    assert "." in stripped["transformer_layers_type"]


def test_rectools_model_config_strips_architecture():
    stripped = rectools_model_config({"cls": "SASRecModel", "architecture": "sasrec", "epochs": 1})
    assert "architecture" not in stripped
    assert stripped["cls"] == "SASRecModel"
    assert stripped["epochs"] == 1


@pytest.mark.parametrize(
    ("architecture", "expected_cls"),
    [("sasrec", "SASRecModel"), ("bert4rec", "BERT4RecModel"), ("hstu", "HSTUModel")],
)
def test_toml_architecture_selects_cls(tmp_path, architecture, expected_cls):
    config_path = write_toml(
        tmp_path,
        f"""
        [job]
        [model.sequential]
        architecture = "{architecture}"
        n_factors = 32
        epochs = 1
        loss = "softmax"
        session_max_len = 8
        train_min_user_interactions = 2
        {_minimal_io_toml()}
        """,
    )
    settings = load_settings(config_path)
    cfg = settings.model_configs["sequential"]
    assert cfg["cls"] == expected_cls
    assert cfg["architecture"] == architecture
    assert cfg["n_factors"] == 32
    assert cfg["epochs"] == 1
    assert cfg["loss"] == "softmax"
    assert cfg["session_max_len"] == 8
    assert cfg["train_min_user_interactions"] == 2
    assert expected_cls == SEQUENTIAL_ARCHITECTURES[architecture]


def test_resolve_model_configs_sequential_hyperparameters():
    configs = resolve_model_configs(
        {
            "sequential": {
                "architecture": "bert4rec",
                "n_factors": 16,
                "epochs": 2,
                "session_max_len": 12,
                "lr": 0.002,
            }
        }
    )
    cfg = configs["sequential"]
    assert cfg["cls"] == "BERT4RecModel"
    assert cfg["n_factors"] == 16
    assert cfg["epochs"] == 2
    assert cfg["session_max_len"] == 12
    assert cfg["lr"] == 0.002


def test_median_distinct_items_per_user():
    events = _sequence_events(n_items=6, n_users=3)
    assert median_distinct_items_per_user(events) == 6.0
    assert median_distinct_items_per_user(pd.DataFrame()) == 0.0


def test_sequential_automl_skip_reason_missing_extra(monkeypatch):
    monkeypatch.setattr("cicerone.automl.sequential_extra_available", lambda: False)
    reason = sequential_automl_skip_reason(_sequence_events(), min_median_interactions=5)
    assert reason is not None
    assert SEQUENTIAL_EXTRA_HINT in reason


def test_sequential_automl_skip_reason_sparse_history(monkeypatch):
    monkeypatch.setattr("cicerone.automl.sequential_extra_available", lambda: True)
    events = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 1},
            {"user_id": "u2", "item_id": "i1", "event_type": "purchase", "quantity": 1},
        ]
    )
    reason = sequential_automl_skip_reason(events, min_median_interactions=5)
    assert reason is not None
    assert "min_median_interactions=5" in reason


def test_sequential_automl_skip_reason_keeps_dense_with_extra(monkeypatch):
    monkeypatch.setattr("cicerone.automl.sequential_extra_available", lambda: True)
    assert sequential_automl_skip_reason(_sequence_events(), min_median_interactions=5) is None


def test_exclude_sequential_from_candidates_drops_solo_and_strips_fusion():
    solo = Candidate(models=["sequential"])
    fusion = Candidate(
        models=["collaborative", "sequential", "popular"],
        weights={"collaborative": 1.0, "sequential": 1.0, "popular": 0.3},
    )
    popular = Candidate(models=["popular"])
    sequential_only_weights = Candidate(
        models=["sequential", "popular"],
        weights={"sequential": 1.0},
    )
    result = exclude_sequential_from_candidates([solo, fusion, popular, sequential_only_weights])
    assert [c.models for c in result] == [["collaborative", "popular"], ["popular"], ["popular"]]
    assert result[0].weights == {"collaborative": 1.0, "popular": 0.3}
    assert result[2].weights is None


def test_evaluate_candidates_skips_sequential_when_extra_missing(
    sample_items, feature_config, caplog, monkeypatch
):
    from cicerone.automl import evaluate_candidates

    monkeypatch.setattr("cicerone.automl.sequential_extra_available", lambda: False)
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {
                "user_id": f"u{i}",
                "item_id": f"i{j}",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": now - pd.Timedelta(days=d),
            }
            for d in (0, 10, 20)
            for i in range(3)
            for j in range(6)
        ]
    )
    with caplog.at_level("INFO", logger="cicerone.automl"):
        results = evaluate_candidates(
            events,
            None,
            sample_items,
            feature_config,
            top_k=2,
            half_life_days=90,
            candidates=[
                {"models": ["sequential"]},
                {"models": ["popular"]},
                {"models": ["sequential", "popular"], "weights": {"sequential": 1.0, "popular": 0.3}},
            ],
            n_splits=1,
            test_days=7,
        )
    assert any("Excluding sequential" in record.message for record in caplog.records)
    assert [result.candidate.models for result in results] == [["popular"], ["popular"]]


def test_evaluate_candidates_skips_sequential_below_median_threshold(
    sample_items, feature_config, caplog, monkeypatch
):
    from cicerone.automl import evaluate_candidates

    monkeypatch.setattr("cicerone.automl.sequential_extra_available", lambda: True)
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": now - pd.Timedelta(days=d),
            }
            for d in (0, 10, 20)
        ]
    )
    with caplog.at_level("INFO", logger="cicerone.automl"):
        results = evaluate_candidates(
            events,
            None,
            sample_items,
            feature_config,
            top_k=2,
            half_life_days=90,
            candidates=[{"models": ["sequential"]}, {"models": ["popular"]}],
            n_splits=1,
            test_days=7,
            sequential_min_median_interactions=5,
        )
    assert any("min_median_interactions=5" in record.message for record in caplog.records)
    assert [result.candidate.models for result in results] == [["popular"]]


def test_evaluate_candidates_raises_when_only_sequential_is_excluded(
    sample_items, feature_config, monkeypatch
):
    from cicerone.automl import evaluate_candidates

    monkeypatch.setattr("cicerone.automl.sequential_extra_available", lambda: False)
    now = pd.Timestamp.utcnow()
    events = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_type": "purchase",
                "quantity": 1,
                "occurred_at": now - pd.Timedelta(days=d),
            }
            for d in (0, 10, 20)
        ]
    )
    with pytest.raises(ValueError, match="no candidates left after excluding sequential"):
        evaluate_candidates(
            events,
            None,
            sample_items,
            feature_config,
            top_k=2,
            half_life_days=90,
            candidates=[{"models": ["sequential"]}],
            n_splits=1,
            test_days=7,
        )


def test_serve_import_does_not_load_torch():
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    code = """
import sys
from cicerone.serve import create_app
assert create_app is not None
assert "torch" not in sys.modules
assert "pytorch_lightning" not in sys.modules
for name in list(sys.modules):
    assert "rectools.models.nn" not in name
"""
    subprocess.check_call([sys.executable, "-c", code], cwd=str(repo), env=env)


@requires_sequential_extra
@pytest.mark.parametrize(
    ("architecture", "expected_name"),
    [("sasrec", "SASRecModel"), ("bert4rec", "BERT4RecModel"), ("hstu", "HSTUModel")],
)
def test_toml_builds_expected_rectools_class(architecture, expected_name):
    from rectools.models import model_from_config

    configs = resolve_model_configs({"sequential": {"architecture": architecture, "epochs": 1}})
    model = model_from_config(rectools_model_config(configs["sequential"]))
    assert type(model).__name__ == expected_name
    params = model.get_params(simple_types=True)
    assert expected_name in str(params["cls"])
    assert params["epochs"] == 1
    if architecture == "sasrec":
        assert "sampled_softmax" in str(params.get("loss", "")).lower()
        assert "LiGRLayers" in str(params.get("transformer_layers_type", ""))


@requires_sequential_extra
def test_sequential_fit_recommend_top_k_shape(feature_config):
    from cicerone.model import train_and_recommend

    events = _sequence_events()
    built = build_dataset(events, None, None, feature_config, half_life_days=90)
    configs = default_model_configs()
    configs["sequential"].update(
        {
            "epochs": 1,
            "n_factors": 8,
            "n_heads": 2,
            "n_blocks": 1,
            "batch_size": 4,
            "session_max_len": 8,
        }
    )
    recs = train_and_recommend(
        built,
        target_users=sorted(events["user_id"].unique()),
        config=feature_config,
        top_k=3,
        enabled_models=["sequential"],
        model_configs=configs,
    )
    assert not recs.empty
    assert set(recs[Columns.User]) <= set(events["user_id"])
    assert recs.groupby(Columns.User).size().max() <= 3
    assert (recs["source"] == "sequential").all()
    assert {"rank", "score"}.issubset(recs.columns)
