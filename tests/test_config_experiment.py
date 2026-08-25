from __future__ import annotations

import pytest
from support.toml_config import write_toml

from cicerone.config import ConfigError, load_experiment_settings, load_settings


def _minimal_io() -> str:
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


def test_load_experiment_defaults_disabled(tmp_path):
    settings = load_settings(write_toml(tmp_path, "[job]\n" + _minimal_io()))
    assert settings.experiment.enabled is False
    assert settings.experiment.variants == ()
    assert settings.experiment.log_exposures is False
    assert settings.experiment.automl_challenger is False


def test_load_experiment_variants_and_remainder_traffic(tmp_path):
    settings = load_settings(
        write_toml(
            tmp_path,
            """
            [job]
            """
            + _minimal_io()
            + """
            [experiment]
            enabled = true
            id = "rrf-vs-blend"
            primary_metric = "purchase"

            [[experiment.variants]]
            name = "control"
            traffic = 0.4

            [[experiment.variants]]
            name = "treatment"
            traffic = 0.4
            combiner = "blend"
            models = ["collaborative", "popular"]
            """,
        )
    )
    experiment = settings.experiment
    assert experiment.enabled is True
    assert experiment.id == "rrf-vs-blend"
    assert experiment.primary_metric == "purchase"
    assert [variant.name for variant in experiment.variants] == ["control", "treatment"]
    assert experiment.variants[0].traffic == pytest.approx(0.4)
    assert experiment.variants[1].traffic == pytest.approx(0.6)
    assert experiment.variants[1].combiner == "blend"
    assert experiment.variants[1].models == ["collaborative", "popular"]


def test_load_experiment_rejects_duplicate_names():
    with pytest.raises(ConfigError, match="unique"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "variants": [
                    {"name": "control", "traffic": 0.5},
                    {"name": "control", "traffic": 0.5},
                ],
            }
        )


def test_load_experiment_rejects_traffic_over_one():
    with pytest.raises(ConfigError, match="exceeds 1"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "variants": [
                    {"name": "a", "traffic": 0.7},
                    {"name": "b", "traffic": 0.4},
                ],
            }
        )


def test_load_experiment_requires_id_and_two_variants():
    with pytest.raises(ConfigError, match="experiment.id"):
        load_experiment_settings({"enabled": True, "variants": [{"name": "a"}, {"name": "b"}]})
    with pytest.raises(ConfigError, match="at least two"):
        load_experiment_settings({"enabled": True, "id": "exp", "variants": [{"name": "only"}]})


def test_load_experiment_rejects_unknown_combiner_and_models():
    with pytest.raises(ConfigError, match="combiner"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "variants": [
                    {"name": "a", "traffic": 0.5, "combiner": "bandit"},
                    {"name": "b", "traffic": 0.5},
                ],
            }
        )
    with pytest.raises(ConfigError, match="unknown model"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "variants": [
                    {"name": "a", "traffic": 0.5, "models": ["not_a_model"]},
                    {"name": "b", "traffic": 0.5},
                ],
            }
        )


def test_load_experiment_automl_challenger_allows_empty_variants():
    settings = load_experiment_settings({"enabled": True, "id": "auto", "automl_challenger": True})
    assert settings.automl_challenger is True
    assert settings.variants == ()
