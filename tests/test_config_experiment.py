from __future__ import annotations

import logging

import pytest
from support.toml_config import write_toml

from cicerone.config import ConfigError, load_experiment_settings, load_settings
from cicerone.feature_config import BoostRule, EligibilityRule


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
    assert settings.experiment.allocation == "fixed"


def test_load_experiment_variants_and_remainder_traffic(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="cicerone.config.experiment"):
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
    assert "assigning remainder" in caplog.text
    assert "treatment" in caplog.text


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


def test_load_settings_rejects_log_exposures_on_object_store(tmp_path):
    with pytest.raises(ConfigError, match="not atomic"):
        load_settings(
            write_toml(
                tmp_path,
                """
                [job]
                [input]
                kind = "dataset"
                [input.options]
                storage_backend = "local"
                path = "/tmp/in"
                [output]
                kind = "dataset"
                [output.options]
                storage_backend = "s3"
                bucket = "recs"
                access_key_id = "id"
                secret_access_key = "secret"
                [experiment]
                enabled = true
                id = "exp"
                log_exposures = true
                [[experiment.variants]]
                name = "control"
                traffic = 0.5
                [[experiment.variants]]
                name = "treatment"
                traffic = 0.5
                """,
            )
        )


def test_load_settings_allows_log_exposures_on_s3_when_experiment_disabled(tmp_path):
    settings = load_settings(
        write_toml(
            tmp_path,
            """
            [job]
            [input]
            kind = "dataset"
            [input.options]
            storage_backend = "local"
            path = "/tmp/in"
            [output]
            kind = "dataset"
            [output.options]
            storage_backend = "s3"
            bucket = "recs"
            access_key_id = "id"
            secret_access_key = "secret"
            [experiment]
            enabled = false
            log_exposures = true
            """,
        )
    )
    assert settings.experiment.enabled is False
    assert settings.experiment.log_exposures is True


def test_load_settings_rejects_log_exposures_with_ha_on_local_dataset(tmp_path):
    with pytest.raises(ConfigError, match="events.ha requires output kind"):
        load_settings(
            write_toml(
                tmp_path,
                """
                [job]
                mode = "serve"
                [job.trigger]
                lock_backend = "redis"
                redis_url = "redis://localhost:6379/0"
                [serve]
                auth_token = "tok"
                [events]
                enabled = true
                kind = "webhook"
                ha = true
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
                [experiment]
                enabled = true
                id = "exp"
                log_exposures = true
                [[experiment.variants]]
                name = "control"
                traffic = 0.5
                [[experiment.variants]]
                name = "treatment"
                traffic = 0.5
                """,
            )
        )


def test_load_settings_allows_log_exposures_with_ha_on_db(tmp_path):
    settings = load_settings(
        write_toml(
            tmp_path,
            f"""
            [job]
            mode = "serve"
            [job.trigger]
            lock_backend = "redis"
            redis_url = "redis://localhost:6379/0"
            [serve]
            auth_token = "tok"
            [events]
            enabled = true
            kind = "webhook"
            ha = true
            [input]
            kind = "dataset"
            [input.options]
            storage_backend = "local"
            path = "/tmp/in"
            [output]
            kind = "db"
            [output.options]
            database_url = "sqlite+pysqlite:///{tmp_path / "recs.db"}"
            [experiment]
            enabled = true
            id = "exp"
            log_exposures = true
            [[experiment.variants]]
            name = "control"
            traffic = 0.5
            [[experiment.variants]]
            name = "treatment"
            traffic = 0.5
            """,
        )
    )
    assert settings.events.ha is True
    assert settings.experiment.log_exposures is True


def test_load_experiment_policy_names_and_tables(tmp_path):
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
            id = "policy"
            [[experiment.variants]]
            name = "control"
            traffic = 0.5
            boosts = ["featured"]
            eligibility = false
            [[experiment.variants]]
            name = "treatment"
            traffic = 0.5
            [[experiment.variants.boost]]
            name = "new-arrivals"
            kind = "boolean"
            item_column = "is_new"
            factor = 1.3
            [[experiment.variants.eligibility]]
            name = "published"
            op = "item_true"
            item_column = "published"
            """,
        )
    )
    control, treatment = settings.experiment.variants
    assert control.boosts == ("featured",)
    assert control.eligibility is False
    assert isinstance(treatment.boosts[0], BoostRule)
    assert treatment.boosts[0].name == "new-arrivals"
    assert isinstance(treatment.eligibility[0], EligibilityRule)
    assert treatment.eligibility[0].op == "item_true"


def test_load_experiment_rejects_boosts_bool_and_tables():
    with pytest.raises(
        ConfigError,
        match=r"must not set both boosts and \[\[experiment.variants.boost\]\]",
    ):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "variants": [
                    {"name": "control", "traffic": 0.5},
                    {
                        "name": "treatment",
                        "traffic": 0.5,
                        "boosts": False,
                        "boost": [{"name": "x", "kind": "boolean", "item_column": "featured", "factor": 1.1}],
                    },
                ],
            }
        )


def test_load_experiment_rejects_invalid_boost_table():
    with pytest.raises(ConfigError, match="boolean"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "variants": [
                    {"name": "control", "traffic": 0.5},
                    {
                        "name": "treatment",
                        "traffic": 0.5,
                        "boosts": [{"name": "x", "kind": "boolean", "item_column": "featured"}],
                    },
                ],
            }
        )


def test_load_experiment_rejects_incomplete_or_typed_policy_tables():
    with pytest.raises(ConfigError, match="kind"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "variants": [
                    {"name": "control", "traffic": 0.5},
                    {
                        "name": "treatment",
                        "traffic": 0.5,
                        "boosts": [{"name": "x", "item_column": "featured", "factor": 1.1}],
                    },
                ],
            }
        )
    with pytest.raises(ConfigError, match="op"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "variants": [
                    {"name": "control", "traffic": 0.5},
                    {
                        "name": "treatment",
                        "traffic": 0.5,
                        "eligibility": [{"name": "x", "item_column": "published"}],
                    },
                ],
            }
        )
    with pytest.raises(ConfigError):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "variants": [
                    {"name": "control", "traffic": 0.5},
                    {
                        "name": "treatment",
                        "traffic": 0.5,
                        "boosts": [
                            {
                                "name": "x",
                                "kind": "boolean",
                                "item_column": "featured",
                                "factor": {"bad": 1},
                            }
                        ],
                    },
                ],
            }
        )


def test_load_experiment_rejects_invalid_policy_spec_shape():
    with pytest.raises(ConfigError, match="must be true, false"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "variants": [
                    {"name": "control", "traffic": 0.5},
                    {"name": "treatment", "traffic": 0.5, "boosts": 1},
                ],
            }
        )
    with pytest.raises(ConfigError, match="must be true, false"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "variants": [
                    {"name": "control", "traffic": 0.5},
                    {"name": "treatment", "traffic": 0.5, "boosts": ["featured", {"name": "x"}]},
                ],
            }
        )


def test_load_experiment_rejects_duplicate_and_empty_policy_names():
    with pytest.raises(ConfigError, match="duplicate rule name"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "variants": [
                    {"name": "control", "traffic": 0.5},
                    {"name": "treatment", "traffic": 0.5, "boosts": ["featured", "featured"]},
                ],
            }
        )
    with pytest.raises(ConfigError, match="non-empty"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "variants": [
                    {"name": "control", "traffic": 0.5},
                    {"name": "treatment", "traffic": 0.5, "boosts": ["featured", "  "]},
                ],
            }
        )


def test_load_experiment_empty_policy_list_drops():
    settings = load_experiment_settings(
        {
            "enabled": True,
            "id": "exp",
            "variants": [
                {"name": "control", "traffic": 0.5},
                {"name": "treatment", "traffic": 0.5, "boosts": []},
            ],
        }
    )
    assert settings.variants[1].boosts == ()


def test_load_experiment_rejects_duplicate_replacement_names():
    with pytest.raises(ConfigError, match="duplicate rule name"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "variants": [
                    {"name": "control", "traffic": 0.5},
                    {
                        "name": "treatment",
                        "traffic": 0.5,
                        "boost": [
                            {"name": "featured", "kind": "boolean", "item_column": "featured", "factor": 1.2},
                            {"name": "featured", "kind": "boolean", "item_column": "sale", "factor": 1.1},
                        ],
                    },
                ],
            }
        )


def test_load_experiment_rejects_unknown_allocation():
    with pytest.raises(ConfigError, match="allocation"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "allocation": "epsilon-greedy",
                "variants": [{"name": "a", "traffic": 0.5}, {"name": "b", "traffic": 0.5}],
            }
        )


def test_load_experiment_thompson_requires_conversion_and_attribution():
    with pytest.raises(ConfigError, match="primary_metric 'conversion'"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "allocation": "thompson",
                "primary_metric": "ctr",
                "attribution": "click",
                "variants": [{"name": "a", "traffic": 0.5}, {"name": "b", "traffic": 0.5}],
            }
        )
    with pytest.raises(ConfigError, match="attribution 'click' or 'impression'"):
        load_experiment_settings(
            {
                "enabled": True,
                "id": "exp",
                "allocation": "thompson",
                "primary_metric": "conversion",
                "attribution": "user",
                "variants": [{"name": "a", "traffic": 0.5}, {"name": "b", "traffic": 0.5}],
            }
        )


def test_load_settings_thompson_requires_extra_and_track(tmp_path, monkeypatch):
    monkeypatch.setattr("cicerone.experiment.thompson.bandits_extra_available", lambda: False)
    with pytest.raises(ConfigError, match="bandits"):
        load_settings(
            write_toml(
                tmp_path,
                """
            [job]
            """
                + _minimal_io()
                + """
            [track]
            enabled = true
            [experiment]
            enabled = true
            id = "ranking-cvr"
            primary_metric = "conversion"
            attribution = "click"
            allocation = "thompson"
            [[experiment.variants]]
            name = "control"
            traffic = 0.5
            [[experiment.variants]]
            name = "treatment"
            traffic = 0.5
            """,
            )
        )
    monkeypatch.setattr("cicerone.experiment.thompson.bandits_extra_available", lambda: True)
    with pytest.raises(ConfigError, match="track.enabled"):
        load_settings(
            write_toml(
                tmp_path,
                """
            [job]
            """
                + _minimal_io()
                + """
            [experiment]
            enabled = true
            id = "ranking-cvr"
            primary_metric = "conversion"
            attribution = "click"
            allocation = "thompson"
            [[experiment.variants]]
            name = "control"
            traffic = 0.5
            [[experiment.variants]]
            name = "treatment"
            traffic = 0.5
            """,
            )
        )
    settings = load_settings(
        write_toml(
            tmp_path,
            """
        [job]
        """
            + _minimal_io()
            + """
        [track]
        enabled = true
        [experiment]
        enabled = true
        id = "ranking-cvr"
        primary_metric = "conversion"
        attribution = "click"
        allocation = "thompson"
        [[experiment.variants]]
        name = "control"
        traffic = 0.5
        [[experiment.variants]]
        name = "treatment"
        traffic = 0.5
        """,
        )
    )
    assert settings.experiment.allocation == "thompson"
    assert settings.experiment.explore_traffic == pytest.approx(0.5)
    assert settings.experiment.rotate_min_prob == pytest.approx(0.9)
