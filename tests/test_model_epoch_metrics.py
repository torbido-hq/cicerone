from __future__ import annotations

import pandas as pd
import pytest
from support.model_events import synthetic_events

from cicerone.config import EpochMetricsSettings
from cicerone.dataset import build_dataset
from cicerone.model import train_and_recommend
from cicerone.model.epoch_metrics import (
    epoch_metric_total_epochs,
    fit_lightfm_with_epoch_metrics,
    invoke_epoch_fit_partial,
    sample_epoch_metric_users,
    should_log_epoch,
    warn_on_epoch_metric_trajectory,
)


def test_should_log_epoch_first_last_and_interval():

    assert should_log_epoch(1, 30, 5)
    assert should_log_epoch(5, 30, 5)
    assert should_log_epoch(30, 30, 5)
    assert not should_log_epoch(2, 30, 5)
    assert not should_log_epoch(29, 30, 5)


def test_invoke_epoch_fit_partial_lightfm_arity():
    calls: list[tuple[object, ...]] = []

    def fit_partial(dataset, epochs):  # type: ignore[no-untyped-def]
        calls.append((dataset, epochs))

    invoke_epoch_fit_partial(fit_partial, "ds")
    assert calls == [("ds", 1)]


def test_invoke_epoch_fit_partial_transformer_arity():
    calls: list[tuple[object, ...]] = []

    def fit_partial(dataset, min_epochs, max_epochs):  # type: ignore[no-untyped-def]
        calls.append((dataset, min_epochs, max_epochs))

    invoke_epoch_fit_partial(fit_partial, "ds")
    assert calls == [("ds", 1, 1)]


def test_invoke_epoch_fit_partial_max_epochs_keyword():
    calls: list[tuple[object, ...]] = []

    def fit_partial(dataset, min_epochs, *, max_epochs=0):  # type: ignore[no-untyped-def]
        calls.append((dataset, min_epochs, max_epochs))

    invoke_epoch_fit_partial(fit_partial, "ds")
    assert calls == [("ds", 1, 1)]


def test_invoke_epoch_fit_partial_lightfm_extra_positionals_keep_defaults():
    calls: list[tuple[object, ...]] = []

    def fit_partial(dataset, epochs, num_threads=4):  # type: ignore[no-untyped-def]
        calls.append((dataset, epochs, num_threads))

    invoke_epoch_fit_partial(fit_partial, "ds")
    assert calls == [("ds", 1, 4)]


def test_invoke_epoch_fit_partial_uninspected_arity(monkeypatch):
    calls: list[tuple[object, ...]] = []

    def fit_partial(dataset, epochs):  # type: ignore[no-untyped-def]
        calls.append((dataset, epochs))

    def _boom(_fn):  # type: ignore[no-untyped-def]
        raise TypeError("no signature")

    monkeypatch.setattr("cicerone.model.epoch_metrics.inspect.signature", _boom)
    invoke_epoch_fit_partial(fit_partial, "ds")
    assert calls == [("ds", 1)]


def test_warn_on_epoch_metric_trajectory_regression_and_plateau(caplog):

    settings = EpochMetricsSettings(every=5)

    with caplog.at_level("WARNING"):
        warn_on_epoch_metric_trajectory([(1, {"Precision@2": 0.5})], settings)
    assert caplog.text == ""

    with caplog.at_level("WARNING"):
        warn_on_epoch_metric_trajectory(
            [
                (1, {"Precision@2": 0.8}),
                (5, {"Precision@2": 0.7}),
                (10, {"Precision@2": 0.4}),
            ],
            settings,
        )
    assert "regressed" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        warn_on_epoch_metric_trajectory(
            [
                (1, {"Recall@2": 0.50}),
                (5, {"Recall@2": 0.501}),
                (10, {"Recall@2": 0.502}),
            ],
            settings,
        )
    assert "plateaued" in caplog.text


def test_warn_on_epoch_metric_trajectory_skips_metrics_missing_from_some_snapshots(caplog):

    # Heterogeneous keys: no KeyError; singleton values skip regression WARN.
    with caplog.at_level("WARNING"):
        warn_on_epoch_metric_trajectory(
            [
                (1, {"Precision@2": 0.9, "Recall@2": 0.5}),
                (5, {"Precision@2": 0.4}),
            ],
            EpochMetricsSettings(every=5),
        )
    assert "Precision@2" in caplog.text
    assert "regressed" in caplog.text
    assert "Recall@2" not in caplog.text


def test_fit_lightfm_with_epoch_metrics_rejects_non_lightfm_wrapper():

    class NoPartial:
        def fit(self, dataset):
            return self

        def recommend(self, **kwargs):
            return pd.DataFrame()

    with pytest.raises(TypeError, match="fit_partial"):
        fit_lightfm_with_epoch_metrics(
            NoPartial(), None, pd.DataFrame(), settings=EpochMetricsSettings(every=1), top_k=2
        )


def test_epoch_metric_total_epochs_prefers_n_epochs_then_epochs():

    class WithNEpochs:
        n_epochs = 7

    class WithEpochs:
        epochs = 3

    class WithNeither:
        pass

    assert epoch_metric_total_epochs(WithNEpochs()) == 7
    assert epoch_metric_total_epochs(WithEpochs()) == 3
    with pytest.raises(TypeError, match="n_epochs/epochs"):
        epoch_metric_total_epochs(WithNeither())


def test_warn_on_epoch_metric_trajectory_plateau_scale_handles_negative_values(caplog):

    # All-negative window: scale by max(|v|), not abs(max(v)).
    with caplog.at_level("WARNING"):
        warn_on_epoch_metric_trajectory(
            [
                (1, {"Score": -10.0}),
                (5, {"Score": -10.01}),
                (10, {"Score": -10.02}),
            ],
            EpochMetricsSettings(every=5),
        )
    assert "plateaued" in caplog.text


def test_sample_epoch_metric_users_is_seeded_not_prefix():
    from cicerone.model.constants import RANDOM_STATE

    ordered = list(range(1000))
    sampled = sample_epoch_metric_users(ordered, max_users=10)
    assert len(sampled) == 10
    assert sampled != ordered[:10]
    assert sample_epoch_metric_users(ordered, max_users=10) == sampled
    assert set(sampled).issubset(ordered)
    assert sample_epoch_metric_users([3, 1, 2], max_users=10) == [3, 1, 2]
    assert RANDOM_STATE == 42


def test_train_and_recommend_logs_epoch_metrics_when_configured(
    sample_items, feature_config, monkeypatch, caplog
):
    from cicerone.model_config import default_model_configs

    # Short epoch loop so this stays a unit test.
    configs = default_model_configs()
    configs["collaborative"]["epochs"] = 4
    events = synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    with caplog.at_level("INFO"):
        recommendations = train_and_recommend(
            built,
            target_users=["u1", "u2", "u3"],
            config=feature_config,
            top_k=2,
            enabled_models=["collaborative"],
            epoch_metrics=EpochMetricsSettings(every=2),
            model_configs=configs,
        )

    assert not recommendations.empty
    assert "Collaborative epoch 1/4 metrics:" in caplog.text
    assert "Collaborative epoch 2/4 metrics:" in caplog.text
    assert "Collaborative epoch 4/4 metrics:" in caplog.text
    assert "Collaborative epoch 3/4 metrics:" not in caplog.text
    assert "Precision@2" in caplog.text
    assert "Recall@2" in caplog.text


def test_train_and_recommend_skips_epoch_metrics_by_default(sample_items, feature_config, caplog):
    events = synthetic_events()
    built = build_dataset(events, None, sample_items, feature_config, half_life_days=90)

    with caplog.at_level("INFO"):
        train_and_recommend(
            built,
            target_users=["u1", "u2", "u3"],
            config=feature_config,
            top_k=2,
            enabled_models=["collaborative"],
        )

    assert "Collaborative epoch" not in caplog.text


def test_warn_on_epoch_metric_trajectory_uses_label(caplog):
    settings = EpochMetricsSettings(every=5)
    with caplog.at_level("WARNING"):
        warn_on_epoch_metric_trajectory(
            [
                (1, {"Precision@2": 0.8}),
                (5, {"Precision@2": 0.1}),
            ],
            settings,
            label="Sequential",
        )
    assert "Sequential epoch metrics:" in caplog.text
    assert "Collaborative epoch metrics:" not in caplog.text
