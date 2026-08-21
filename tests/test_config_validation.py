from __future__ import annotations

import pytest

from cicerone.config import ConfigError, resolve_epoch_metrics, validate_model_weights
from cicerone.config.validation import require_non_negative_int


def test_validate_model_weights_rejects_negative_and_allows_non_negative():
    validate_model_weights(None)
    validate_model_weights({})
    validate_model_weights({"popular": 0.0, "latest": 1.5})
    with pytest.raises(ConfigError, match="non-negative"):
        validate_model_weights({"popular": -0.1})


def test_resolve_epoch_metrics_rejects_non_positive_when_enabled():
    with pytest.raises(ConfigError, match="epoch_metrics_every"):
        resolve_epoch_metrics(log_epoch_metrics=True, every=0)


def test_resolve_epoch_metrics_rejects_fraction_outside_unit_interval():
    with pytest.raises(ConfigError, match="epoch_metrics_regression_drop"):
        resolve_epoch_metrics(log_epoch_metrics=True, regression_drop=1.5)
    with pytest.raises(ConfigError, match="epoch_metrics_plateau_eps"):
        resolve_epoch_metrics(log_epoch_metrics=True, plateau_eps=0)
    assert resolve_epoch_metrics(log_epoch_metrics=True, regression_drop=1.0).regression_drop == 1.0


def test_validate_model_weights_no_op_when_none():
    validate_model_weights(None)
    validate_model_weights({"popular": 1.0})


def test_require_non_negative_int():
    assert require_non_negative_int(0, name="job.explain.max_similar_items") == 0
    with pytest.raises(ConfigError, match="job.explain.max_similar_items"):
        require_non_negative_int(-1, name="job.explain.max_similar_items")
