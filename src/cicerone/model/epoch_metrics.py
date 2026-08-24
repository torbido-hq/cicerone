"""Optional per-epoch Precision/Recall logging during collaborative or sequential fit."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset
from rectools.metrics import Precision, Recall, calc_metrics

from cicerone.config import EpochMetricsSettings
from cicerone.model.constants import RANDOM_STATE
from cicerone.model.strategies import RecommenderModel

logger = logging.getLogger(__name__)

_EPOCH_METRICS_RNG = random.Random()


def should_log_epoch(epoch: int, total_epochs: int, every: int) -> bool:
    return epoch == 1 or epoch == total_epochs or epoch % every == 0


def sample_epoch_metric_users(external_ids, max_users: int) -> list:
    users = list(external_ids)
    if len(users) <= max_users:
        return users
    _EPOCH_METRICS_RNG.seed(RANDOM_STATE)
    return _EPOCH_METRICS_RNG.sample(users, max_users)


def _epoch_metric_fit_partial(model: object) -> Callable:
    fit_partial = getattr(model, "fit_partial", None)
    if not callable(fit_partial):
        raise TypeError(
            f"{type(model).__name__} does not support fit_partial(); "
            "epoch metric logging requires a model with fit_partial and an epoch count"
        )
    return fit_partial


def epoch_metric_total_epochs(model: object) -> int:
    # rectools stores epochs as n_epochs; accept epochs for other wrappers.
    for attr in ("n_epochs", "epochs"):
        value = getattr(model, attr, None)
        if value is not None:
            return int(value)
    raise TypeError(
        f"{type(model).__name__} has no n_epochs/epochs attribute; "
        "epoch metric logging needs a known epoch count"
    )


def warn_on_epoch_metric_trajectory(
    history: list[tuple[int, dict[str, float]]],
    settings: EpochMetricsSettings,
    *,
    label: str = "Collaborative",
) -> None:
    """WARN when a tracked metric regresses from its best or plateaus late."""
    if len(history) < 2:
        return
    metric_names: set[str] = set()
    for _, snapshot in history:
        metric_names.update(snapshot)
    for metric_name in sorted(metric_names):
        values = [snapshot[metric_name] for _, snapshot in history if metric_name in snapshot]
        if len(values) < 2:
            continue
        best = max(values)
        last = values[-1]
        if best > 0 and (best - last) / best >= settings.regression_drop:
            logger.warning(
                "%s epoch metrics: %s regressed from best %.4f to final %.4f "
                "(drop >= %.0f%% across logged epochs)",
                label,
                metric_name,
                best,
                last,
                settings.regression_drop * 100,
            )
        if len(values) >= settings.plateau_window:
            recent = values[-settings.plateau_window :]
            span = max(recent) - min(recent)
            scale = max(max(abs(v) for v in recent), 1e-9)
            if span / scale <= settings.plateau_eps:
                logger.warning(
                    "%s epoch metrics: %s plateaued near %.4f over the last %d logged snapshots (span %.4f)",
                    label,
                    metric_name,
                    recent[-1],
                    settings.plateau_window,
                    span,
                )


def interactions_for_epoch_metrics(
    dataset: Dataset, interactions: pd.DataFrame, max_users: int
) -> pd.DataFrame:
    users = sample_epoch_metric_users(dataset.user_id_map.external_ids, max_users)
    return interactions[interactions[Columns.User].isin(users)]


def fit_with_epoch_metrics(
    model: RecommenderModel,
    dataset: Dataset,
    interactions: pd.DataFrame,
    settings: EpochMetricsSettings,
    top_k: int,
    *,
    label: str = "Collaborative",
) -> RecommenderModel:
    """``fit_partial`` loop with in-sample Precision/Recall@K logs."""
    fit_partial = _epoch_metric_fit_partial(model)
    total_epochs = epoch_metric_total_epochs(model)
    metric_defs = {
        f"Precision@{top_k}": Precision(k=top_k),
        f"Recall@{top_k}": Recall(k=top_k),
    }
    users = list(dict.fromkeys(interactions[Columns.User].tolist()))
    history: list[tuple[int, dict[str, float]]] = []

    for epoch in range(1, total_epochs + 1):
        fit_partial(dataset, 1)
        if not should_log_epoch(epoch, total_epochs, settings.every):
            continue
        reco = model.recommend(
            users=users,
            dataset=dataset,
            k=top_k,
            filter_viewed=False,
        )
        snapshot = calc_metrics(metric_defs, reco=reco, interactions=interactions)
        history.append((epoch, snapshot))
        logger.info("%s epoch %d/%d metrics: %s", label, epoch, total_epochs, snapshot)

    warn_on_epoch_metric_trajectory(history, settings, label=label)
    return model


def fit_lightfm_with_epoch_metrics(
    model: RecommenderModel,
    dataset: Dataset,
    interactions: pd.DataFrame,
    settings: EpochMetricsSettings,
    top_k: int,
) -> RecommenderModel:
    return fit_with_epoch_metrics(model, dataset, interactions, settings, top_k, label="Collaborative")
