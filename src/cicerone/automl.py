"""AutoML harness: backtests candidate strategy/weight configurations over
time-based folds and picks the best one by a ranking metric, so a job run
can automatically combine popular/latest/collaborative/item-based instead of
relying on a hand-picked, static config.

Deliberately does its own lightweight time-based event split rather than
rectools.model_selection's Interactions-level splitters: candidates here
span multiple STRATEGIES combined by cicerone.model.train_and_recommend
(which needs raw events/users/items to rebuild a BuiltDataset per fold), not
a single rectools model, so splitting at the raw-events level and reusing
cicerone.dataset.build_dataset for both sides of each fold is simpler and
avoids reimplementing dataset reconstruction from internal interaction ids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from rectools.metrics import MAP, NDCG, Recall, calc_metrics
from rectools.metrics.base import MetricAtK

from cicerone.dataset import build_dataset
from cicerone.feature_config import FeatureConfig
from cicerone.model import DEFAULT_MODELS, STRATEGIES, train_and_recommend

logger = logging.getLogger(__name__)

DEFAULT_N_SPLITS = 2
DEFAULT_TEST_DAYS = 14
DEFAULT_PRIMARY_METRIC = "MAP"

# Tried when [job.automl] doesn't configure its own "candidates": every
# strategy alone, the default priority combine, and a weighted-fusion blend
# across all four strategies.
DEFAULT_CANDIDATES: list[dict[str, Any]] = [
    {"models": ["popular"]},
    {"models": ["latest"]},
    {"models": ["collaborative"]},
    {"models": ["item_based"]},
    {"models": DEFAULT_MODELS},
    {
        "models": ["collaborative", "item_based", "popular", "latest"],
        "weights": {"collaborative": 1.0, "item_based": 0.7, "popular": 0.3, "latest": 0.3},
    },
]


@dataclass(frozen=True)
class Candidate:
    models: list[str]
    weights: dict[str, float] | None = None
    rrf_k: float | None = None

    @property
    def label(self) -> str:
        if self.weights is None:
            return "+".join(self.models)
        weighted = ",".join(f"{name}={self.weights.get(name, 1.0)}" for name in self.models)
        return f"fusion({weighted})"


@dataclass(frozen=True)
class CandidateResult:
    candidate: Candidate
    metrics: dict[str, float]
    n_folds: int


def _parse_candidates(raw: list[dict[str, Any]] | None) -> list[Candidate]:
    parsed = []
    for entry in raw if raw is not None else DEFAULT_CANDIDATES:
        models = list(entry["models"])
        unknown = [name for name in models if name not in STRATEGIES]
        if unknown:
            raise ValueError(
                f"Unknown model(s) in automl candidate {unknown}; available: {sorted(STRATEGIES)}"
            )  # noqa: E501
        weights = {str(k): float(v) for k, v in entry["weights"].items()} if "weights" in entry else None
        if weights is not None:
            unknown_weights = [name for name in weights if name not in models]
            if unknown_weights:
                raise ValueError(f"automl candidate weight key(s) {unknown_weights} not in models {models}")
        parsed.append(
            Candidate(
                models=models,
                weights=weights,
                rrf_k=float(entry["rrf_k"]) if "rrf_k" in entry else None,
            )
        )
    return parsed


def _time_based_folds(
    events: pd.DataFrame, n_splits: int, test_days: int
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Walks backward from the most recent event in fixed-size, non-overlapping
    `test_days`-day windows, each becoming one (train, test) fold: everything
    strictly before the window is "train", everything inside it is "test".
    Folds are returned oldest-test-window-first. A fold is skipped if either
    side ends up empty (e.g. not enough history for the requested n_splits).
    """
    occurred_at = pd.to_datetime(events["occurred_at"], utc=True)
    max_ts = occurred_at.max()
    window = pd.Timedelta(days=test_days)

    folds = []
    for i in range(n_splits):
        test_end = max_ts - window * i
        test_start = test_end - window
        train_events = events[occurred_at < test_start]
        test_events = events[(occurred_at >= test_start) & (occurred_at < test_end)]
        if train_events.empty or test_events.empty:
            continue
        folds.append((train_events, test_events))
    return list(reversed(folds))


def _make_metrics(top_k: int) -> dict[str, MetricAtK]:
    return {
        f"MAP@{top_k}": MAP(k=top_k),
        f"NDCG@{top_k}": NDCG(k=top_k),
        f"Recall@{top_k}": Recall(k=top_k),
    }


def evaluate_candidates(
    events: pd.DataFrame,
    users: pd.DataFrame | None,
    items: pd.DataFrame | None,
    config: FeatureConfig,
    top_k: int,
    half_life_days: float,
    candidates: list[dict[str, Any]] | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    test_days: int = DEFAULT_TEST_DAYS,
) -> list[CandidateResult]:
    """Backtests every candidate config over up to `n_splits` time-based
    folds and returns one CandidateResult per candidate (metrics averaged
    across the folds that had data), in the same order as `candidates`.
    """
    parsed_candidates = _parse_candidates(candidates)
    folds = _time_based_folds(events, n_splits=n_splits, test_days=test_days)
    if not folds:
        raise ValueError(
            f"Not enough event history for {n_splits} fold(s) of {test_days} day(s) each; "
            "reduce automl n_splits/test_days or provide more historical events"
        )

    metrics = _make_metrics(top_k)
    results = []
    for candidate in parsed_candidates:
        fold_metrics = []
        for train_events, test_events in folds:
            built = build_dataset(train_events, users, items, config, half_life_days=half_life_days)
            test_users = sorted(set(test_events["user_id"]))
            reco = train_and_recommend(
                built,
                test_users,
                config,
                top_k=top_k,
                enabled_models=candidate.models,
                weights=candidate.weights,
                rrf_k=candidate.rrf_k,
            )
            test_built = build_dataset(test_events, users, items, config, half_life_days=half_life_days)
            fold_metrics.append(calc_metrics(metrics, reco=reco, interactions=test_built.interactions))
        averaged = dict(pd.DataFrame(fold_metrics).mean()) if fold_metrics else dict.fromkeys(metrics, 0.0)
        results.append(CandidateResult(candidate=candidate, metrics=averaged, n_folds=len(fold_metrics)))
        logger.info(
            "AutoML candidate '%s' scored %s over %d fold(s)", candidate.label, averaged, len(fold_metrics)
        )
    return results


def select_best_candidate(
    results: list[CandidateResult], primary_metric: str = DEFAULT_PRIMARY_METRIC
) -> CandidateResult:
    """Picks the candidate with the highest average value for the metric
    whose name starts with `primary_metric` (e.g. "MAP" matches "MAP@10").
    Ties are broken by candidate list order (first one wins).
    """
    if not results:
        raise ValueError("No candidate results to select from")
    metric_key = next((key for key in results[0].metrics if key.startswith(primary_metric)), None)
    if metric_key is None:
        raise ValueError(
            f"No metric starting with '{primary_metric}' found in results: {list(results[0].metrics)}"
        )
    return max(results, key=lambda result: result.metrics[metric_key])
