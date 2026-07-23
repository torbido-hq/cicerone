"""AutoML harness: backtests candidate strategy/weight configurations over
time-based folds and picks the best one by a ranking metric, so a job run
can automatically combine popular/latest/collaborative/item-based instead of
relying on a hand-picked, static config.

Deliberately does its own lightweight time-based event split rather than
rectools.model_selection's Interactions-level splitters: candidates here
span multiple STRATEGIES combined by cicerone.model.train_and_recommend
(which needs raw events/users/items to rebuild a BuiltDataset per fold), not
a single rectools model, so splitting at the raw-events level and reusing
cicerone.dataset.build_dataset (train side) / build_interactions (test
side, interactions-only -- no need for a full feature-laden Dataset just to
score against ground truth) is simpler and avoids reimplementing dataset
reconstruction from internal interaction ids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from rectools.metrics import MAP, NDCG, Recall, calc_metrics
from rectools.metrics.base import MetricAtK

from cicerone.config import AUTOML_DEFAULT_N_SPLITS, AUTOML_DEFAULT_PRIMARY_METRIC, AUTOML_DEFAULT_TEST_DAYS
from cicerone.dataset import build_dataset, build_interactions
from cicerone.feature_config import FeatureConfig
from cicerone.model import (
    DEFAULT_MODELS,
    STRATEGIES,
    RecommenderModel,
    train_and_recommend,
    validate_model_weights,
    validate_rrf_k,
)

logger = logging.getLogger(__name__)

# Re-exported under automl-local names for backward-compat/readability at
# call sites here; the actual values live in cicerone.config so they can't
# drift apart from the [job.automl] TOML defaults.
DEFAULT_N_SPLITS = AUTOML_DEFAULT_N_SPLITS
DEFAULT_TEST_DAYS = AUTOML_DEFAULT_TEST_DAYS
DEFAULT_PRIMARY_METRIC = AUTOML_DEFAULT_PRIMARY_METRIC

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
        weighted = ",".join(f"{name}={self.weights[name]}" for name in self.models)
        return f"fusion({weighted})"


@dataclass(frozen=True)
class CandidateResult:
    candidate: Candidate
    metrics: dict[str, float]
    n_folds: int


def _parse_candidates(raw: list[dict[str, Any]] | None) -> list[Candidate]:
    if raw is not None and len(raw) == 0:
        raise ValueError(
            "automl_candidates is an empty list; omit [job.automl.candidates] entirely to use the "
            "default search space, or provide at least one [[job.automl.candidates]] entry"
        )
    parsed = []
    for entry in raw if raw is not None else DEFAULT_CANDIDATES:
        models_value = entry["models"]
        if isinstance(models_value, str) or not isinstance(models_value, (list, tuple)):  # noqa: UP038
            raise ValueError(f"automl candidate 'models' must be a list of model names, got {models_value!r}")
        if not all(isinstance(name, str) for name in models_value):
            raise ValueError(f"automl candidate 'models' must contain only strings, got {models_value!r}")
        models = list(models_value)
        if not models:
            raise ValueError("automl candidate 'models' must not be empty")
        unknown = [name for name in models if name not in STRATEGIES]
        if unknown:
            raise ValueError(
                f"Unknown model(s) in automl candidate {unknown}; available: {sorted(STRATEGIES)}"
            )  # noqa: E501
        weights_value = entry.get("weights")
        if weights_value is not None and not isinstance(weights_value, dict):
            raise ValueError(
                f"automl candidate 'weights' must be a table of model name -> weight, got {weights_value!r}"
            )
        weights = {str(k): float(v) for k, v in weights_value.items()} if weights_value is not None else None
        if weights is not None:
            unknown_weights = [name for name in weights if name not in models]
            if unknown_weights:
                raise ValueError(f"automl candidate weight key(s) {unknown_weights} not in models {models}")
            missing_weights = [name for name in models if name not in weights]
            if missing_weights:
                raise ValueError(
                    f"automl candidate weights missing model(s) {missing_weights}; "
                    f"provide an explicit weight for every model in {models}, "
                    "or omit weights entirely for equal (priority) weighting"
                )
            validate_model_weights(weights, context="automl candidate weights")
        rrf_k = float(entry["rrf_k"]) if "rrf_k" in entry else None
        validate_rrf_k(rrf_k, context="automl candidate rrf_k")
        parsed.append(
            Candidate(
                models=models,
                weights=weights,
                rrf_k=rrf_k,
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
    # +1us so the mask's strict "< test_end" bound doesn't exclude the very
    # last event: without this, an event exactly at max_ts would fall
    # outside every fold (i=0's test_end is max_ts itself). Shifting every
    # boundary by the same 1us keeps folds non-overlapping and doesn't
    # meaningfully change day-sized windows.
    max_ts = occurred_at.max() + pd.Timedelta(microseconds=1)
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
    if len(folds) < n_splits:
        logger.warning(
            "AutoML requested %d fold(s) of %d day(s) each but only %d had enough event history; "
            "backtest coverage is reduced",
            n_splits,
            test_days,
            len(folds),
        )

    metrics = _make_metrics(top_k)
    fold_metrics_by_candidate: list[list[dict[str, float]]] = [[] for _ in parsed_candidates]
    for train_events, test_events in folds:
        # Built once per fold and reused across every candidate below: the
        # dataset only depends on the fold's events/users/items, not on which
        # strategies/weights a candidate combines. The test side only needs
        # interactions for calc_metrics, not a full feature-laden Dataset, so
        # build_interactions() is used there instead of build_dataset().
        built = build_dataset(train_events, users, items, config, half_life_days=half_life_days)
        test_interactions = build_interactions(test_events, config, half_life_days=half_life_days)
        test_users = sorted(set(test_events["user_id"]))
        # Reset per fold (a fold's fitted models are only valid for that
        # fold's built dataset) but shared across every candidate within
        # this fold: candidates that enable the same strategy (e.g.
        # "popular" alone and a fusion candidate that also enables
        # "popular") reuse its fitted model -- refitting is skipped, but
        # recommend() still runs fresh each time, so this works even across
        # candidates with different top_k/weights.
        strategy_cache: dict[str, RecommenderModel] = {}
        for idx, candidate in enumerate(parsed_candidates):
            reco = train_and_recommend(
                built,
                test_users,
                config,
                top_k=top_k,
                enabled_models=candidate.models,
                weights=candidate.weights,
                rrf_k=candidate.rrf_k,
                strategy_cache=strategy_cache,
            )
            fold_metrics_by_candidate[idx].append(
                calc_metrics(metrics, reco=reco, interactions=test_interactions)
            )

    results = []
    for candidate, fold_metrics in zip(parsed_candidates, fold_metrics_by_candidate, strict=True):
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

    # Validated once, up front, against the first candidate's metric keys
    # (every candidate shares the same metric set -- see _make_metrics) so a
    # typo'd/mismatched primary_metric fails immediately with a clear error
    # instead of only surfacing after `max` has already iterated into
    # `_metric_value` for some candidate.
    available_metrics = list(results[0].metrics)
    if not any(key.startswith(primary_metric) for key in available_metrics):
        raise ValueError(
            f"No metric starting with '{primary_metric}' found; available metrics: {available_metrics}"
        )

    def _metric_value(result: CandidateResult) -> float:
        metric_key = next((key for key in result.metrics if key.startswith(primary_metric)), None)
        if metric_key is None:
            raise ValueError(
                f"No metric starting with '{primary_metric}' found for candidate "
                f"'{result.candidate.label}': {list(result.metrics)}"
            )
        return result.metrics[metric_key]

    return max(results, key=_metric_value)
