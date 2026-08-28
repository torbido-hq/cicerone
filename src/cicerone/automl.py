"""Time-fold backtest of strategy/weight candidates; picks the best ranking metric.

Custom event split (not rectools splitters) so each fold rebuilds BuiltDataset
and runs train_and_recommend.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from typing import Any

import pandas as pd
from rectools.metrics import MAP, NDCG, Recall, calc_metrics
from rectools.metrics.base import MetricAtK
from rectools.metrics.debias import DebiasConfig

from cicerone.config import (
    AUTOML_DEFAULT_N_SPLITS,
    AUTOML_DEFAULT_PRIMARY_METRIC,
    AUTOML_DEFAULT_TEST_DAYS,
    DEFAULT_SEQUENTIAL_MIN_MEDIAN_INTERACTIONS,
    validate_model_weights,
    validate_rrf_k,
)
from cicerone.config.settings import ExplainSettings
from cicerone.dataset import build_dataset, build_interactions
from cicerone.feature_config import FeatureConfig
from cicerone.model import (
    DEFAULT_MODELS,
    STRATEGIES,
    RecommenderModel,
    train_and_recommend,
)
from cicerone.model.constants import RANDOM_STATE
from cicerone.model_config import SEQUENTIAL_EXTRA_HINT, SEQUENTIAL_STRATEGY, sequential_extra_available

logger = logging.getLogger(__name__)

DEFAULT_N_SPLITS = AUTOML_DEFAULT_N_SPLITS
DEFAULT_TEST_DAYS = AUTOML_DEFAULT_TEST_DAYS
DEFAULT_PRIMARY_METRIC = AUTOML_DEFAULT_PRIMARY_METRIC


def median_distinct_items_per_user(events: pd.DataFrame) -> float:
    """Median number of distinct items per user in ``events`` (0 if empty)."""
    if events.empty or "user_id" not in events.columns or "item_id" not in events.columns:
        return 0.0
    counts = events.groupby("user_id")["item_id"].nunique()
    if counts.empty:
        return 0.0
    return float(counts.median())


def sequential_automl_skip_reason(
    events: pd.DataFrame,
    *,
    min_median_interactions: int,
) -> str | None:
    """Why sequential should be dropped from AutoML, or ``None`` to keep it."""
    if not sequential_extra_available():
        return f"optional sequential extra is not installed ({SEQUENTIAL_EXTRA_HINT})"
    median = median_distinct_items_per_user(events)
    if median < min_median_interactions:
        return (
            f"median distinct items per user is {median:.1f}, "
            f"below job.sequential.min_median_interactions={min_median_interactions}"
        )
    return None


def exclude_sequential_from_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Drop sequential from each candidate; omit candidates that become empty."""
    result: list[Candidate] = []
    for candidate in candidates:
        if SEQUENTIAL_STRATEGY not in candidate.models:
            result.append(candidate)
            continue
        models = [name for name in candidate.models if name != SEQUENTIAL_STRATEGY]
        if not models:
            continue
        weights = None
        if candidate.weights is not None:
            weights = {
                name: weight for name, weight in candidate.weights.items() if name != SEQUENTIAL_STRATEGY
            } or None
        result.append(Candidate(models=models, weights=weights, rrf_k=candidate.rrf_k))
    return result


# From STRATEGIES: each alone, DEFAULT_MODELS priority, and one weighted fusion of all.
DEFAULT_CANDIDATES: list[dict[str, Any]] = [
    *({"models": [name]} for name in STRATEGIES),
    {"models": DEFAULT_MODELS},
    {
        "models": list(STRATEGIES),
        "weights": {name: (1.0 if strategy.personalized else 0.3) for name, strategy in STRATEGIES.items()},
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
            base = "+".join(self.models)
        else:
            weighted = ",".join(f"{name}={self.weights[name]}" for name in self.models)
            base = f"fusion({weighted})"
        if self.rrf_k is None:
            return base
        return f"{base};rrf_k={self.rrf_k}"


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
        if not isinstance(models_value, (list, tuple)):
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
            )
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
    `test_days`-day windows, each becoming one (train, test) fold. Folds are
    returned oldest-test-window-first; a fold is skipped if either side is empty.
    """
    occurred_at = pd.to_datetime(events["occurred_at"], utc=True)
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


def _make_metrics(top_k: int, *, debias: bool = False) -> dict[str, MetricAtK]:
    # rectools.metrics — not a custom MAP/NDCG/Recall implementation.
    debias_config = DebiasConfig(random_state=RANDOM_STATE) if debias else None
    return {
        f"MAP@{top_k}": MAP(k=top_k, debias_config=debias_config),
        f"NDCG@{top_k}": NDCG(k=top_k, debias_config=debias_config),
        f"Recall@{top_k}": Recall(k=top_k, debias_config=debias_config),
    }


def _evaluate_fold(
    train_events: pd.DataFrame,
    test_events: pd.DataFrame,
    users: pd.DataFrame | None,
    items: pd.DataFrame | None,
    config: FeatureConfig,
    top_k: int,
    half_life_days: float,
    candidates: list[Candidate],
    metrics: dict[str, MetricAtK],
    model_configs: dict[str, dict[str, Any]] | None = None,
    content_fallback_enabled: bool = False,
) -> list[dict[str, float]]:
    """Score every candidate on one fold (picklable for ProcessPoolExecutor)."""
    built = build_dataset(train_events, users, items, config, half_life_days=half_life_days)
    test_interactions = build_interactions(test_events, config, half_life_days=half_life_days)
    test_users = sorted(set(test_events["user_id"]))
    strategy_cache: dict[str, RecommenderModel] = {}
    recommend_cache: dict[tuple[Any, ...], Any] = {}

    fold_metrics = []
    for candidate in candidates:
        reco = train_and_recommend(
            built,
            test_users,
            config,
            top_k=top_k,
            enabled_models=candidate.models,
            weights=candidate.weights,
            rrf_k=candidate.rrf_k,
            strategy_cache=strategy_cache,
            model_configs=model_configs,
            recommend_cache=recommend_cache,
            explain=ExplainSettings(enabled=False),
            content_fallback_enabled=content_fallback_enabled,
        )
        fold_metrics.append(calc_metrics(metrics, reco=reco, interactions=test_interactions))
    return fold_metrics


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
    max_workers: int = 1,
    model_configs: dict[str, dict[str, Any]] | None = None,
    sequential_min_median_interactions: int = DEFAULT_SEQUENTIAL_MIN_MEDIAN_INTERACTIONS,
    debias: bool = False,
    content_fallback_enabled: bool = False,
) -> list[CandidateResult]:
    """Backtest candidates over time folds. ``max_workers > 1`` evaluates folds in parallel."""
    parsed_candidates = _parse_candidates(candidates)
    if any(SEQUENTIAL_STRATEGY in candidate.models for candidate in parsed_candidates):
        skip_reason = sequential_automl_skip_reason(
            events, min_median_interactions=sequential_min_median_interactions
        )
        if skip_reason is not None:
            logger.info("Excluding sequential from AutoML candidate pool: %s", skip_reason)
            parsed_candidates = exclude_sequential_from_candidates(parsed_candidates)
            if not parsed_candidates:
                raise ValueError("AutoML has no candidates left after excluding sequential; " + skip_reason)
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

    metrics = _make_metrics(top_k, debias=debias)
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(folds))) as executor:
            fold_results = list(
                executor.map(
                    _evaluate_fold,
                    (train_events for train_events, _ in folds),
                    (test_events for _, test_events in folds),
                    repeat(users),
                    repeat(items),
                    repeat(config),
                    repeat(top_k),
                    repeat(half_life_days),
                    repeat(parsed_candidates),
                    repeat(metrics),
                    repeat(model_configs),
                    repeat(content_fallback_enabled),
                )
            )
    else:
        fold_results = [
            _evaluate_fold(
                train_events,
                test_events,
                users,
                items,
                config,
                top_k,
                half_life_days,
                parsed_candidates,
                metrics,
                model_configs,
                content_fallback_enabled,
            )
            for train_events, test_events in folds
        ]

    fold_metrics_by_candidate: list[list[dict[str, float]]] = [[] for _ in parsed_candidates]
    for fold_metrics in fold_results:
        for idx, candidate_metrics in enumerate(fold_metrics):
            fold_metrics_by_candidate[idx].append(candidate_metrics)

    results = []
    for candidate, fold_metrics in zip(parsed_candidates, fold_metrics_by_candidate, strict=True):
        averaged = dict(pd.DataFrame(fold_metrics).mean()) if fold_metrics else dict.fromkeys(metrics, 0.0)
        results.append(CandidateResult(candidate=candidate, metrics=averaged, n_folds=len(fold_metrics)))
        logger.info(
            "AutoML candidate '%s' scored %s over %d fold(s)", candidate.label, averaged, len(fold_metrics)
        )
    return results


def _resolve_metric_key(available_metrics: list[str], primary_metric: str) -> str:
    if primary_metric in available_metrics:
        return primary_metric
    matches = [key for key in available_metrics if key.startswith(f"{primary_metric}@")]
    if not matches:
        raise ValueError(
            f"No metric matching '{primary_metric}' found; available metrics: {available_metrics}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous primary_metric '{primary_metric}' matches {matches}; "
            "use an exact metric name (e.g. 'MAP@10')"
        )
    return matches[0]


def select_best_candidate(
    results: list[CandidateResult], primary_metric: str = DEFAULT_PRIMARY_METRIC
) -> CandidateResult:
    """Pick the highest ``primary_metric`` (exact name or ``NAME@k``). Ties keep list order."""
    if not results:
        raise ValueError("No candidate results to select from")

    metric_key = _resolve_metric_key(list(results[0].metrics), primary_metric)

    def _metric_value(result: CandidateResult) -> float:
        if metric_key not in result.metrics:
            raise ValueError(
                f"Metric '{metric_key}' missing for candidate '{result.candidate.label}': "
                f"{list(result.metrics)}"
            )
        return result.metrics[metric_key]

    return max(results, key=_metric_value)
