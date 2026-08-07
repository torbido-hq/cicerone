"""Trains one or more recommendation strategies (see STRATEGIES) and combines
their outputs into top-K recommendations per user, with a non-personalized
fallback for cold-start users who have too little (or no) personal signal.
"""

from __future__ import annotations

import inspect
import logging
import random
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from itertools import repeat
from typing import Any, Protocol

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset
from rectools.metrics import Precision, Recall, calc_metrics
from rectools.models import model_from_config

from cicerone.blending import (
    COLD_START_USER_ID,
    PERSONALIZED_SOURCE,
    POPULAR_SOURCE,
    append_cold_start_rows,
    blend_for_users,
    expand_latest_ranking,
    interaction_counts,
    rank_latest_items,
    resolve_latest_date_column,
)
from cicerone.config import (
    DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
    STRATEGY_NAMES,
    EpochMetricsSettings,
    validate_model_weights,
    validate_rrf_k,
)
from cicerone.content_fallback import (
    CONTENT_FALLBACK_SOURCE,
    ContentFallbackModel,
    build_content_fallback_model,
)
from cicerone.dataset import BuiltDataset
from cicerone.feature_config import DEFAULT_BOOST_OVERFETCH_FACTOR, FeatureColumn, FeatureConfig
from cicerone.ids import interacting_external_user_ids
from cicerone.model_config import (
    RECTOOLS_STRATEGY_NAMES,
    default_model_configs,
    resolve_model_configs,
)
from cicerone.policy import (
    allowed_items_for_cohort,
    apply_boosts,
    group_users_by_cohort,
    has_user_scoped_eligibility,
    index_users_by_id,
    is_user_scoped,
    resolve_eligibility,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_MODELS = ["collaborative", "item_based", "popular"]
LATEST_WINDOW_DAYS = 14
# Reciprocal rank fusion constant (Cormack et al., 2009); default for rrf_k.
RRF_K = 60
SOURCE_COLUMN = "source"
WEIGHT_COLUMN = "_weight"  # internal-only; dropped before returning to callers

COLLABORATIVE_EPOCHS = 30  # LightFMWrapperModel.fit() runs these in one fit_partial
# ProcessPool: LightFM num_threads=1 to avoid workers × BLAS oversubscription.
_LIGHTFM_NUM_THREADS_SEQUENTIAL = 4
_LIGHTFM_NUM_THREADS_PARALLEL = 1
_EPOCH_METRICS_RNG = random.Random()
_FIT_POOL_LIGHTFM_THREADS = _LIGHTFM_NUM_THREADS_SEQUENTIAL


def _recommend_k(top_k: int, has_boosts: bool, overfetch_factor: int = DEFAULT_BOOST_OVERFETCH_FACTOR) -> int:
    if not has_boosts:
        return top_k
    factor = overfetch_factor if overfetch_factor >= 1 else DEFAULT_BOOST_OVERFETCH_FACTOR
    return max(top_k, top_k * factor)


class RecommenderModel(Protocol):
    def fit(self, dataset: Dataset) -> object: ...

    def recommend(
        self,
        *,
        users: list[str],
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        items_to_recommend: list | None = None,
    ) -> pd.DataFrame: ...


_RECOMMEND_PARAMS = {"users", "dataset", "k", "filter_viewed", "items_to_recommend"}


def _as_recommender_model(model: object) -> RecommenderModel:
    """Fail fast if `model` does not implement RecommenderModel."""
    fit = getattr(model, "fit", None)
    recommend = getattr(model, "recommend", None)
    if not callable(fit) or not callable(recommend):
        raise TypeError(
            f"{type(model).__name__} does not implement the RecommenderModel protocol "
            "(missing a callable fit() and/or recommend())"
        )
    recommend_params = set(inspect.signature(recommend).parameters)
    missing_params = _RECOMMEND_PARAMS - recommend_params
    if missing_params:
        raise TypeError(
            f"{type(model).__name__}.recommend() is missing expected parameter(s) {sorted(missing_params)}; "
            "the RecommenderModel protocol may have drifted from the installed rectools/implicit version"
        )
    return model  # type: ignore[return-value]


@dataclass(frozen=True)
class Strategy:
    personalized: bool
    source_label: str
    # Item-KNN / content_fallback need history; LightFM hybrid can score feature-only users.
    requires_interactions: bool = False
    # Optional factory override (tests / content_fallback placeholder). When set,
    # skips RecTools model_from_config for this strategy.
    factory: Callable[[], RecommenderModel] | None = None


def build_strategy_model(
    name: str,
    *,
    model_configs: dict[str, dict[str, Any]] | None = None,
    lightfm_num_threads: int | None = None,
) -> RecommenderModel:
    """Instantiate one RecTools strategy via ``model_from_config``.

    ``content_fallback`` is not RecTools-backed — use ``build_content_fallback_model``.
    """
    if name not in RECTOOLS_STRATEGY_NAMES:
        raise ValueError(
            f"Cannot build {name!r} via RecTools config; available: {list(RECTOOLS_STRATEGY_NAMES)}"
        )
    configs = model_configs if model_configs is not None else default_model_configs()
    if name not in configs:
        raise ValueError(f"No model config for strategy {name!r}; have {sorted(configs)}")
    cfg = deepcopy(configs[name])
    if name == "collaborative":
        threads = lightfm_num_threads if lightfm_num_threads is not None else _FIT_POOL_LIGHTFM_THREADS
        cfg["num_threads"] = threads
    return _as_recommender_model(model_from_config(cfg))


def _build_content_fallback() -> RecommenderModel:
    """Placeholder factory; real instances are built in ``_fit_strategy`` with items/features."""
    return _as_recommender_model(ContentFallbackModel())


STRATEGIES: dict[str, Strategy] = {
    "collaborative": Strategy(personalized=True, source_label="personalized"),
    "item_based": Strategy(
        personalized=True,
        source_label="item_based",
        requires_interactions=True,
    ),
    "content_fallback": Strategy(
        factory=_build_content_fallback,
        personalized=True,
        source_label=CONTENT_FALLBACK_SOURCE,
        requires_interactions=True,
    ),
    "popular": Strategy(personalized=False, source_label="popular_fallback"),
    "latest": Strategy(personalized=False, source_label="latest"),
}


def _should_log_epoch(epoch: int, total_epochs: int, every: int) -> bool:
    return epoch == 1 or epoch == total_epochs or epoch % every == 0


def _sample_epoch_metric_users(external_ids, max_users: int) -> list:
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


def _epoch_metric_total_epochs(model: object) -> int:
    # rectools stores epochs as n_epochs; accept epochs for other wrappers.
    for attr in ("n_epochs", "epochs"):
        value = getattr(model, attr, None)
        if value is not None:
            return int(value)
    raise TypeError(
        f"{type(model).__name__} has no n_epochs/epochs attribute; "
        "epoch metric logging needs a known epoch count"
    )


def _warn_on_epoch_metric_trajectory(
    history: list[tuple[int, dict[str, float]]], settings: EpochMetricsSettings
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
                "Collaborative epoch metrics: %s regressed from best %.4f to final %.4f "
                "(drop >= %.0f%% across logged epochs)",
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
                    "Collaborative epoch metrics: %s plateaued near %.4f over the last %d "
                    "logged snapshots (span %.4f)",
                    metric_name,
                    recent[-1],
                    settings.plateau_window,
                    span,
                )


def _interactions_for_epoch_metrics(
    dataset: Dataset, interactions: pd.DataFrame, max_users: int
) -> pd.DataFrame:
    users = _sample_epoch_metric_users(dataset.user_id_map.external_ids, max_users)
    return interactions[interactions[Columns.User].isin(users)]


def _fit_lightfm_with_epoch_metrics(
    model: RecommenderModel,
    dataset: Dataset,
    interactions: pd.DataFrame,
    settings: EpochMetricsSettings,
    top_k: int,
) -> RecommenderModel:
    """Epoch-by-epoch LightFM fit with in-sample Precision/Recall@K logs.

    ``interactions`` should already be limited to the scored-user subset.
    Scores with filter_viewed=False (trajectory signal, not holdout).
    """
    fit_partial = _epoch_metric_fit_partial(model)
    total_epochs = _epoch_metric_total_epochs(model)
    metric_defs = {
        f"Precision@{top_k}": Precision(k=top_k),
        f"Recall@{top_k}": Recall(k=top_k),
    }
    users = list(dict.fromkeys(interactions[Columns.User].tolist()))
    history: list[tuple[int, dict[str, float]]] = []

    for epoch in range(1, total_epochs + 1):
        fit_partial(dataset, 1)
        if not _should_log_epoch(epoch, total_epochs, settings.every):
            continue
        reco = model.recommend(
            users=users,
            dataset=dataset,
            k=top_k,
            filter_viewed=False,
        )
        snapshot = calc_metrics(metric_defs, reco=reco, interactions=interactions)
        history.append((epoch, snapshot))
        logger.info("Collaborative epoch %d/%d metrics: %s", epoch, total_epochs, snapshot)

    _warn_on_epoch_metric_trajectory(history, settings)
    return model


def _validate_strategy_names(strategies: dict[str, Strategy], strategy_names: tuple[str, ...]) -> None:
    """Raises if STRATEGIES' keys and cicerone.config.STRATEGY_NAMES drift apart."""
    if set(strategies) != set(strategy_names):
        raise RuntimeError(
            f"cicerone.model.STRATEGIES keys {sorted(strategies)} must match "
            f"cicerone.config.STRATEGY_NAMES {sorted(strategy_names)} — update both together"
        )


_validate_strategy_names(STRATEGIES, STRATEGY_NAMES)


def _combine_by_priority(frames: list[pd.DataFrame], top_k: int) -> pd.DataFrame:
    """Fill top-K from strategy outputs in list order: earlier strategies keep
    duplicate (user, item) pairs and fill slots before later ones.
    """
    tagged = []
    for priority, frame in enumerate(frames):
        part = frame.copy()
        part["_priority"] = priority
        tagged.append(part)
    combined = pd.concat(tagged, ignore_index=True)
    combined = combined.drop_duplicates(subset=[Columns.User, Columns.Item], keep="first")
    combined = combined.sort_values([Columns.User, "_priority", Columns.Rank])
    combined = combined.groupby(Columns.User, as_index=False).head(top_k)
    combined[Columns.Rank] = combined.groupby(Columns.User).cumcount() + 1
    return combined.drop(columns=[WEIGHT_COLUMN, "_priority"])


def _combine_by_weighted_fusion(
    frames: list[pd.DataFrame], top_k: int, rrf_k: float, source_label_order: list[str]
) -> pd.DataFrame:
    """Weighted reciprocal rank fusion: each strategy's contribution to an
    item's fused score is `weight / (rrf_k + rank)`, summed across every
    strategy that recommended that (user, item) pair. Combined source labels
    are joined in `source_label_order` rather than alphabetically.
    """
    combined = pd.concat(frames, ignore_index=True)
    combined[Columns.Score] = combined[WEIGHT_COLUMN] / (rrf_k + combined[Columns.Rank])

    def _join_labels_in_order(labels: pd.Series) -> str:
        present = set(labels)
        return "+".join(label for label in source_label_order if label in present)

    fused = combined.groupby([Columns.User, Columns.Item], as_index=False).agg(
        **{
            Columns.Score: (Columns.Score, "sum"),
            SOURCE_COLUMN: (SOURCE_COLUMN, _join_labels_in_order),
        }
    )
    fused = fused.sort_values([Columns.User, Columns.Score], ascending=[True, False])
    fused[Columns.Rank] = fused.groupby(Columns.User).cumcount() + 1
    fused = fused.groupby(Columns.User, as_index=False).head(top_k)
    return fused[[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN]]


def _init_fit_worker(lightfm_threads: int) -> None:
    """ProcessPool initializer: keep LightFM single-threaded inside workers."""
    global _FIT_POOL_LIGHTFM_THREADS
    _FIT_POOL_LIGHTFM_THREADS = lightfm_threads


def _resolve_model_configs_for_fit(
    model_configs: dict[str, dict[str, Any]] | None,
    item_based_k_neighbors: int | None,
) -> dict[str, dict[str, Any]]:
    """Apply optional k_neighbors override onto a copy of model configs."""
    if model_configs is None and item_based_k_neighbors is None:
        return default_model_configs()
    if model_configs is None:
        return resolve_model_configs(
            legacy_k_neighbors=item_based_k_neighbors,
            legacy_k_neighbors_explicit=item_based_k_neighbors is not None,
        )
    configs = {name: deepcopy(cfg) for name, cfg in model_configs.items()}
    if item_based_k_neighbors is not None:
        item_cfg = configs.setdefault("item_based", deepcopy(default_model_configs()["item_based"]))
        model_section = item_cfg.setdefault("model", {})
        model_section["K"] = int(item_based_k_neighbors)
        configs["item_based"] = item_cfg
    return configs


def _fit_strategy(
    name: str,
    dataset: Dataset,
    epoch_interactions: pd.DataFrame | None,
    epoch_metrics: EpochMetricsSettings | None,
    epoch_metrics_top_k: int,
    model_configs: dict[str, dict[str, Any]] | None = None,
    content_feature_columns: list[FeatureColumn] | None = None,
    content_max_neighbors: int = DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
    content_items: pd.DataFrame | None = None,
    content_interactions: pd.DataFrame | None = None,
) -> tuple[str, RecommenderModel]:
    """Fit one strategy (picklable for ProcessPoolExecutor workers)."""
    strategy = STRATEGIES[name]
    if name == "content_fallback":
        model = _as_recommender_model(
            build_content_fallback_model(
                feature_columns=content_feature_columns or [],
                max_neighbors=content_max_neighbors,
                items=content_items,
                interactions=content_interactions,
            )
        )
    elif strategy.factory is not None:
        model = strategy.factory()
    else:
        model = build_strategy_model(
            name,
            model_configs=model_configs,
            lightfm_num_threads=_FIT_POOL_LIGHTFM_THREADS,
        )
    if name == "collaborative" and epoch_metrics is not None:
        if epoch_interactions is None:
            raise ValueError("epoch_interactions is required when epoch metric logging is enabled")
        _fit_lightfm_with_epoch_metrics(
            model, dataset, epoch_interactions, settings=epoch_metrics, top_k=epoch_metrics_top_k
        )
    else:
        model.fit(dataset)
    return name, model


def _resolve_enabled_models(enabled_models: list[str] | None) -> list[str]:
    resolved = enabled_models if enabled_models is not None else DEFAULT_MODELS
    if not resolved:
        raise ValueError(
            "enabled_models is empty; provide at least one model name, or omit enabled_models/pass None "
            "to use the default"
        )
    unknown_models = [name for name in resolved if name not in STRATEGIES]
    if unknown_models:
        raise ValueError(f"Unknown model(s) {unknown_models}; available: {sorted(STRATEGIES)}")
    return resolved


def content_fallback_enabled_from_models(models: list[str] | tuple[str, ...] | None) -> bool:
    """Whether a stored/candidate model list should run content_fallback.

    Shared by artifact replay and AutoML: presence of ``content_fallback`` in
    the list means the tier was active when that list was produced (config
    ``enabled`` already applied at list-build time).
    """
    if not models:
        return False
    return "content_fallback" in models


@dataclass(frozen=True)
class ModelRunPlan:
    """Resolved model lists for one train/recommend run.

    Built once via ``plan_model_run`` so content_fallback / blending adjustments
    are not re-derived at every call site.
    """

    enabled_models: tuple[str, ...]
    recommend_models: tuple[str, ...]

    @property
    def content_fallback_active(self) -> bool:
        return "content_fallback" in self.recommend_models


def plan_model_run(
    enabled_models: list[str] | None,
    *,
    blending_enabled: bool,
    content_fallback_enabled: bool | None = None,
) -> ModelRunPlan:
    """Resolve enabled + recommend model lists for a run.

    When ``content_fallback_enabled`` is omitted, it is derived from whether
    ``content_fallback`` is already present in the requested model list
    (artifact / AutoML replay). Job runs pass the config flag explicitly.
    """
    resolved = _resolve_enabled_models(enabled_models)
    if content_fallback_enabled is None:
        content_fallback_enabled = content_fallback_enabled_from_models(resolved)
    recommend = resolve_recommend_models(
        resolved, blending_enabled, content_fallback_enabled=content_fallback_enabled
    )
    return ModelRunPlan(enabled_models=tuple(resolved), recommend_models=tuple(recommend))


def resolve_run_models(
    enabled_models: list[str] | None,
    *,
    blending_enabled: bool,
    content_fallback_enabled: bool = False,
) -> tuple[list[str], list[str]]:
    """Resolve requested models and the effective fit/recommend list.

    Prefer ``plan_model_run`` for new call sites; this returns plain lists for
    existing callers/tests.
    """
    plan = plan_model_run(
        enabled_models,
        blending_enabled=blending_enabled,
        content_fallback_enabled=content_fallback_enabled,
    )
    return list(plan.enabled_models), list(plan.recommend_models)


def fit_strategies(
    built: BuiltDataset,
    target_users: list[str],
    enabled_models: list[str] | None = None,
    strategy_cache: dict[str, RecommenderModel] | None = None,
    max_workers: int = 1,
    epoch_metrics: EpochMetricsSettings | None = None,
    epoch_metrics_top_k: int = 10,
    item_based_k_neighbors: int | None = None,
    model_configs: dict[str, dict[str, Any]] | None = None,
    content_fallback_max_neighbors: int = DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
    content_feature_columns: list[FeatureColumn] | None = None,
) -> tuple[list[str], dict[str, RecommenderModel]]:
    """Fit (or cache-hit) enabled strategies. ``max_workers > 1`` fits in parallel.

    ``epoch_metrics`` enables collaborative fit_partial logging; default
    ``None`` keeps a single LightFM ``fit()``.

    Models are built via RecTools ``model_from_config`` using ``model_configs``
    (defaults from ``cicerone.model_config``). ``item_based_k_neighbors`` remains
    as a convenience override for the legacy ``job.item_based.k_neighbors`` knob.
    """
    dataset = built.dataset
    enabled_models = _resolve_enabled_models(enabled_models)
    resolved_configs = _resolve_model_configs_for_fit(model_configs, item_based_k_neighbors)

    known_users = set(dataset.user_id_map.external_ids)
    warm_users = [u for u in target_users if u in known_users]
    cold_users = [u for u in target_users if u not in known_users]
    if cold_users:
        if any(not STRATEGIES[name].personalized for name in enabled_models):
            logger.info(
                "%d/%d users have no usable signal yet; falling back to non-personalized strategies for them",
                len(cold_users),
                len(target_users),
            )
        else:
            logger.info(
                "%d/%d users have no usable signal yet and no non-personalized strategy is "
                "enabled; they will receive no recommendations",
                len(cold_users),
                len(target_users),
            )

    models: dict[str, RecommenderModel] = {}
    if strategy_cache is not None:
        for name in enabled_models:
            if name in strategy_cache:
                models[name] = strategy_cache[name]

    to_fit = list(
        dict.fromkeys(
            name
            for name in enabled_models
            if name not in models and not (STRATEGIES[name].personalized and not warm_users)
        )
    )
    # Pre-slice interactions in the parent to shrink ProcessPool pickles.
    epoch_interactions = (
        _interactions_for_epoch_metrics(dataset, built.interactions, epoch_metrics.max_users)
        if epoch_metrics is not None and "collaborative" in to_fit
        else None
    )
    content_cols = list(content_feature_columns or [])
    content_items = built.items if "content_fallback" in to_fit else None
    content_interactions = built.interactions if "content_fallback" in to_fit else None
    if to_fit:
        if max_workers > 1 and len(to_fit) > 1:
            with ProcessPoolExecutor(
                max_workers=min(max_workers, len(to_fit)),
                initializer=_init_fit_worker,
                initargs=(_LIGHTFM_NUM_THREADS_PARALLEL,),
            ) as executor:
                for name, model in executor.map(
                    _fit_strategy,
                    to_fit,
                    repeat(dataset),
                    repeat(epoch_interactions),
                    repeat(epoch_metrics),
                    repeat(epoch_metrics_top_k),
                    repeat(resolved_configs),
                    repeat(content_cols),
                    repeat(content_fallback_max_neighbors),
                    repeat(content_items),
                    repeat(content_interactions),
                ):
                    logger.info("Fitted '%s' on %d interactions", name, len(built.interactions))
                    models[name] = model
        else:
            for name in to_fit:
                logger.info("Fitting '%s' on %d interactions", name, len(built.interactions))
                _, model = _fit_strategy(
                    name,
                    dataset,
                    epoch_interactions,
                    epoch_metrics,
                    epoch_metrics_top_k,
                    resolved_configs,
                    content_cols,
                    content_fallback_max_neighbors,
                    content_items,
                    content_interactions,
                )
                models[name] = model
        if strategy_cache is not None:
            for name in to_fit:
                strategy_cache[name] = models[name]

    return enabled_models, models


def resolve_recommend_models(
    enabled_models: list[str],
    blending_enabled: bool,
    content_fallback_enabled: bool = False,
) -> list[str]:
    """Models to fit/recommend for a run.

    When ``content_fallback_enabled`` is true and the strategy is not already
    listed, insert it immediately before the first non-personalized strategy.
    When false, drop it even if listed (with a log line).

    When blending is on: ensure ``popular`` is present, and drop strategy
    ``latest`` (trending PopularModel) — blending's date-based ``latest`` is
    built from items, not that strategy.
    """
    models = list(enabled_models)
    if not content_fallback_enabled:
        if "content_fallback" in models:
            logger.info(
                "content_fallback is listed in models but job.content_fallback.enabled is false — skipping"
            )
            models = [name for name in models if name != "content_fallback"]
    elif "content_fallback" not in models:
        insert_at = len(models)
        for index, name in enumerate(models):
            if name in STRATEGIES and not STRATEGIES[name].personalized:
                insert_at = index
                break
        models.insert(insert_at, "content_fallback")

    if not blending_enabled:
        return models
    if "latest" in models:
        models = [name for name in models if name != "latest"]
    if "popular" not in models:
        models.append("popular")
    return models


@dataclass(frozen=True)
class _CohortPlan:
    cohorts: list[tuple[object, list[str]]]
    allowed_by_cohort: dict[object, list]
    eligibility: list
    users_frame: pd.DataFrame | None
    known_users: set
    interacting_users: set
    has_any_warm_user: bool
    unique_target_users: list[str]
    all_item_ids: Sequence
    has_boosts: bool
    recommend_k: int


@dataclass(frozen=True)
class _StrategyFrames:
    frames: list[pd.DataFrame]
    personalized_frames: list[pd.DataFrame]
    popular_frames: list[pd.DataFrame]
    latest_by_cohort: dict[object, list[tuple[str, int, float]]]
    date_column: str | None
    latest_available: bool


def _resolve_cohort_plan(
    built: BuiltDataset,
    config: FeatureConfig,
    target_users: list[str],
    recommend_models: list[str],
    top_k: int,
) -> _CohortPlan:
    dataset = built.dataset
    all_item_ids = dataset.item_id_map.external_ids
    eligibility = resolve_eligibility(config)
    users_frame = built.users if built.users is not None and not built.users.empty else None
    if has_user_scoped_eligibility(eligibility) and users_frame is None:
        logger.warning(
            "User-scoped eligibility rules are configured but no users frame is available — "
            "applying only item-global rules"
        )
        eligibility = [r for r in eligibility if not is_user_scoped(r)]
    use_cohorts = has_user_scoped_eligibility(eligibility) and users_frame is not None
    has_boosts = bool(config.boosts)
    recommend_k = _recommend_k(top_k, has_boosts, config.boost_overfetch_factor)

    known_users = set(dataset.user_id_map.external_ids)
    unique_target_users = list(dict.fromkeys(target_users))
    has_any_warm_user = bool(known_users.intersection(unique_target_users))
    needs_interacting_users = any(
        STRATEGIES[name].requires_interactions for name in recommend_models if name in STRATEGIES
    )
    interacting_users = interacting_external_user_ids(built) if needs_interacting_users else set()

    users_by_id = index_users_by_id(users_frame)
    if use_cohorts:
        cohorts = group_users_by_cohort(
            unique_target_users, users_frame, eligibility, users_by_id=users_by_id
        )
        allowed_by_cohort = {
            key: allowed_items_for_cohort(
                cohort_users,
                users_frame,
                built.items,
                eligibility,
                all_item_ids,
                users_by_id=users_by_id,
            )
            for key, cohort_users in cohorts
        }
    else:
        allowed_by_cohort = {
            None: allowed_items_for_cohort(
                unique_target_users,
                users_frame,
                built.items,
                eligibility,
                all_item_ids,
                users_by_id=users_by_id,
            )
        }
        cohorts = [(None, unique_target_users)]

    return _CohortPlan(
        cohorts=cohorts,
        allowed_by_cohort=allowed_by_cohort,
        eligibility=eligibility,
        users_frame=users_frame,
        known_users=known_users,
        interacting_users=interacting_users,
        has_any_warm_user=has_any_warm_user,
        unique_target_users=unique_target_users,
        all_item_ids=all_item_ids,
        has_boosts=has_boosts,
        recommend_k=recommend_k,
    )


def _recommend_per_strategy(
    models: dict[str, RecommenderModel],
    built: BuiltDataset,
    recommend_models: list[str],
    cohort_plan: _CohortPlan,
    *,
    blending_enabled: bool,
    blending_latest_date_columns: tuple[str, ...],
    top_k: int,
) -> _StrategyFrames:
    frames: list[pd.DataFrame] = []
    personalized_frames: list[pd.DataFrame] = []
    popular_frames: list[pd.DataFrame] = []
    latest_by_cohort: dict[object, list[tuple[str, int, float]]] = {}

    date_column = (
        resolve_latest_date_column(built.items, blending_latest_date_columns) if blending_enabled else None
    )
    latest_available = bool(blending_enabled and date_column is not None and built.items is not None)
    if blending_enabled and not latest_available:
        logger.info(
            "Blending: no usable date column among %s on items — disabling 'latest' "
            "and redistributing its weight onto popular",
            list(blending_latest_date_columns),
        )

    dataset = built.dataset
    for name in recommend_models:
        strategy = STRATEGIES[name]
        if strategy.personalized and not cohort_plan.has_any_warm_user:
            continue
        if name not in models:
            raise ValueError(f"Fitted model for strategy {name!r} is missing; available: {sorted(models)}")

        for cohort_key_value, cohort_users in cohort_plan.cohorts:
            allowed_items = cohort_plan.allowed_by_cohort[cohort_key_value]
            if not allowed_items:
                continue

            if strategy.personalized:
                cohort_warm = [u for u in cohort_users if u in cohort_plan.known_users]
                if strategy.requires_interactions:
                    cohort_warm = [u for u in cohort_warm if u in cohort_plan.interacting_users]
                if not cohort_warm:
                    continue
                recommend_users = cohort_warm
            else:
                recommend_users = cohort_users

            recs = models[name].recommend(
                users=recommend_users,
                dataset=dataset,
                k=cohort_plan.recommend_k,
                filter_viewed=strategy.personalized,
                items_to_recommend=allowed_items,
            )
            recs[SOURCE_COLUMN] = strategy.source_label
            recs[WEIGHT_COLUMN] = 1.0
            frames.append(recs)
            if blending_enabled:
                if strategy.personalized:
                    personalized_frames.append(recs)
                elif name == "popular":
                    popular_frames.append(recs)

    if blending_enabled and latest_available and date_column is not None and built.items is not None:
        latest_k = cohort_plan.recommend_k if cohort_plan.has_boosts else top_k
        for cohort_key, _cohort_users in cohort_plan.cohorts:
            allowed_items = cohort_plan.allowed_by_cohort[cohort_key]
            if not allowed_items:
                continue
            latest_by_cohort[cohort_key] = rank_latest_items(
                built.items, date_column, allowed_items, latest_k
            )

    return _StrategyFrames(
        frames=frames,
        personalized_frames=personalized_frames,
        popular_frames=popular_frames,
        latest_by_cohort=latest_by_cohort,
        date_column=date_column,
        latest_available=latest_available,
    )


def _combine_strategy_frames(
    models: dict[str, RecommenderModel],
    built: BuiltDataset,
    config: FeatureConfig,
    recommend_models: list[str],
    cohort_plan: _CohortPlan,
    strategy_frames: _StrategyFrames,
    *,
    blending_enabled: bool,
    weights: dict[str, float] | None,
    rrf_k: float | None,
    top_k: int,
) -> pd.DataFrame:
    combine_k = cohort_plan.recommend_k if cohort_plan.has_boosts else top_k
    empty_recs = pd.DataFrame(
        columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN]
    )

    if blending_enabled:
        blending = config.blending
        personalized = (
            pd.concat(strategy_frames.personalized_frames, ignore_index=True)
            if strategy_frames.personalized_frames
            else empty_recs
        )
        if not personalized.empty:
            personalized = personalized.copy()
            personalized[SOURCE_COLUMN] = PERSONALIZED_SOURCE

        popular = (
            pd.concat(strategy_frames.popular_frames, ignore_index=True)
            if strategy_frames.popular_frames
            else empty_recs
        )
        if not popular.empty:
            popular = popular.copy()
            popular[SOURCE_COLUMN] = POPULAR_SOURCE

        latest_parts: list[pd.DataFrame] = []
        for cohort_key, cohort_users in cohort_plan.cohorts:
            ranked = strategy_frames.latest_by_cohort.get(cohort_key)
            if ranked:
                latest_parts.append(expand_latest_ranking(ranked, cohort_users))
        latest_frame = pd.concat(latest_parts, ignore_index=True) if latest_parts else None

        counts = interaction_counts(built.interactions)
        combined = blend_for_users(
            personalized=personalized,
            popular=popular,
            latest=latest_frame,
            counts=counts,
            target_users=cohort_plan.unique_target_users,
            config=blending,
            top_k=combine_k,
            latest_available=strategy_frames.latest_available,
        )

        cold_popular = empty_recs.copy()
        cold_shared_latest: list[tuple[str, int, float]] | None = None
        if "popular" in models:
            global_rules = [rule for rule in cohort_plan.eligibility if not is_user_scoped(rule)]
            global_allowed = allowed_items_for_cohort(
                [],
                None,
                built.items,
                global_rules,
                cohort_plan.all_item_ids,
            )
            if global_allowed:
                cold_popular = models["popular"].recommend(
                    users=[COLD_START_USER_ID],
                    dataset=built.dataset,
                    k=combine_k,
                    filter_viewed=False,
                    items_to_recommend=global_allowed,
                )
                cold_popular[SOURCE_COLUMN] = POPULAR_SOURCE
                if (
                    strategy_frames.latest_available
                    and strategy_frames.date_column is not None
                    and built.items is not None
                ):
                    cold_shared_latest = rank_latest_items(
                        built.items,
                        strategy_frames.date_column,
                        global_allowed,
                        combine_k,
                    )

        combined = append_cold_start_rows(
            combined,
            popular=cold_popular,
            latest=None,
            config=blending,
            top_k=combine_k,
            latest_available=strategy_frames.latest_available,
            shared_latest=cold_shared_latest,
        )
        if WEIGHT_COLUMN in combined.columns:
            combined = combined.drop(columns=[WEIGHT_COLUMN])
        return combined

    if not strategy_frames.frames:
        return empty_recs
    if weights is not None:
        label_weights = {STRATEGIES[name].source_label: weights.get(name, 1.0) for name in recommend_models}
        stamped: list[pd.DataFrame] = []
        for frame in strategy_frames.frames:
            part = frame.copy()
            part[WEIGHT_COLUMN] = part[SOURCE_COLUMN].map(label_weights).fillna(1.0)
            stamped.append(part)
        source_label_order = [STRATEGIES[name].source_label for name in recommend_models]
        return _combine_by_weighted_fusion(
            stamped, combine_k, rrf_k if rrf_k is not None else RRF_K, source_label_order
        )
    return _combine_by_priority(strategy_frames.frames, combine_k)


def recommend_with_models(
    models: dict[str, RecommenderModel],
    built: BuiltDataset,
    target_users: list[str],
    config: FeatureConfig,
    top_k: int,
    enabled_models: list[str],
    weights: dict[str, float] | None = None,
    rrf_k: float | None = None,
    run_plan: ModelRunPlan | None = None,
) -> pd.DataFrame:
    """Runs recommend + combine on already-fitted strategies (no fit).

    Phases: resolve cohorts → recommend per strategy → combine → boosts.
    Used by ``train_and_recommend`` and by ``artifact.recommend_from_artifact``.
    """
    blending = config.blending
    blending_enabled = blending.enabled
    if run_plan is None:
        run_plan = plan_model_run(
            enabled_models,
            blending_enabled=blending_enabled,
            content_fallback_enabled=None,
        )
    enabled_models = list(run_plan.enabled_models)
    recommend_models = list(run_plan.recommend_models)
    if blending_enabled and "latest" in enabled_models:
        logger.warning(
            "Blending is enabled: date-based 'latest' comes from items; "
            "strategy 'latest' (trending PopularModel) is skipped for this run"
        )
    if blending_enabled and weights is not None:
        logger.warning(
            "Blending is enabled: job.model_weights / AutoML weights are ignored "
            "(per-user blend curve controls source mix instead)"
        )

    if weights is not None and not blending_enabled:
        unknown_weights = [name for name in weights if name not in recommend_models]
        if unknown_weights:
            raise ValueError(
                f"model_weights key(s) {unknown_weights} are not in recommend models {recommend_models}"
            )
        validate_model_weights(weights)
    validate_rrf_k(rrf_k)

    cohort_plan = _resolve_cohort_plan(built, config, target_users, recommend_models, top_k)
    strategy_frames = _recommend_per_strategy(
        models,
        built,
        recommend_models,
        cohort_plan,
        blending_enabled=blending_enabled,
        blending_latest_date_columns=tuple(blending.latest_date_columns),
        top_k=top_k,
    )
    combined = _combine_strategy_frames(
        models,
        built,
        config,
        recommend_models,
        cohort_plan,
        strategy_frames,
        blending_enabled=blending_enabled,
        weights=weights,
        rrf_k=rrf_k,
        top_k=top_k,
    )

    if cohort_plan.has_boosts:
        combined = apply_boosts(combined, built.items, config.boosts, top_k=top_k)

    return combined.reset_index(drop=True)


def train_and_recommend(
    built: BuiltDataset,
    target_users: list[str],
    config: FeatureConfig,
    top_k: int,
    enabled_models: list[str] | None = None,
    weights: dict[str, float] | None = None,
    rrf_k: float | None = None,
    strategy_cache: dict[str, RecommenderModel] | None = None,
    max_workers: int = 1,
    epoch_metrics: EpochMetricsSettings | None = None,
    item_based_k_neighbors: int | None = None,
    model_configs: dict[str, dict[str, Any]] | None = None,
    content_fallback_enabled: bool | None = None,
    content_fallback_max_neighbors: int = DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
    run_plan: ModelRunPlan | None = None,
) -> pd.DataFrame:
    """Fit enabled strategies, then recommend + combine. See ``fit_strategies``
    and ``recommend_with_models`` for the split used by model artifacts.

    Pass ``run_plan`` from ``plan_model_run`` when the caller already resolved
    models (e.g. job). Otherwise ``content_fallback_enabled`` is the config
    flag, or ``None`` to derive from ``enabled_models`` (AutoML lists).
    """
    if run_plan is None:
        run_plan = plan_model_run(
            enabled_models,
            blending_enabled=config.blending.enabled,
            content_fallback_enabled=content_fallback_enabled,
        )
    _, fitted = fit_strategies(
        built,
        target_users,
        enabled_models=list(run_plan.recommend_models),
        strategy_cache=strategy_cache,
        max_workers=max_workers,
        epoch_metrics=epoch_metrics,
        epoch_metrics_top_k=top_k,
        item_based_k_neighbors=item_based_k_neighbors,
        model_configs=model_configs,
        content_fallback_max_neighbors=content_fallback_max_neighbors,
        content_feature_columns=config.item_features,
    )
    return recommend_with_models(
        fitted,
        built,
        target_users,
        config,
        top_k=top_k,
        enabled_models=list(run_plan.enabled_models),
        weights=weights,
        rrf_k=rrf_k,
        run_plan=run_plan,
    )
