"""Trains one or more recommendation strategies (see STRATEGIES) and combines
their outputs into top-K recommendations per user, with a non-personalized
fallback for cold-start users who have too little (or no) personal signal.
"""

from __future__ import annotations

import inspect
import logging
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from typing import Protocol

import pandas as pd
from implicit.nearest_neighbours import TFIDFRecommender
from lightfm import LightFM
from rectools import Columns
from rectools.dataset import Dataset
from rectools.metrics import Precision, Recall, calc_metrics
from rectools.models import ImplicitItemKNNWrapperModel, LightFMWrapperModel, PopularModel

from cicerone.blending import (
    COLD_START_USER_ID,
    PERSONALIZED_SOURCE,
    POPULAR_SOURCE,
    append_cold_start_rows,
    blend_for_users,
    build_latest_ranking,
    interaction_counts,
    resolve_latest_date_column,
)
from cicerone.config import (
    DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
    DEFAULT_ITEM_BASED_K_NEIGHBORS,
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
_EPOCH_METRICS_RNG = random.Random()


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
        users: list,
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
    factory: Callable[[], RecommenderModel]
    personalized: bool
    source_label: str
    # Item-KNN / content fallback need interaction history; LightFM hybrid can
    # still score feature-only (dataset-known) users.
    requires_interactions: bool = False


def _build_collaborative() -> RecommenderModel:
    return _as_recommender_model(
        LightFMWrapperModel(
            LightFM(
                no_components=64,
                loss="warp",
                learning_rate=0.05,
                item_alpha=1e-6,
                user_alpha=1e-6,
                random_state=RANDOM_STATE,
            ),
            epochs=COLLABORATIVE_EPOCHS,
            num_threads=4,
        )
    )


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
    # rectools LightFMWrapperModel stores constructor epochs= as n_epochs;
    # accept epochs as a fallback for other wrappers.
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


def _build_item_based(k_neighbors: int = DEFAULT_ITEM_BASED_K_NEIGHBORS) -> RecommenderModel:
    return _as_recommender_model(ImplicitItemKNNWrapperModel(TFIDFRecommender(K=k_neighbors)))


def _build_popular() -> RecommenderModel:
    return _as_recommender_model(PopularModel())


def _build_latest() -> RecommenderModel:
    return _as_recommender_model(
        PopularModel(popularity="n_interactions", period=pd.Timedelta(days=LATEST_WINDOW_DAYS))
    )


def _build_content_fallback() -> RecommenderModel:
    """Placeholder factory; real instances are built in ``_fit_strategy`` with items/features."""
    return _as_recommender_model(ContentFallbackModel())


STRATEGIES: dict[str, Strategy] = {
    "collaborative": Strategy(_build_collaborative, personalized=True, source_label="personalized"),
    "item_based": Strategy(
        _build_item_based,
        personalized=True,
        source_label="item_based",
        requires_interactions=True,
    ),
    "content_fallback": Strategy(
        _build_content_fallback,
        personalized=True,
        source_label=CONTENT_FALLBACK_SOURCE,
        requires_interactions=True,
    ),
    "popular": Strategy(_build_popular, personalized=False, source_label="popular_fallback"),
    "latest": Strategy(_build_latest, personalized=False, source_label="latest"),
}


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


def _fit_strategy(
    name: str,
    dataset: Dataset,
    epoch_interactions: pd.DataFrame | None,
    epoch_metrics: EpochMetricsSettings | None,
    epoch_metrics_top_k: int,
    item_based_k_neighbors: int = DEFAULT_ITEM_BASED_K_NEIGHBORS,
    content_feature_columns: list[FeatureColumn] | None = None,
    content_max_neighbors: int = DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
    content_items: pd.DataFrame | None = None,
    content_interactions: pd.DataFrame | None = None,
) -> tuple[str, RecommenderModel]:
    """Fit one strategy (picklable for ProcessPoolExecutor workers)."""
    if name == "item_based":
        model = _build_item_based(item_based_k_neighbors)
    elif name == "content_fallback":
        model = _as_recommender_model(
            build_content_fallback_model(
                feature_columns=content_feature_columns or [],
                max_neighbors=content_max_neighbors,
                items=content_items,
                interactions=content_interactions,
            )
        )
    else:
        model = STRATEGIES[name].factory()
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


def resolve_run_models(
    enabled_models: list[str] | None,
    *,
    blending_enabled: bool,
    content_fallback_enabled: bool = False,
) -> tuple[list[str], list[str]]:
    """Resolve requested models and the effective fit/recommend list.

    Single entry point so content_fallback / blending adjustments cannot
    drift across job, train_and_recommend, and recommend_with_models.
    """
    resolved = _resolve_enabled_models(enabled_models)
    recommend = resolve_recommend_models(
        resolved, blending_enabled, content_fallback_enabled=content_fallback_enabled
    )
    return resolved, recommend


def _interacting_external_user_ids(built: BuiltDataset) -> set:
    """External user IDs with ≥1 interaction (same namespace as ``target_users``).

    ``BuiltDataset.interactions`` keeps original event user/item ids under
    rectools ``Columns.User`` / ``Columns.Item`` — these are *external* ids,
    not rectools' dense internal indices.
    """
    if built.interactions is None or built.interactions.empty:
        return set()
    return set(built.interactions[Columns.User].tolist())


def fit_strategies(
    built: BuiltDataset,
    target_users: list[str],
    enabled_models: list[str] | None = None,
    strategy_cache: dict[str, RecommenderModel] | None = None,
    max_workers: int = 1,
    epoch_metrics: EpochMetricsSettings | None = None,
    epoch_metrics_top_k: int = 10,
    item_based_k_neighbors: int = DEFAULT_ITEM_BASED_K_NEIGHBORS,
    content_fallback_max_neighbors: int = DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
    content_feature_columns: list[FeatureColumn] | None = None,
) -> tuple[list[str], dict[str, RecommenderModel]]:
    """Fit (or cache-hit) enabled strategies. ``max_workers > 1`` fits in parallel.

    ``epoch_metrics`` enables collaborative fit_partial logging; default
    ``None`` keeps a single LightFM ``fit()``.
    """
    dataset = built.dataset
    enabled_models = _resolve_enabled_models(enabled_models)

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
    # Pre-slice in the parent so ProcessPool workers do not pickle the full
    # interactions frame (None when epoch logging is off or collaborative isn't fitting).
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
            with ProcessPoolExecutor(max_workers=min(max_workers, len(to_fit))) as executor:
                for name, model in executor.map(
                    _fit_strategy,
                    to_fit,
                    repeat(dataset),
                    repeat(epoch_interactions),
                    repeat(epoch_metrics),
                    repeat(epoch_metrics_top_k),
                    repeat(item_based_k_neighbors),
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
                    item_based_k_neighbors,
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


def recommend_with_models(
    models: dict[str, RecommenderModel],
    built: BuiltDataset,
    target_users: list[str],
    config: FeatureConfig,
    top_k: int,
    enabled_models: list[str],
    weights: dict[str, float] | None = None,
    rrf_k: float | None = None,
    content_fallback_enabled: bool = False,
) -> pd.DataFrame:
    """Runs recommend + combine on already-fitted strategies (no fit).

    Used by ``train_and_recommend`` and by ``artifact.recommend_from_artifact``
    so a loaded model artifact can produce recommendations without re-training.
    """
    blending = config.blending
    blending_enabled = blending.enabled
    enabled_models, recommend_models = resolve_run_models(
        enabled_models,
        blending_enabled=blending_enabled,
        content_fallback_enabled=content_fallback_enabled,
    )
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
    # External ids — same namespace as target_users / cohort_users.
    interacting_users = _interacting_external_user_ids(built)

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

    frames = []
    personalized_frames: list[pd.DataFrame] = []
    popular_frames: list[pd.DataFrame] = []
    latest_frames: list[pd.DataFrame] = []

    date_column = (
        resolve_latest_date_column(built.items, blending.latest_date_columns) if blending_enabled else None
    )
    latest_available = bool(blending_enabled and date_column is not None and built.items is not None)
    if blending_enabled and not latest_available:
        logger.info(
            "Blending: no usable date column among %s on items — disabling 'latest' "
            "and redistributing its weight onto popular",
            list(blending.latest_date_columns),
        )

    for name in recommend_models:
        strategy = STRATEGIES[name]
        if strategy.personalized and not has_any_warm_user:
            continue
        if name not in models:
            raise ValueError(f"Fitted model for strategy {name!r} is missing; available: {sorted(models)}")

        for cohort_key_value, cohort_users in cohorts:
            allowed_items = allowed_by_cohort[cohort_key_value]
            if not allowed_items:
                continue

            if strategy.personalized:
                cohort_warm = [u for u in cohort_users if u in known_users]
                if strategy.requires_interactions:
                    cohort_warm = [u for u in cohort_warm if u in interacting_users]
                if not cohort_warm:
                    continue
                recommend_users = cohort_warm
            else:
                recommend_users = cohort_users

            # rectools ModelBase.recommend() owns external↔internal ID mapping
            # via dataset.*_id_map; _combine_by_* only merges already-external frames.
            recs = models[name].recommend(
                users=recommend_users,
                dataset=dataset,
                k=recommend_k,
                filter_viewed=strategy.personalized,
                items_to_recommend=allowed_items,
            )
            recs[SOURCE_COLUMN] = strategy.source_label
            recs[WEIGHT_COLUMN] = weights.get(name, 1.0) if weights is not None else 1.0
            frames.append(recs)
            if blending_enabled:
                if strategy.personalized:
                    personalized_frames.append(recs)
                elif name == "popular":
                    popular_frames.append(recs)

    if blending_enabled and latest_available and date_column is not None and built.items is not None:
        for _cohort_key, cohort_users in cohorts:
            allowed_items = allowed_by_cohort[_cohort_key]
            if not allowed_items:
                continue
            latest_frames.append(
                build_latest_ranking(
                    built.items,
                    date_column,
                    allowed_items,
                    recommend_k if has_boosts else top_k,
                    cohort_users,
                )
            )

    combine_k = recommend_k if has_boosts else top_k

    if blending_enabled:
        empty_recs = pd.DataFrame(
            columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN]
        )
        personalized = (
            pd.concat(personalized_frames, ignore_index=True) if personalized_frames else empty_recs
        )
        if not personalized.empty:
            personalized = personalized.copy()
            personalized[SOURCE_COLUMN] = PERSONALIZED_SOURCE

        popular = pd.concat(popular_frames, ignore_index=True) if popular_frames else empty_recs
        if not popular.empty:
            popular = popular.copy()
            popular[SOURCE_COLUMN] = POPULAR_SOURCE

        latest_frame = pd.concat(latest_frames, ignore_index=True) if latest_frames else None

        counts = interaction_counts(built.interactions)
        combined = blend_for_users(
            personalized=personalized,
            popular=popular,
            latest=latest_frame,
            counts=counts,
            target_users=unique_target_users,
            config=blending,
            top_k=combine_k,
            latest_available=latest_available,
        )

        # __cold_start__: global (item-scoped) allowlist only — not a user cohort.
        cold_popular = empty_recs.copy()
        cold_latest = None
        if "popular" in models:
            global_rules = [rule for rule in eligibility if not is_user_scoped(rule)]
            global_allowed = allowed_items_for_cohort(
                [],
                None,
                built.items,
                global_rules,
                all_item_ids,
            )
            if global_allowed:
                cold_popular = models["popular"].recommend(
                    users=[COLD_START_USER_ID],
                    dataset=dataset,
                    k=combine_k,
                    filter_viewed=False,
                    items_to_recommend=global_allowed,
                )
                cold_popular[SOURCE_COLUMN] = POPULAR_SOURCE
                if latest_available and date_column is not None and built.items is not None:
                    cold_latest = build_latest_ranking(
                        built.items,
                        date_column,
                        global_allowed,
                        combine_k,
                        [COLD_START_USER_ID],
                    )

        combined = append_cold_start_rows(
            combined,
            popular=cold_popular,
            latest=cold_latest,
            config=blending,
            top_k=combine_k,
            latest_available=latest_available,
        )
        if WEIGHT_COLUMN in combined.columns:
            combined = combined.drop(columns=[WEIGHT_COLUMN])
    elif not frames:
        return pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank, Columns.Score, SOURCE_COLUMN])
    elif weights is not None:
        source_label_order = [STRATEGIES[name].source_label for name in recommend_models]
        combined = _combine_by_weighted_fusion(
            frames, combine_k, rrf_k if rrf_k is not None else RRF_K, source_label_order
        )
    else:
        combined = _combine_by_priority(frames, combine_k)

    if has_boosts:
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
    item_based_k_neighbors: int = DEFAULT_ITEM_BASED_K_NEIGHBORS,
    content_fallback_enabled: bool = False,
    content_fallback_max_neighbors: int = DEFAULT_CONTENT_FALLBACK_MAX_NEIGHBORS,
) -> pd.DataFrame:
    """Fit enabled strategies, then recommend + combine. See ``fit_strategies``
    and ``recommend_with_models`` for the split used by model artifacts.
    """
    resolved_models, fit_models = resolve_run_models(
        enabled_models,
        blending_enabled=config.blending.enabled,
        content_fallback_enabled=content_fallback_enabled,
    )
    _, fitted = fit_strategies(
        built,
        target_users,
        enabled_models=fit_models,
        strategy_cache=strategy_cache,
        max_workers=max_workers,
        epoch_metrics=epoch_metrics,
        epoch_metrics_top_k=top_k,
        item_based_k_neighbors=item_based_k_neighbors,
        content_fallback_max_neighbors=content_fallback_max_neighbors,
        content_feature_columns=config.item_features,
    )
    return recommend_with_models(
        fitted,
        built,
        target_users,
        config,
        top_k=top_k,
        enabled_models=resolved_models,
        weights=weights,
        rrf_k=rrf_k,
        content_fallback_enabled=content_fallback_enabled,
    )
