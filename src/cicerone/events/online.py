"""Online collaborative refresh: LightFM fit_partial + user-scoped recommend."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset
from rectools.dataset.interactions import Interactions

from cicerone.artifact import ModelArtifact, dumps_artifact, loads_artifact
from cicerone.dataset import BuiltDataset, build_interactions
from cicerone.events.base import NormalizedEvent
from cicerone.events.normalize import events_to_dataframe
from cicerone.events.online_result import OnlineRefreshResult, empty_online_rows
from cicerone.io.base import OutputSink
from cicerone.io.recommendation_reader import ITEM_COLUMN, USER_COLUMN
from cicerone.io.recommendation_schema import recommendation_output_columns
from cicerone.locks import LockLostError
from cicerone.model import recommend_with_models
from cicerone.model_config import SEQUENTIAL_STRATEGY, sequential_extra_available

logger = logging.getLogger(__name__)

ONLINE_ARTIFACT_REQUIRED = (
    "events.online.enabled requires a model artifact in [output]; "
    "the batch job must set save_model_artifact = true"
)

_INTERACTION_COLS = (Columns.User, Columns.Item, Columns.Weight, Columns.Datetime)


class OnlineArtifactError(RuntimeError):
    """Online mode is enabled but the output store has no model artifact."""


def _digest(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def _external_ids(values: Iterable[object]) -> set[str]:
    return {str(value) for value in values}


def _interaction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[:, list(_INTERACTION_COLS)].copy()
    out[Columns.User] = out[Columns.User].astype(str)
    out[Columns.Item] = out[Columns.Item].astype(str)
    out[Columns.Weight] = pd.to_numeric(out[Columns.Weight], errors="coerce")
    out[Columns.Datetime] = pd.to_datetime(out[Columns.Datetime], utc=True).dt.tz_convert(None)
    return out


def _append_known_interactions(dataset: Dataset, extra: pd.DataFrame) -> Dataset:
    raw = dataset.get_raw_interactions()
    parts: list[pd.DataFrame] = []
    if raw is not None and not raw.empty:
        parts.append(_interaction_frame(raw))
    if not extra.empty:
        parts.append(_interaction_frame(extra))
    if not parts:
        return dataset
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.groupby([Columns.User, Columns.Item], as_index=False).agg(
        {Columns.Weight: "sum", Columns.Datetime: "max"}
    )
    interactions = Interactions.from_raw(combined, dataset.user_id_map, dataset.item_id_map)
    return Dataset(
        user_id_map=dataset.user_id_map,
        item_id_map=dataset.item_id_map,
        interactions=interactions,
        user_features=dataset.user_features,
        item_features=dataset.item_features,
    )


class OnlineTrainer:
    """Load the last artifact, continue LightFM on hot IDs, re-recommend affected users."""

    def __init__(
        self,
        *,
        sink: OutputSink,
        top_k: int,
        half_life_days: float,
        fit_partial_epochs: int,
        fit_min_events: int,
        fence_check: Callable[[], bool] | None = None,
    ):
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if fit_partial_epochs < 0:
            raise ValueError("fit_partial_epochs must be >= 0")
        if fit_min_events < 1:
            raise ValueError("fit_min_events must be >= 1")
        self._sink = sink
        self._top_k = top_k
        self._half_life_days = half_life_days
        self._fit_partial_epochs = fit_partial_epochs
        self._fit_min_events = fit_min_events
        self._fence_check = fence_check
        self._artifact: ModelArtifact | None = None
        self._working: Dataset | None = None
        self._payload_digest: bytes | None = None
        self._hot_users: frozenset[str] = frozenset()
        self._hot_items: frozenset[str] = frozenset()
        self._pending_fit_events = 0

    def invalidate(self) -> None:
        self._artifact = None
        self._working = None
        self._payload_digest = None
        self._hot_users = frozenset()
        self._hot_items = frozenset()
        self._pending_fit_events = 0

    def ensure_loaded(self) -> None:
        if not self._reload():
            raise OnlineArtifactError(ONLINE_ARTIFACT_REQUIRED)

    def refresh(self, events: Sequence[NormalizedEvent]) -> OnlineRefreshResult:
        if not events:
            return OnlineRefreshResult(rows=empty_online_rows())
        if not self._reload():
            logger.warning("%s; skipping online refresh", ONLINE_ARTIFACT_REQUIRED)
            return OnlineRefreshResult(rows=empty_online_rows())
        artifact = self._artifact
        working = self._working
        if artifact is None or working is None:
            return OnlineRefreshResult(rows=empty_online_rows())

        batch = events_to_dataframe(events)
        users = batch[USER_COLUMN].astype(str)
        items = batch[ITEM_COLUMN].astype(str)
        known_mask = users.isin(self._hot_users) & items.isin(self._hot_items)
        dropped = int((~known_mask).sum())
        known = batch.loc[known_mask]
        if known.empty:
            return OnlineRefreshResult(
                rows=empty_online_rows(),
                events_dropped_unknown=dropped,
            )

        extra = build_interactions(known, artifact.feature_config, self._half_life_days)
        if extra.empty:
            return OnlineRefreshResult(
                rows=empty_online_rows(),
                events_dropped_unknown=dropped,
                events_known=int(len(known)),
            )

        extra = extra.copy()
        extra[Columns.User] = extra[Columns.User].astype(str)
        extra[Columns.Item] = extra[Columns.Item].astype(str)
        working = _append_known_interactions(working, extra)
        self._working = working
        self._pending_fit_events += int(len(known))

        models, fitted = self._recommend_models(artifact)
        epochs_run = 0
        collaborative = fitted.get("collaborative")
        if (
            collaborative is not None
            and self._fit_partial_epochs > 0
            and self._pending_fit_events >= self._fit_min_events
            and callable(getattr(collaborative, "fit_partial", None))
        ):
            self._ensure_fence()
            collaborative.fit_partial(working, self._fit_partial_epochs)
            epochs_run = self._fit_partial_epochs
            self._pending_fit_events = 0
            fitted = {**fitted, "collaborative": collaborative}
            artifact = replace(artifact, fitted={**artifact.fitted, "collaborative": collaborative})
            self._artifact = artifact

        target_users = sorted({str(user_id) for user_id in extra[Columns.User].unique()})
        built = BuiltDataset(
            dataset=working,
            interactions=working.get_raw_interactions(),
            items=artifact.items,
            users=artifact.users,
        )
        rows = recommend_with_models(
            fitted,
            built,
            target_users,
            artifact.feature_config,
            top_k=self._top_k,
            enabled_models=models,
            weights=dict(artifact.model_weights) if artifact.model_weights is not None else None,
            rrf_k=artifact.rrf_k,
        )
        artifact = replace(artifact, dataset=working, fitted={**artifact.fitted, **fitted})
        self._artifact = artifact
        self._persist(artifact)
        if rows.empty:
            return OnlineRefreshResult(
                rows=empty_online_rows(),
                fit_partial_epochs=epochs_run,
                events_dropped_unknown=dropped,
                events_known=int(len(known)),
            )
        rows = rows.copy()
        rows[USER_COLUMN] = rows[USER_COLUMN].astype(str)
        rows = rows[recommendation_output_columns(rows)]
        refreshed = int(rows[USER_COLUMN].nunique())
        logger.info(
            "Online refresh wrote recommendations for %d user(s) "
            "(%d known event(s), %d dropped, fit_partial=%d)",
            refreshed,
            len(known),
            dropped,
            epochs_run,
        )
        return OnlineRefreshResult(
            rows=rows,
            users_refreshed=refreshed,
            fit_partial_epochs=epochs_run,
            events_dropped_unknown=dropped,
            events_known=int(len(known)),
        )

    def _reload(self) -> bool:
        payload = self._sink.read_model_artifact()
        if payload is None:
            return False
        digest = _digest(payload)
        if digest == self._payload_digest and self._artifact is not None and self._working is not None:
            return True
        artifact = loads_artifact(payload)
        self._install(artifact, digest)
        return True

    def _install(self, artifact: ModelArtifact, digest: bytes) -> None:
        raw = artifact.dataset.get_raw_interactions()
        if raw is None or raw.empty:
            hot_users: set[str] = set()
            hot_items: set[str] = set()
        else:
            hot_users = _external_ids(raw[Columns.User])
            hot_items = _external_ids(raw[Columns.Item])
        self._artifact = artifact
        self._working = artifact.dataset
        self._payload_digest = digest
        self._hot_users = frozenset(hot_users)
        self._hot_items = frozenset(hot_items)
        self._pending_fit_events = 0

    def _recommend_models(self, artifact: ModelArtifact) -> tuple[list[str], dict]:
        skip_sequential = SEQUENTIAL_STRATEGY in artifact.models and not sequential_extra_available()
        models = [
            name
            for name in artifact.models
            if name in artifact.fitted and not (skip_sequential and name == SEQUENTIAL_STRATEGY)
        ]
        fitted = {name: artifact.fitted[name] for name in models}
        if skip_sequential:
            logger.info("Online refresh skipping sequential (torch extra is not installed)")
        return models, fitted

    def _persist(self, artifact: ModelArtifact) -> None:
        self._ensure_fence()
        payload = dumps_artifact(artifact)
        self._sink.write_model_artifact(payload)
        self._payload_digest = _digest(payload)

    def _ensure_fence(self) -> None:
        if self._fence_check is not None and not self._fence_check():
            raise LockLostError("events apply lock lost before online write")
