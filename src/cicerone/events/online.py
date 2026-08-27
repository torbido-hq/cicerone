"""Online collaborative refresh: LightFM fit_partial + user-scoped recommend."""

from __future__ import annotations

import copy
import hashlib
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset
from rectools.dataset.interactions import Interactions

from cicerone.artifact import ModelArtifact, dumps_artifact, loads_artifact
from cicerone.config.constants import DEFAULT_EVENTS_ONLINE_MAX_EXTRA_INTERACTIONS
from cicerone.config.settings import ExplainSettings
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


@dataclass
class _PendingRefresh:
    artifact: ModelArtifact
    working: Dataset
    pending_fit_events: int
    extra_raw: pd.DataFrame
    baseline_token: str | None
    baseline_digest: bytes | None


def _digest(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def _external_ids(values: Iterable[object]) -> set[str]:
    return {str(value) for value in values}


def _empty_interaction_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_INTERACTION_COLS))


def _interaction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[:, list(_INTERACTION_COLS)].copy()
    out[Columns.User] = out[Columns.User].astype(str)
    out[Columns.Item] = out[Columns.Item].astype(str)
    out[Columns.Weight] = pd.to_numeric(out[Columns.Weight], errors="coerce")
    out[Columns.Datetime] = pd.to_datetime(out[Columns.Datetime], utc=True).dt.tz_convert(None)
    return out


def _trim_recent_interactions(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows < 1 or frame.empty or len(frame) <= max_rows:
        return frame
    ordered = frame
    if Columns.Datetime in frame.columns:
        ordered = frame.sort_values(Columns.Datetime, kind="mergesort")
    return ordered.tail(max_rows).reset_index(drop=True)


def _merge_interaction_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = [_interaction_frame(frame) for frame in frames if frame is not None and not frame.empty]
    if not parts:
        return _empty_interaction_frame()
    combined = pd.concat(parts, ignore_index=True)
    return combined.groupby([Columns.User, Columns.Item], as_index=False).agg(
        {Columns.Weight: "sum", Columns.Datetime: "max"}
    )


def _dataset_with_interactions(dataset: Dataset, combined: pd.DataFrame) -> Dataset:
    if combined.empty:
        return dataset
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
        max_extra_interactions: int = DEFAULT_EVENTS_ONLINE_MAX_EXTRA_INTERACTIONS,
        fence_check: Callable[[], bool] | None = None,
        explain: ExplainSettings | None = None,
    ):
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if fit_partial_epochs < 0:
            raise ValueError("fit_partial_epochs must be >= 0")
        if fit_min_events < 1:
            raise ValueError("fit_min_events must be >= 1")
        if max_extra_interactions < 1:
            raise ValueError("max_extra_interactions must be >= 1")
        self._sink = sink
        self._top_k = top_k
        self._half_life_days = half_life_days
        self._fit_partial_epochs = fit_partial_epochs
        self._fit_min_events = fit_min_events
        self._max_extra_interactions = max_extra_interactions
        self._fence_check = fence_check
        self._explain = explain if explain is not None else ExplainSettings()
        self._artifact: ModelArtifact | None = None
        self._working: Dataset | None = None
        self._job_raw = _empty_interaction_frame()
        self._extra_raw = _empty_interaction_frame()
        self._payload_digest: bytes | None = None
        self._artifact_token: str | None = None
        self._hot_users: frozenset[str] = frozenset()
        self._hot_items: frozenset[str] = frozenset()
        self._pending_fit_events = 0
        self._pending: _PendingRefresh | None = None

    def invalidate(self) -> None:
        self._pending = None
        self._artifact = None
        self._working = None
        self._job_raw = _empty_interaction_frame()
        self._extra_raw = _empty_interaction_frame()
        self._payload_digest = None
        self._artifact_token = None
        self._hot_users = frozenset()
        self._hot_items = frozenset()
        self._pending_fit_events = 0

    def abort(self) -> None:
        """Drop an uncommitted refresh; committed artifact on disk is unchanged."""
        self._pending = None

    def commit(self) -> None:
        """Persist a successful refresh after recommendation rows are durable / acked."""
        pending = self._pending
        if pending is None:
            return
        self._ensure_fence()
        if self._artifact_replaced(pending):
            logger.warning("Model artifact changed during online refresh; dropping pending fit")
            self._pending = None
            return
        self._persist(pending.artifact)
        digest = self._payload_digest
        if digest is None:
            digest = _digest(dumps_artifact(pending.artifact))
        self._artifact = pending.artifact
        self._working = pending.working
        self._extra_raw = pending.extra_raw
        self._payload_digest = digest
        self._artifact_token = self._fingerprint()
        self._pending_fit_events = pending.pending_fit_events
        self._pending = None

    def ensure_loaded(self) -> None:
        if not self._reload():
            raise OnlineArtifactError(ONLINE_ARTIFACT_REQUIRED)

    def refresh(self, events: Sequence[NormalizedEvent]) -> OnlineRefreshResult:
        if not events:
            return OnlineRefreshResult(rows=empty_online_rows())
        if not self._reload():
            logger.warning("%s; skipping online refresh", ONLINE_ARTIFACT_REQUIRED)
            return OnlineRefreshResult(rows=empty_online_rows())
        artifact = self._pending.artifact if self._pending is not None else self._artifact
        working = self._pending.working if self._pending is not None else self._working
        pending_fit = (
            self._pending.pending_fit_events if self._pending is not None else self._pending_fit_events
        )
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
        extra_raw = _merge_interaction_frames(self._pending_extra(), extra)
        extra_raw = _trim_recent_interactions(extra_raw, self._max_extra_interactions)
        if extra_raw.empty:
            return OnlineRefreshResult(
                rows=empty_online_rows(),
                events_dropped_unknown=dropped,
                events_known=int(len(known)),
            )
        maps = artifact.dataset
        working = _dataset_with_interactions(maps, _merge_interaction_frames(self._job_raw, extra_raw))
        pending_fit += int(len(known))

        models, fitted = self._recommend_models(artifact)
        epochs_run = 0
        collaborative = fitted.get("collaborative")
        if (
            collaborative is not None
            and self._fit_partial_epochs > 0
            and pending_fit >= self._fit_min_events
            and callable(getattr(collaborative, "fit_partial", None))
        ):
            self._ensure_fence()
            collaborative = copy.deepcopy(collaborative)
            collaborative.fit_partial(working, self._fit_partial_epochs)
            epochs_run = self._fit_partial_epochs
            pending_fit = 0
            fitted = {**fitted, "collaborative": collaborative}
            artifact = replace(artifact, fitted={**artifact.fitted, "collaborative": collaborative})

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
            explain=self._explain,
        )
        artifact = replace(artifact, dataset=working, fitted={**artifact.fitted, **fitted})
        self._pending = _PendingRefresh(
            artifact=artifact,
            working=working,
            pending_fit_events=pending_fit,
            extra_raw=extra_raw,
            baseline_token=self._artifact_token,
            baseline_digest=self._payload_digest,
        )
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
            "Online refresh prepared recommendations for %d user(s) "
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
        if self._pending is not None:
            return True
        token = self._fingerprint()
        if (
            token is not None
            and token == self._artifact_token
            and self._artifact is not None
            and self._working is not None
        ):
            return True
        payload = self._sink.read_model_artifact()
        if payload is None:
            return False
        digest = _digest(payload)
        if digest == self._payload_digest and self._artifact is not None and self._working is not None:
            self._artifact_token = token
            return True
        artifact = loads_artifact(payload)
        self._install(artifact, digest, token)
        return True

    def _fingerprint(self) -> str | None:
        getter = getattr(self._sink, "model_artifact_fingerprint", None)
        if not callable(getter):
            return None
        token = getter()
        return None if token is None else str(token)

    def _install(self, artifact: ModelArtifact, digest: bytes, token: str | None = None) -> None:
        raw = artifact.dataset.get_raw_interactions()
        # LightFM identity features follow interacting IDs, not the full id map.
        if raw is None or raw.empty:
            hot_users: set[str] = set()
            hot_items: set[str] = set()
        else:
            hot_users = _external_ids(raw[Columns.User])
            hot_items = _external_ids(raw[Columns.Item])
        self._artifact = artifact
        self._working = artifact.dataset
        self._payload_digest = digest
        self._artifact_token = token
        self._hot_users = frozenset(hot_users)
        self._hot_items = frozenset(hot_items)
        self._pending_fit_events = 0
        if raw is None or raw.empty:
            self._job_raw = _empty_interaction_frame()
        else:
            self._job_raw = _interaction_frame(raw)
        self._extra_raw = _empty_interaction_frame()

    def _pending_extra(self) -> pd.DataFrame:
        if self._pending is not None:
            return self._pending.extra_raw
        return self._extra_raw

    def _artifact_replaced(self, pending: _PendingRefresh) -> bool:
        token = self._fingerprint()
        if pending.baseline_token is not None and token is not None:
            return token != pending.baseline_token
        payload = self._sink.read_model_artifact()
        if payload is None:
            return True
        if pending.baseline_digest is None:
            return False
        return _digest(payload) != pending.baseline_digest

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
        payload = dumps_artifact(artifact)
        self._sink.write_model_artifact(payload)
        self._payload_digest = _digest(payload)
        self._artifact_token = self._fingerprint()

    def _ensure_fence(self) -> None:
        if self._fence_check is not None and not self._fence_check():
            raise LockLostError("events apply lock lost before online write")
