from __future__ import annotations

import pandas as pd

from cicerone.artifact import ModelArtifact
from cicerone.dataset import BuiltDataset
from cicerone.model import recommend_with_models


def recommend_from_loaded(artifact: ModelArtifact, users: list[str], top_k: int) -> pd.DataFrame:
    built = BuiltDataset(
        dataset=artifact.dataset,
        interactions=artifact.dataset.get_raw_interactions(),
        items=artifact.items,
        users=artifact.users,
    )
    return recommend_with_models(
        dict(artifact.fitted),
        built,
        users,
        artifact.feature_config,
        top_k=top_k,
        enabled_models=list(artifact.models),
        weights=artifact.model_weights,
        rrf_k=artifact.rrf_k,
    )
