from __future__ import annotations

import pandas as pd

from cicerone.blending import COLD_START_USER_ID
from cicerone.config import Settings
from cicerone.feature_config import FeatureConfig
from support.settings import make_settings


def serve_settings(**overrides) -> Settings:
    return make_settings(**{"mode": "serve", "serve_auth_token": "secret", **overrides})


def recs_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"},
            {"user_id": "u1", "item_id": "i2", "rank": 2, "score": 0.5, "source": "personalized"},
            {"user_id": "u1", "item_id": "i3", "rank": 3, "score": 0.1, "source": "personalized"},
            {
                "user_id": COLD_START_USER_ID,
                "item_id": "i2",
                "rank": 1,
                "score": 0.4,
                "source": "popular_fallback",
            },
            {
                "user_id": COLD_START_USER_ID,
                "item_id": "i1",
                "rank": 2,
                "score": 0.3,
                "source": "popular_fallback",
            },
        ]
    )


def items_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "wine", "published": True, "in_stock": True},
            {"item_id": "i3", "category": "beer", "published": False, "in_stock": True},
        ]
    )


def serve_feature_config() -> FeatureConfig:
    return FeatureConfig(
        event_weights={},
        quantity_scaled_events=set(),
        event_caps={},
        user_features=[],
        item_features=[],
        item_availability_filters=["published", "in_stock"],
    )
