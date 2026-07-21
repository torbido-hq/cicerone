from __future__ import annotations

import pandas as pd
import pytest

from cicerone.feature_config import FeatureColumn, FeatureConfig


@pytest.fixture
def feature_config() -> FeatureConfig:
    return FeatureConfig(
        event_weights={
            "purchase": 4.0,
            "review_positive": 5.0,
            "review_negative": -3.0,
            "saved": 2.0,
            "cart_add": 1.0,
            "view": 0.3,
        },
        quantity_scaled_events={"purchase"},
        event_caps={"view": 5},
        user_features=[
            FeatureColumn(column="favorite_styles", type="list"),
            FeatureColumn(column="region_slug", type="categorical"),
        ],
        item_features=[
            FeatureColumn(column="category", type="categorical"),
            FeatureColumn(column="producer_id", type="categorical"),
        ],
        item_availability_filters=["published", "in_stock"],
    )


@pytest.fixture
def sample_events() -> pd.DataFrame:
    now = pd.Timestamp.utcnow()
    return pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "event_type": "purchase", "quantity": 3, "occurred_at": now},
            {"user_id": "u1", "item_id": "i2", "event_type": "view", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i1", "event_type": "review_positive", "quantity": 1, "occurred_at": now},
            {"user_id": "u2", "item_id": "i3", "event_type": "saved", "quantity": 1, "occurred_at": now},
            {"user_id": "u3", "item_id": "i2", "event_type": "cart_add", "quantity": 1, "occurred_at": now},
        ]
    )


@pytest.fixture
def sample_users() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "favorite_styles": ["ipa", "stout"], "region_slug": "lazio"},
            {"user_id": "u2", "favorite_styles": ["lager"], "region_slug": "toscana"},
            {"user_id": "u3", "favorite_styles": [], "region_slug": None},
            {"user_id": "u4", "favorite_styles": ["ipa"], "region_slug": "lazio"},
        ]
    )


@pytest.fixture
def sample_items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "producer_id": "p1", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "beer", "producer_id": "p2", "published": True, "in_stock": True},
            {"item_id": "i3", "category": "wine", "producer_id": "p1", "published": True, "in_stock": False},
            {"item_id": "i4", "category": "wine", "producer_id": "p3", "published": False, "in_stock": True},
        ]
    )
