from __future__ import annotations

import pandas as pd
import pytest

from cicerone.config import IOSettings, Settings
from cicerone.feature_config import FeatureColumn, FeatureConfig


def make_settings(**overrides) -> Settings:
    """Shared factory for tests that need a fully-populated `Settings`
    directly (rather than via `load_settings(path)` from a TOML file, see
    test_config.py) -- centralized so the full field list/defaults used by
    test_dashboard.py, test_serve.py and test_trigger.py can't drift apart
    as config.py gains new fields. Callers pass only the overrides relevant
    to what they're testing (e.g. `mode`, `*_enabled`, `*_auth_token`).
    """
    base = dict(
        input=IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/in"}),
        output=IOSettings(kind="dataset", options={"storage_backend": "local", "path": "/tmp/out"}),
        feature_config_path="/app/config/features.toml",
        top_k=10,
        half_life_days=90,
        cron_schedule="0 3 * * *",
        models=None,
        model_weights=None,
        rrf_k=None,
        save_model_artifact=False,
        automl_enabled=False,
        automl_n_splits=2,
        automl_test_days=14,
        automl_primary_metric="MAP",
        automl_candidates=None,
        mode="batch",
        serve_host="0.0.0.0",
        serve_port=8000,
        serve_auth_token=None,
        serve_default_k=10,
        serve_refresh_interval_seconds=60,
        trigger_enabled=False,
        trigger_host="0.0.0.0",
        trigger_port=8080,
        trigger_auth_token=None,
        trigger_debounce_seconds=60,
        trigger_poll_input_bucket=False,
        trigger_poll_interval_seconds=300,
        dashboard_enabled=False,
        dashboard_host="0.0.0.0",
        dashboard_port=8090,
        dashboard_users_path="/tmp/dashboard_users.toml",
        dashboard_refresh_interval_seconds=30,
        dashboard_history_limit=20,
    )
    base.update(overrides)
    return Settings(**base)


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
            {
                "user_id": "u2",
                "item_id": "i1",
                "event_type": "review_positive",
                "quantity": 1,
                "occurred_at": now,
            },
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
