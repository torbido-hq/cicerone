from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from support.system_db import TEST_DATABASE_URL, reset_schema
from support.system_spec import SKIP_NO_TEST_DB

from cicerone.config import make_settings
from cicerone.feature_config import FeatureColumn, FeatureConfig
from cicerone.io import options as io_options
from cicerone.io.engines import dispose_engines

# Re-export for existing `from conftest import make_settings` call sites.
__all__ = ["make_settings"]

_READ_S3_PARQUET_PYARROW = io_options._read_s3_parquet_pyarrow


@pytest.fixture(autouse=True)
def _disable_native_arrow_s3(monkeypatch: pytest.MonkeyPatch) -> None:
    def _disabled(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise OSError("native Arrow S3 is disabled in tests")

    monkeypatch.setattr(io_options, "_read_s3_parquet_pyarrow", _disabled)


@pytest.fixture
def enable_native_arrow_s3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(io_options, "_read_s3_parquet_pyarrow", _READ_S3_PARQUET_PYARROW)


@pytest.fixture(autouse=True)
def _dispose_shared_engines() -> Iterator[None]:
    dispose_engines()
    yield
    dispose_engines()


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Engine]:
    if not TEST_DATABASE_URL:
        pytest.skip(SKIP_NO_TEST_DB)
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def clean_schema(db_engine: Engine) -> Iterator[None]:
    reset_schema(db_engine)
    yield
    reset_schema(db_engine)


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
    now = pd.Timestamp.now(tz="UTC")
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
