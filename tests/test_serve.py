from __future__ import annotations

import threading

import pandas as pd
import pytest
from conftest import make_settings
from fastapi.testclient import TestClient

from cicerone.blending import COLD_START_USER_ID
from cicerone.config import Settings
from cicerone.feature_config import FeatureConfig
from cicerone.io.recommendation_reader import select_cold_start_fallback
from cicerone.serve import create_app, main
from cicerone.serve.app import _available_item_ids, _ItemsFilterCache, _start_refresh_loop


def _settings(**overrides) -> Settings:
    return make_settings(**{"mode": "serve", "serve_auth_token": "secret", **overrides})


class _FakeReader:
    def __init__(self, recs: pd.DataFrame, items: pd.DataFrame | None = None):
        self._recs = recs
        self._items = items
        self._items_version = 0
        self.refresh_calls = 0

    def refresh(self) -> None:
        self.refresh_calls += 1
        self._items_version += 1

    def get_recommendations(self, user_id: str, k: int) -> pd.DataFrame:
        rows = self._recs[self._recs["user_id"] == user_id].sort_values("rank")
        return rows.head(k).reset_index(drop=True)

    def get_items(self) -> pd.DataFrame | None:
        return self._items

    def items_version(self) -> int:
        return self._items_version

    def get_cold_start_fallback(self, k: int) -> pd.DataFrame:
        return select_cold_start_fallback(self._recs, k)

    def configure_item_filters(
        self,
        *,
        category_column: str | None = None,
        availability_filters=(),
    ) -> None:
        del category_column, availability_filters
        self._items_version += 1


def _recs_df() -> pd.DataFrame:
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


def _items_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "published": True, "in_stock": True},
            {"item_id": "i2", "category": "wine", "published": True, "in_stock": True},
            {"item_id": "i3", "category": "beer", "published": False, "in_stock": True},
        ]
    )


def _feature_config() -> FeatureConfig:
    return FeatureConfig(
        event_weights={},
        quantity_scaled_events=set(),
        event_caps={},
        user_features=[],
        item_features=[],
        item_availability_filters=["published", "in_stock"],
    )


class _FakeManifest:
    def read_latest(self):
        return {"generated_at": "2026-08-04T12:00:00+00:00"}

    def read_recent(self, limit: int):
        del limit
        latest = self.read_latest()
        return [latest] if latest else []


def test_available_item_ids_missing_item_id_returns_none():
    items = pd.DataFrame([{"sku": "x", "published": True}])
    assert _available_item_ids(items, ["published"]) is None


def test_items_filter_cache_tolerates_snapshot_without_item_id():
    reader = _FakeReader(_recs_df(), pd.DataFrame([{"sku": "x", "published": True}]))
    cache = _ItemsFilterCache(
        reader,
        category_column="category",
        availability_filters=["published"],
    )
    items, available, by_category = cache.get()
    assert items is not None
    assert available is None
    assert by_category == {}


def test_items_filter_cache_normalizes_once_per_refresh():
    reader = _FakeReader(_recs_df(), _items_df())
    cache = _ItemsFilterCache(
        reader,
        category_column="category",
        availability_filters=["published", "in_stock"],
    )
    items_a, available_a, by_cat_a = cache.get()
    items_b, available_b, by_cat_b = cache.get()
    assert items_a is items_b
    assert available_a is available_b
    assert by_cat_a is by_cat_b
    assert available_a == frozenset({"i1", "i2"})
    assert by_cat_a["beer"] == frozenset({"i1", "i3"})
    assert by_cat_a["wine"] == frozenset({"i2"})

    # In-place mutation alone must not be relied on; bump the version token.
    reader._items.loc[reader._items["item_id"] == "i3", "published"] = True
    items_same, _, by_cat_same = cache.get()
    assert items_same is items_a
    assert by_cat_same is by_cat_a

    reader.refresh()
    items_c, available_c, by_cat_c = cache.get()
    assert items_c is not items_a
    assert available_c == frozenset({"i1", "i2", "i3"})
    assert by_cat_c is not by_cat_a


def test_health_requires_no_auth():
    app = create_app(_settings(), _FakeReader(_recs_df()))
    client = TestClient(app)

    assert client.get("/health").status_code == 200


def test_recommendations_requires_auth():
    app = create_app(_settings(), _FakeReader(_recs_df()))
    client = TestClient(app)

    response = client.get("/recommendations/u1")

    assert response.status_code == 401


def test_recommendations_rejects_wrong_token():
    app = create_app(_settings(), _FakeReader(_recs_df()))
    client = TestClient(app)

    response = client.get("/recommendations/u1", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401


def test_recommendations_returns_records_with_valid_token():
    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        manifest_reader=_FakeManifest(),
        feature_config=_feature_config(),
    )
    client = TestClient(app)

    response = client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["generated_at"] == "2026-08-04T12:00:00+00:00"
    assert body["fallback"] is False
    assert [row["item_id"] for row in body["items"]] == ["i1", "i2"]
    assert response.headers["X-Generated-At"] == "2026-08-04T12:00:00+00:00"


def test_recommendations_respects_limit_query_param():
    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        feature_config=_feature_config(),
    )
    client = TestClient(app)

    response = client.get("/recommendations/u1?limit=1", headers={"Authorization": "Bearer secret"})

    assert len(response.json()["items"]) == 1


def test_recommendations_rejects_conflicting_limit_and_k():
    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        feature_config=_feature_config(),
    )
    client = TestClient(app)

    response = client.get(
        "/recommendations/u1?limit=1&k=2",
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 400


def test_recommendations_accepts_matching_limit_and_k():
    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        feature_config=_feature_config(),
    )
    client = TestClient(app)

    response = client.get(
        "/recommendations/u1?limit=1&k=1",
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200
    assert len(response.json()["items"]) == 1


def test_recommendations_limit_larger_than_available():
    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        feature_config=_feature_config(),
    )
    client = TestClient(app)

    response = client.get("/recommendations/u1?limit=100", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert len(response.json()["items"]) == 2


def test_recommendations_unknown_user_returns_cold_start_fallback():
    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        feature_config=_feature_config(),
    )
    client = TestClient(app)

    response = client.get("/recommendations/nobody", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["fallback"] is True
    assert [row["item_id"] for row in body["items"]] == ["i2", "i1"]


def test_recommendations_category_filter_can_empty_results():
    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        feature_config=_feature_config(),
    )
    client = TestClient(app)

    response = client.get(
        "/recommendations/u1?category=spirits",
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_recommendations_category_filter_keeps_matching():
    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        feature_config=_feature_config(),
    )
    client = TestClient(app)

    response = client.get(
        "/recommendations/u1?category=wine",
        headers={"Authorization": "Bearer secret"},
    )

    assert [row["item_id"] for row in response.json()["items"]] == ["i2"]


def test_recommendations_missing_category_column_returns_empty(caplog):
    items = pd.DataFrame([{"item_id": "i1", "published": True, "in_stock": True}])
    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), items),
        feature_config=_feature_config(),
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    with caplog.at_level("WARNING"):
        first = client.get("/recommendations/u1?category=beer", headers=headers)
        second = client.get("/recommendations/u1?category=beer", headers=headers)

    assert first.json()["items"] == []
    assert second.json()["items"] == []
    warnings = [r for r in caplog.records if "no column" in r.getMessage()]
    assert len(warnings) == 1


def test_recommendations_without_items_skips_filters():
    app = create_app(_settings(), _FakeReader(_recs_df(), items=None))
    client = TestClient(app)

    response = client.get("/recommendations/u1?limit=2", headers={"Authorization": "Bearer secret"})
    assert [row["item_id"] for row in response.json()["items"]] == ["i1", "i2"]


def test_recommendations_exclude_unavailable_false_keeps_unpublished():
    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        feature_config=_feature_config(),
    )
    client = TestClient(app)

    response = client.get(
        "/recommendations/u1?exclude_unavailable=false&limit=3",
        headers={"Authorization": "Bearer secret"},
    )
    assert [row["item_id"] for row in response.json()["items"]] == ["i1", "i2", "i3"]


def test_recommendations_empty_everywhere_returns_404():
    empty = pd.DataFrame(columns=["user_id", "item_id", "rank", "score", "source"])
    app = create_app(_settings(), _FakeReader(empty))
    client = TestClient(app)

    response = client.get("/recommendations/nobody", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 404


def test_recommendations_manifest_without_generated_at():
    class EmptyManifest:
        def read_latest(self):
            return {}

        def read_recent(self, limit: int):
            del limit
            return []

    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        manifest_reader=EmptyManifest(),
        feature_config=_feature_config(),
    )
    client = TestClient(app)
    body = client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"}).json()
    assert body["generated_at"] is None


def test_recommendations_survive_manifest_reader_errors():
    class BrokenManifest:
        def read_latest(self):
            raise RuntimeError("manifest store down")

        def read_recent(self, limit: int):
            del limit
            raise RuntimeError("manifest store down")

    app = create_app(
        _settings(),
        _FakeReader(_recs_df(), _items_df()),
        manifest_reader=BrokenManifest(),
        feature_config=_feature_config(),
    )
    client = TestClient(app)
    response = client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert response.json()["generated_at"] is None
    assert "X-Generated-At" not in response.headers


def test_start_refresh_loop_calls_refresh_periodically(monkeypatch):
    reader = _FakeReader(_recs_df())
    calls = {"sleep": 0}

    def fake_sleep(_seconds):
        calls["sleep"] += 1
        if calls["sleep"] >= 3:
            raise SystemExit("stop after three ticks")

    monkeypatch.setattr("cicerone.serve.app.time.sleep", fake_sleep)
    monkeypatch.setattr(threading.Thread, "start", lambda self: self.run())

    with pytest.raises(SystemExit):
        _start_refresh_loop(reader, interval_seconds=0.01)

    assert reader.refresh_calls >= 2


def test_main_requires_serve_mode(tmp_path, monkeypatch):
    config_path = tmp_path / "cicerone.toml"
    config_path.write_text(
        f"""
        [job]
        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "{tmp_path}"
        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "{tmp_path}"
        """
    )
    monkeypatch.setenv("CICERONE_CONFIG_PATH", str(config_path))

    with pytest.raises(RuntimeError, match="requires mode = 'serve'"):
        main()


def test_main_starts_serve_app_in_serve_mode(tmp_path, monkeypatch):
    config_path = tmp_path / "cicerone.toml"
    features_path = tmp_path / "features.toml"
    features_path.write_text("item_availability_filters = []\n")
    config_path.write_text(
        f"""
        [job]
        mode = "serve"
        feature_config_path = "{features_path}"

        [serve]
        auth_token = "secret"

        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "{tmp_path}"
        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "{tmp_path}"
        """
    )
    monkeypatch.setenv("CICERONE_CONFIG_PATH", str(config_path))

    refresh_calls = []
    refresh_kwargs: list[dict] = []
    uvicorn_calls = {}
    served_app = None

    import cicerone.serve.app as serve_module

    def fake_start_refresh_loop(reader, interval, **kwargs):
        refresh_calls.append(reader)
        refresh_kwargs.append(kwargs)

    def fake_uvicorn_run(app, host, port):
        nonlocal served_app
        served_app = app
        uvicorn_calls.update(host=host, port=port)

    monkeypatch.setattr(serve_module, "_start_refresh_loop", fake_start_refresh_loop)
    monkeypatch.setattr(serve_module, "uvicorn", type("_U", (), {"run": staticmethod(fake_uvicorn_run)}))

    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}]
    ).to_parquet(tmp_path / "recommendations.parquet", index=False)

    main()

    assert len(refresh_calls) == 1
    assert uvicorn_calls == {"host": "0.0.0.0", "port": 8000}
    assert served_app is not None
    assert refresh_kwargs[0]["generated_at_cache"] is served_app.state.generated_at_cache


def test_main_fails_closed_on_invalid_feature_config(tmp_path, monkeypatch):
    config_path = tmp_path / "cicerone.toml"
    features_path = tmp_path / "features.toml"
    features_path.write_text("blending = { enabled = true, steepness = -1 }\n")
    config_path.write_text(
        f"""
        [job]
        mode = "serve"
        feature_config_path = "{features_path}"

        [serve]
        auth_token = "secret"

        [input]
        kind = "dataset"
        [input.options]
        storage_backend = "local"
        path = "{tmp_path}"
        [output]
        kind = "dataset"
        [output.options]
        storage_backend = "local"
        path = "{tmp_path}"
        """
    )
    monkeypatch.setenv("CICERONE_CONFIG_PATH", str(config_path))
    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}]
    ).to_parquet(tmp_path / "recommendations.parquet", index=False)

    with pytest.raises(ValueError, match="steepness"):
        main()
