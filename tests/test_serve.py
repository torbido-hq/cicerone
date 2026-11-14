from __future__ import annotations

import threading
from types import SimpleNamespace

import pandas as pd
import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from support.fakes import FakeManifest, FakeReader, patch_module_ref, raising
from support.serve import items_df, recs_df, serve_feature_config, serve_settings

from cicerone.experiment.assignment import snapshot_variant_names
from cicerone.refreshed_value import RefreshedValue
from cicerone.serve import create_app, main
from cicerone.serve.app import (
    _append_with_busy_retry,
    _read_generated_at,
    _refresh_step,
    _start_refresh_loop,
)
from cicerone.serve.item_filters import ItemsFilterCache, available_item_ids
from cicerone.serve.recommendations import read_assignment_overlay


def test_available_item_ids_missing_item_id_returns_none():
    items = pd.DataFrame([{"sku": "x", "published": True}])
    assert available_item_ids(items, ["published"]) is None


def test_items_filter_cache_tolerates_snapshot_without_item_id():
    reader = FakeReader(recs_df(), pd.DataFrame([{"sku": "x", "published": True}]))
    cache = ItemsFilterCache(
        reader,
        category_column="category",
        availability_filters=["published"],
    )
    items, available, by_category = cache.get()
    assert items is not None
    assert available is None
    assert by_category == {}


def test_items_filter_cache_normalizes_once_per_refresh():
    reader = FakeReader(recs_df(), items_df())
    cache = ItemsFilterCache(
        reader,
        category_column="category",
        availability_filters=["published", "in_stock"],
    )
    items_a, available_a, by_cat_a = cache.get()
    items_b, available_b, by_cat_b = cache.get()
    assert reader.get_items_calls == 1
    assert items_a is items_b
    assert available_a is available_b
    assert by_cat_a is by_cat_b
    assert available_a == frozenset({"i1", "i2"})
    assert by_cat_a["beer"] == frozenset({"i1", "i3"})
    assert by_cat_a["wine"] == frozenset({"i2"})
    assert all(isinstance(item_id, str) for item_id in by_cat_a["beer"])

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


def test_items_filter_cache_retries_when_version_moves_during_rebuild():
    class _BumpOnce(FakeReader):
        def get_items(self) -> pd.DataFrame | None:
            self.get_items_calls += 1
            if self.get_items_calls == 1:
                self._items_version += 1
            return self._items

    reader = _BumpOnce(recs_df(), items_df())
    cache = ItemsFilterCache(
        reader,
        category_column="category",
        availability_filters=["published"],
    )
    items, available, by_cat = cache.get()
    assert reader.get_items_calls == 2
    assert cache._snapshot is not None
    assert cache._snapshot[0] == reader.items_version()
    assert items is not None
    assert available == frozenset({"i1", "i2"})
    assert by_cat["beer"] == frozenset({"i1", "i3"})


def _item_scores_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item_id": "a", "popular_score": 1.0, "latest_score": 0.0, "n_users": 1},
            {"item_id": "b", "popular_score": 2.0, "latest_score": 1.5, "n_users": 2},
            {"item_id": "c", "popular_score": 3.0, "latest_score": 0.0, "n_users": 1},
        ]
    )


def test_health_requires_no_auth():
    app = create_app(serve_settings(), FakeReader(recs_df()))
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_recommendations_requires_auth():
    app = create_app(serve_settings(), FakeReader(recs_df()))
    client = TestClient(app)

    response = client.get("/recommendations/u1")

    assert response.status_code == 401


def test_recommendations_rejects_wrong_token():
    app = create_app(serve_settings(), FakeReader(recs_df()))
    client = TestClient(app)

    response = client.get("/recommendations/u1", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401


def test_recommendations_returns_records_with_valid_token():
    app = create_app(
        serve_settings(),
        FakeReader(recs_df(), items_df()),
        manifest_reader=FakeManifest(),
        feature_config=serve_feature_config(),
    )
    client = TestClient(app)

    response = client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["generated_at"] == "2026-08-04T12:00:00+00:00"
    assert body["fallback"] is False
    assert [row["item_id"] for row in body["items"]] == ["i1", "i2"]
    assert all(row["reasons"] is None for row in body["items"])
    assert response.headers["X-Generated-At"] == "2026-08-04T12:00:00+00:00"


def test_recommendations_returns_parsed_reasons():
    recs = recs_df()
    recs["reasons"] = [
        (
            '{"sources":[{"label":"personalized","rank":1,"weight":1.0,"contribution":null}],'
            '"boosts":[],"similar_items":[{"item_id":"i9","score":0.5}],'
            '"matched_attributes":[{"column":"style","value":"lager"}]}'
        ),
        None,
        None,
        None,
        None,
    ]
    app = create_app(
        serve_settings(),
        FakeReader(recs, items_df()),
        feature_config=serve_feature_config(),
    )
    client = TestClient(app)

    response = client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    first = response.json()["items"][0]
    assert first["reasons"]["sources"][0]["label"] == "personalized"
    assert first["reasons"]["similar_items"][0]["item_id"] == "i9"
    assert response.json()["items"][1]["reasons"] is None


def test_recommendations_respects_limit_query_param():
    app = create_app(
        serve_settings(),
        FakeReader(recs_df(), items_df()),
        feature_config=serve_feature_config(),
    )
    client = TestClient(app)

    response = client.get("/recommendations/u1?limit=1", headers={"Authorization": "Bearer secret"})

    assert len(response.json()["items"]) == 1


def test_recommendations_accepts_k_alias():
    app = create_app(
        serve_settings(),
        FakeReader(recs_df(), items_df()),
        feature_config=serve_feature_config(),
    )
    response = TestClient(app).get("/recommendations/u1?k=1", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert len(response.json()["items"]) == 1


def test_recommendations_rejects_conflicting_limit_and_k():
    app = create_app(
        serve_settings(),
        FakeReader(recs_df(), items_df()),
        feature_config=serve_feature_config(),
    )
    client = TestClient(app)

    response = client.get(
        "/recommendations/u1?limit=1&k=2",
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 400


def test_recommendations_accepts_matching_limit_and_k():
    app = create_app(
        serve_settings(),
        FakeReader(recs_df(), items_df()),
        feature_config=serve_feature_config(),
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
        serve_settings(),
        FakeReader(recs_df(), items_df()),
        feature_config=serve_feature_config(),
    )
    client = TestClient(app)

    response = client.get("/recommendations/u1?limit=100", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert len(response.json()["items"]) == 2


def test_recommendations_limit_over_cap_is_unprocessable():
    app = create_app(serve_settings(), FakeReader(recs_df()))
    response = TestClient(app).get(
        "/recommendations/u1?limit=101", headers={"Authorization": "Bearer secret"}
    )
    assert response.status_code == 422


def test_recommendations_unknown_user_returns_cold_start_fallback():
    app = create_app(
        serve_settings(),
        FakeReader(recs_df(), items_df()),
        feature_config=serve_feature_config(),
    )
    client = TestClient(app)

    response = client.get("/recommendations/nobody", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["fallback"] is True
    assert [row["item_id"] for row in body["items"]] == ["i2", "i1"]


def test_items_filter_cache_stringifies_non_string_item_ids():
    """Category allowlists must be str so they match filtered recommendation ids."""
    from cicerone.serve.item_filters import filter_recommendations

    recs = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "10", "rank": 1, "score": 1.0, "source": "personalized"},
            {"user_id": "u1", "item_id": "20", "rank": 2, "score": 0.5, "source": "personalized"},
        ]
    )
    items = pd.DataFrame(
        [
            {"item_id": 10, "category": "beer", "published": True},
            {"item_id": 20, "category": "wine", "published": True},
        ]
    )
    reader = FakeReader(recs, items)
    cache = ItemsFilterCache(
        reader,
        category_column="category",
        availability_filters=["published"],
    )
    cached_items, available, by_cat = cache.get()
    assert by_cat["beer"] == frozenset({"10"})
    filtered = filter_recommendations(
        recs,
        items=cached_items,
        available_ids=available,
        category="beer",
        category_column="category",
        exclude_unavailable=True,
        ids_by_category=by_cat,
        on_missing_category_column=lambda: None,
    )
    assert list(filtered["item_id"]) == ["10"]


def test_recommendations_category_filter_can_empty_results():
    app = create_app(
        serve_settings(),
        FakeReader(recs_df(), items_df()),
        feature_config=serve_feature_config(),
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
        serve_settings(),
        FakeReader(recs_df(), items_df()),
        feature_config=serve_feature_config(),
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
        serve_settings(),
        FakeReader(recs_df(), items),
        feature_config=serve_feature_config(),
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
    app = create_app(serve_settings(), FakeReader(recs_df(), items=None))
    client = TestClient(app)

    response = client.get("/recommendations/u1?limit=2", headers={"Authorization": "Bearer secret"})
    assert [row["item_id"] for row in response.json()["items"]] == ["i1", "i2"]


def test_recommendations_exclude_unavailable_false_keeps_unpublished():
    app = create_app(
        serve_settings(),
        FakeReader(recs_df(), items_df()),
        feature_config=serve_feature_config(),
    )
    client = TestClient(app)

    response = client.get(
        "/recommendations/u1?exclude_unavailable=false&limit=3",
        headers={"Authorization": "Bearer secret"},
    )
    assert [row["item_id"] for row in response.json()["items"]] == ["i1", "i2", "i3"]


def test_recommendations_exclude_unavailable_treats_false_strings_as_unavailable():
    items = pd.DataFrame(
        [
            {"item_id": "i1", "category": "beer", "published": "true", "in_stock": "1"},
            {"item_id": "i2", "category": "wine", "published": "false", "in_stock": "true"},
            {"item_id": "i3", "category": "beer", "published": "0", "in_stock": "yes"},
        ]
    )
    recs = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rank": 1, "score": 1.0, "source": "personalized"},
            {"user_id": "u1", "item_id": "i2", "rank": 2, "score": 0.5, "source": "personalized"},
            {"user_id": "u1", "item_id": "i3", "rank": 3, "score": 0.1, "source": "personalized"},
        ]
    )
    app = create_app(
        serve_settings(),
        FakeReader(recs, items),
        feature_config=serve_feature_config(),
    )
    client = TestClient(app)
    response = client.get(
        "/recommendations/u1?exclude_unavailable=true&limit=3",
        headers={"Authorization": "Bearer secret"},
    )
    assert [row["item_id"] for row in response.json()["items"]] == ["i1"]


def test_recommendations_empty_everywhere_returns_404():
    empty = pd.DataFrame(columns=["user_id", "item_id", "rank", "score", "source"])
    app = create_app(serve_settings(), FakeReader(empty))
    client = TestClient(app)

    response = client.get("/recommendations/nobody", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 404


def test_recommendations_batch_requires_auth():
    app = create_app(serve_settings(), FakeReader(recs_df()))
    response = TestClient(app).post("/recommendations/batch", json={"user_ids": ["u1"]})
    assert response.status_code == 401


def test_recommendations_batch_matches_single_get():
    recs = pd.concat(
        [
            recs_df(),
            pd.DataFrame(
                [{"user_id": "u2", "item_id": "i2", "rank": 1, "score": 0.8, "source": "personalized"}]
            ),
        ],
        ignore_index=True,
    )
    app = create_app(
        serve_settings(),
        FakeReader(recs, items_df()),
        manifest_reader=FakeManifest(),
        feature_config=serve_feature_config(),
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    batch = client.post(
        "/recommendations/batch",
        json={"user_ids": ["u1", "u2", "nobody", "u1"], "limit": 2},
        headers=headers,
    )
    assert batch.status_code == 200
    assert batch.headers.get("X-Generated-At")
    users = {row["user_id"]: row for row in batch.json()["users"]}
    assert list(users) == ["u1", "u2", "nobody"]
    for user_id in users:
        single = client.get(f"/recommendations/{user_id}?limit=2", headers=headers)
        assert single.status_code == 200
        assert users[user_id] == single.json()


def test_recommendations_batch_missing_user_is_empty_not_404():
    empty = pd.DataFrame(columns=["user_id", "item_id", "rank", "score", "source"])
    app = create_app(serve_settings(), FakeReader(empty), manifest_reader=FakeManifest())
    response = TestClient(app).post(
        "/recommendations/batch",
        json={"user_ids": ["nobody"]},
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["users"] == [
        {
            "generated_at": "2026-08-04T12:00:00+00:00",
            "user_id": "nobody",
            "fallback": False,
            "items": [],
            "experiment_id": None,
            "variant": None,
        }
    ]


def test_recommendations_batch_rejects_blank_user_id():
    app = create_app(serve_settings(), FakeReader(recs_df()))
    response = TestClient(app).post(
        "/recommendations/batch",
        json={"user_ids": ["u1", "  "]},
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "user_ids must be non-blank"


def test_recommendations_batch_rejects_empty_and_oversized_lists():
    app = create_app(serve_settings(), FakeReader(recs_df()))
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    assert client.post("/recommendations/batch", json={"user_ids": []}, headers=headers).status_code == 422
    too_many = {"user_ids": [f"u{i}" for i in range(101)]}
    assert client.post("/recommendations/batch", json=too_many, headers=headers).status_code == 422


def test_recommendations_batch_uses_request_scoped_generated_at(monkeypatch):
    stamps = iter(["t-1", "t-2", "t-3"])
    recs = pd.concat(
        [
            recs_df(),
            pd.DataFrame(
                [{"user_id": "u2", "item_id": "i2", "rank": 1, "score": 0.8, "source": "personalized"}]
            ),
        ],
        ignore_index=True,
    )
    app = create_app(serve_settings(), FakeReader(recs), manifest_reader=FakeManifest())
    monkeypatch.setattr(app.state.generated_at_cache, "get", lambda: next(stamps))
    response = TestClient(app).post(
        "/recommendations/batch",
        json={"user_ids": ["u1", "u2"]},
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["generated_at"] == "t-1"
    assert [row["generated_at"] for row in body["users"]] == ["t-1", "t-1"]
    assert response.headers["X-Generated-At"] == "t-1"


def test_recommendations_batch_uses_bulk_reader():
    class CountingReader(FakeReader):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.single_calls = 0
            self.bulk_calls = 0

        def get_recommendations(self, user_id: str, k: int, *, variant: str | None = None) -> pd.DataFrame:
            self.single_calls += 1
            return super().get_recommendations(user_id, k, variant=variant)

        def get_recommendations_for_users(
            self, user_ids, k: int, *, variant: str | None = None
        ) -> dict[str, pd.DataFrame]:
            self.bulk_calls += 1
            return {
                str(user_id): FakeReader.get_recommendations(self, str(user_id), k, variant=variant)
                for user_id in user_ids
            }

        def get_cold_start_fallback(self, k: int, *, variant: str | None = None) -> pd.DataFrame:
            return super().get_cold_start_fallback(k, variant=variant)

    recs = pd.concat(
        [
            recs_df(),
            pd.DataFrame(
                [{"user_id": "u2", "item_id": "i2", "rank": 1, "score": 0.8, "source": "personalized"}]
            ),
        ],
        ignore_index=True,
    )
    reader = CountingReader(recs)
    app = create_app(serve_settings(), reader, manifest_reader=FakeManifest())
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    single = client.get("/recommendations/u1", headers=headers)
    assert single.status_code == 200
    assert reader.single_calls == 1
    assert reader.bulk_calls == 0
    batch = client.post("/recommendations/batch", json={"user_ids": ["u1", "u2", "nobody"]}, headers=headers)
    assert batch.status_code == 200
    assert reader.bulk_calls == 1
    assert reader.single_calls == 1


def test_recommendations_batch_applies_category():
    app = create_app(
        serve_settings(),
        FakeReader(recs_df(), items_df()),
        feature_config=serve_feature_config(),
    )
    response = TestClient(app).post(
        "/recommendations/batch",
        json={"user_ids": ["u1"], "category": "wine"},
        headers={"Authorization": "Bearer secret"},
    )
    assert [row["item_id"] for row in response.json()["users"][0]["items"]] == ["i2"]


def test_recommendations_manifest_without_generated_at():
    app = create_app(
        serve_settings(),
        FakeReader(recs_df(), items_df()),
        manifest_reader=FakeManifest({}),
        feature_config=serve_feature_config(),
    )
    client = TestClient(app)
    body = client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"}).json()
    assert body["generated_at"] is None


@pytest.mark.parametrize(
    "error", [OSError("manifest store down"), ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")]
)
def test_recommendations_survive_manifest_reader_errors(error):
    class BrokenManifest:
        def read_latest(self):
            raise error

        def read_recent(self, limit: int):
            del limit
            raise error

    app = create_app(
        serve_settings(),
        FakeReader(recs_df(), items_df()),
        manifest_reader=BrokenManifest(),
        feature_config=serve_feature_config(),
    )
    client = TestClient(app)
    response = client.get("/recommendations/u1", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert response.json()["generated_at"] is None
    assert "X-Generated-At" not in response.headers


def test_serve_startup_fails_on_unexpected_manifest_read_error():
    manifest = FakeManifest({})
    manifest.read_latest = raising(TypeError("manifest reader bug"))  # type: ignore[method-assign]
    with pytest.raises(TypeError, match="manifest reader bug"):
        create_app(serve_settings(), FakeReader(recs_df(), items_df()), manifest_reader=manifest)


def test_append_logs_an_unexpected_error_without_raising(caplog):
    write = raising(TypeError("append bug"))
    _append_with_busy_retry(write, label="record exposure", user_id="u1")
    assert write.calls == 1
    assert "Failed to record exposure for user_id='u1'" in caplog.text


def test_generated_at_cache_reads_and_refreshes_manifest():
    class MutableManifest:
        def __init__(self) -> None:
            self.payload: dict | None = {"generated_at": "2026-08-01T00:00:00+00:00"}

        def read_latest(self):
            return self.payload

        def read_recent(self, limit: int):
            del limit
            latest = self.read_latest()
            return [latest] if latest else []

    manifest = MutableManifest()
    cache = RefreshedValue(lambda: _read_generated_at(manifest))
    cache.refresh()
    assert cache.get() == "2026-08-01T00:00:00+00:00"

    manifest.payload = {"generated_at": "2026-08-12T12:00:00+00:00"}
    assert cache.get() == "2026-08-01T00:00:00+00:00"
    cache.refresh()
    assert cache.get() == "2026-08-12T12:00:00+00:00"

    manifest.payload = None
    cache.refresh()
    assert cache.get() is None


def _run_refresh_ticks(monkeypatch, reader, ticks: int, generated_at_cache=None) -> dict[str, int]:
    calls = {"sleep": 0, "generated": 0, "overlay": 0, "surfaces": 0}

    class FakeCache:
        def __init__(self, key: str) -> None:
            self._key = key

        def refresh(self) -> None:
            calls[self._key] += 1

    def fake_sleep(_seconds):
        calls["sleep"] += 1
        if calls["sleep"] > ticks:
            raise SystemExit("stop")

    patch_module_ref(monkeypatch, "cicerone.serve.app.time", sleep=fake_sleep)

    class _InlineThread(threading.Thread):
        def start(self) -> None:
            self.run()

    patch_module_ref(monkeypatch, "cicerone.serve.app.threading", Thread=_InlineThread)
    with pytest.raises(SystemExit):
        _start_refresh_loop(
            reader,
            interval_seconds=0.01,
            generated_at_cache=generated_at_cache or FakeCache("generated"),
            overlay_cache=FakeCache("overlay"),
            surfaces=FakeCache("surfaces"),
        )
    return calls


def test_refresh_loop_omits_generated_at_after_an_unexpected_manifest_error(monkeypatch):
    manifest = FakeManifest()
    cache = RefreshedValue(lambda: _read_generated_at(manifest))
    cache.refresh()
    assert cache.get() is not None
    manifest.read_latest = raising(TypeError("manifest bug"))  # type: ignore[method-assign]
    _run_refresh_ticks(monkeypatch, FakeReader(recs_df()), ticks=1, generated_at_cache=cache)
    assert manifest.read_latest.calls == 1
    assert cache.get() is None


def test_start_refresh_loop_calls_refresh_periodically(monkeypatch):
    reader = FakeReader(recs_df())
    calls = _run_refresh_ticks(monkeypatch, reader, ticks=2)
    assert reader.refresh_calls == 2
    assert calls["generated"] == 2
    assert calls["overlay"] == 2
    assert calls["surfaces"] == 2


def test_start_refresh_loop_survives_a_failing_step(monkeypatch, caplog):
    failures = {"reader": 0}

    class FailingReader(FakeReader):
        def refresh(self) -> None:
            failures["reader"] += 1
            raise TypeError("custom reader bug")

    with caplog.at_level("ERROR", logger="cicerone.serve.app"):
        calls = _run_refresh_ticks(monkeypatch, FailingReader(recs_df()), ticks=3)
    assert failures["reader"] == 3
    assert calls["overlay"] == 3
    # The reader kept its old rows, so the stamp must not move to a newer manifest.
    assert calls["generated"] == 0
    assert sum("Background refresh of FailingReader failed" in r.getMessage() for r in caplog.records) == 3


def test_start_refresh_loop_holds_the_stamp_when_the_reader_keeps_old_rows(monkeypatch):
    outcomes = iter([False, True, None])

    class KeepingReader(FakeReader):
        def refresh(self) -> bool | None:
            return next(outcomes)

    calls = _run_refresh_ticks(monkeypatch, KeepingReader(recs_df()), ticks=3)
    assert calls["generated"] == 2
    assert calls["overlay"] == 3


def test_assignment_overlay_cache_keeps_last_on_refresh_error():
    from cicerone.config.settings import ExperimentSettings, VariantSettings

    settings = serve_settings()
    assert RefreshedValue(lambda: read_assignment_overlay(settings, None)).get() == (None, None)
    assert read_assignment_overlay(settings, None) == (None, None)

    enabled = serve_settings(
        experiment=ExperimentSettings(
            enabled=True,
            id="exp",
            variants=(
                VariantSettings(name="control", traffic=0.5),
                VariantSettings(name="treatment", traffic=0.5),
            ),
        )
    )
    assert read_assignment_overlay(enabled, None) == (None, None)

    class _Store:
        n = 0

        def assignment_overlay(self, experiment_id):
            del experiment_id
            self.n += 1
            if self.n > 1:
                raise RuntimeError("later")
            return "treatment", ("control", "treatment")

    store = _Store()
    cache = RefreshedValue(lambda: read_assignment_overlay(enabled, store))
    cache.refresh()
    assert cache.get() == ("treatment", ("control", "treatment"))
    assert _refresh_step(cache) is False
    assert cache.get() == ("treatment", ("control", "treatment"))


def test_snapshot_variant_names_from_reader():
    assert snapshot_variant_names(FakeReader(recs_df())) is None

    class _Named(FakeReader):
        def present_variant_names(self):
            return None

    assert snapshot_variant_names(_Named(recs_df())) is None

    class _Present(FakeReader):
        def present_variant_names(self):
            return ("control", "blend")

    assert snapshot_variant_names(_Present(recs_df())) == ("control", "blend")


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
    assert uvicorn_calls == {"host": "127.0.0.1", "port": 8000}
    assert served_app is not None
    assert refresh_kwargs[0]["generated_at_cache"] is served_app.state.generated_at_cache
    assert refresh_kwargs[0]["surfaces"] is not None


def test_main_serve_error_wins_over_a_failing_events_stop(monkeypatch, caplog):
    import cicerone.serve.app as serve_module

    stop = raising(TypeError("stop bug"))
    original = OSError("bind failed")
    monkeypatch.setattr(serve_module, "load_settings", serve_settings)
    monkeypatch.setattr(
        "cicerone.io.factory.build_recommendation_reader", lambda _output: FakeReader(recs_df())
    )
    monkeypatch.setattr("cicerone.io.factory.build_manifest_reader", lambda _output: FakeManifest())
    monkeypatch.setattr(
        serve_module,
        "start_events_runtime",
        lambda *_args, **_kwargs: SimpleNamespace(webhook_source=None, worker=None, stop=stop),
    )
    monkeypatch.setattr(serve_module, "_start_refresh_loop", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(serve_module, "uvicorn", SimpleNamespace(run=raising(original)))
    with pytest.raises(OSError) as caught:
        main()
    assert caught.value is original
    assert stop.calls == 1
    assert "Event runtime stop after a serve error failed" in caplog.text


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


def test_main_allows_missing_feature_config(tmp_path, monkeypatch):
    config_path = tmp_path / "cicerone.toml"
    missing_features = tmp_path / "missing-features.toml"
    config_path.write_text(
        f"""
        [job]
        mode = "serve"
        feature_config_path = "{missing_features}"

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

    uvicorn_calls = {}
    import cicerone.serve.app as serve_module

    def fake_start_refresh_loop(reader, interval, **kwargs):
        return None

    def fake_uvicorn_run(app, host, port):
        uvicorn_calls.update(host=host, port=port)

    monkeypatch.setattr(serve_module, "_start_refresh_loop", fake_start_refresh_loop)
    monkeypatch.setattr(serve_module, "uvicorn", type("_U", (), {"run": staticmethod(fake_uvicorn_run)}))

    pd.DataFrame(
        [{"user_id": "u1", "item_id": "i1", "rank": 1, "score": 0.9, "source": "personalized"}]
    ).to_parquet(tmp_path / "recommendations.parquet", index=False)

    main()
    assert uvicorn_calls == {"host": "127.0.0.1", "port": 8000}


def test_item_scores_requires_auth():
    app = create_app(serve_settings(), FakeReader(recs_df(), item_scores=_item_scores_df()))
    assert TestClient(app).get("/item-scores").status_code == 401


def test_item_scores_empty_store():
    app = create_app(serve_settings(), FakeReader(recs_df()))
    response = TestClient(app).get("/item-scores", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}


def test_item_scores_paginates_and_filters():
    app = create_app(serve_settings(), FakeReader(recs_df(), item_scores=_item_scores_df()))
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    first = client.get("/item-scores", params={"limit": 2}, headers=headers)
    assert first.status_code == 200
    body = first.json()
    assert [row["item_id"] for row in body["items"]] == ["a", "b"]
    assert body["next_cursor"] == "b"

    second = client.get("/item-scores", params={"limit": 2, "cursor": "b"}, headers=headers)
    assert [row["item_id"] for row in second.json()["items"]] == ["c"]
    assert second.json()["next_cursor"] is None

    one = client.get("/item-scores", params={"item_id": "b"}, headers=headers)
    assert one.json() == {
        "items": [{"item_id": "b", "popular_score": 2.0, "latest_score": 1.5, "n_users": 2}],
        "next_cursor": None,
    }
    missing = client.get("/item-scores", params={"item_id": "z"}, headers=headers)
    assert missing.json() == {"items": [], "next_cursor": None}
