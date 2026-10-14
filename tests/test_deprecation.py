from __future__ import annotations

import importlib

import pytest

RENAMED = [
    ("cicerone.evaluation", "_annotate_source", "cicerone.evaluation.tracking", "annotate_impression_source"),
    ("cicerone.evaluation", "_merge_asof_events", "cicerone.evaluation.metrics", "merge_asof_events"),
    ("cicerone.evaluation", "_recs_from_history", "cicerone.evaluation.served", "recs_from_history"),
    (
        "cicerone.io.recommendation_reader",
        "_FALLBACK_SOURCES",
        "cicerone.io.recommendation_reader_common",
        "FALLBACK_SOURCES",
    ),
    (
        "cicerone.io.recommendation_reader",
        "_FALLBACK_SOURCE_PRIORITY",
        "cicerone.io.recommendation_reader_common",
        "FALLBACK_SOURCE_PRIORITY",
    ),
    (
        "cicerone.io.recommendation_reader",
        "_ItemFilterMixin",
        "cicerone.io.recommendation_reader_common",
        "ItemFilterMixin",
    ),
    (
        "cicerone.io.recommendation_reader",
        "_best_fallback_user_id",
        "cicerone.io.recommendation_reader_common",
        "best_fallback_user_id",
    ),
    (
        "cicerone.io.recommendation_reader",
        "_index_recommendations_by_user",
        "cicerone.io.recommendation_reader_common",
        "index_recommendations_by_user",
    ),
    (
        "cicerone.io.recommendation_reader",
        "_pick_fallback_user",
        "cicerone.io.recommendation_reader_common",
        "pick_fallback_user",
    ),
    (
        "cicerone.io.recommendation_reader",
        "_resolve_fallback_user_id",
        "cicerone.io.recommendation_reader_common",
        "resolve_fallback_user_id",
    ),
    ("cicerone.track.store", "_history_frame", "cicerone.track.store_common", "history_frame"),
    ("cicerone.track.store", "_history_part_name", "cicerone.track.store_dataset", "history_part_name"),
    ("cicerone.track.store", "_jsonish", "cicerone.track.store_common", "jsonish"),
    ("cicerone.locks", "_acquire_once", "cicerone.locks.hold", "acquire_once"),
    ("cicerone.locks", "_acquire_until", "cicerone.locks.hold", "acquire_until"),
    ("cicerone.locks", "_bind_writer_generation", "cicerone.locks.hold", "bind_writer_generation"),
    ("cicerone.locks", "_bound_writer_generation", "cicerone.locks.hold", "bound_writer_generation"),
    ("cicerone.locks", "_lock_owned", "cicerone.locks.hold", "lock_owned"),
    ("cicerone.locks", "_unbind_writer_generation", "cicerone.locks.hold", "unbind_writer_generation"),
    ("cicerone.locks", "_writer_hold", "cicerone.locks.hold", "writer_hold"),
]


@pytest.mark.parametrize(("module", "old", "home", "new"), RENAMED)
def test_renamed_export_warns_and_resolves_to_public_name(module: str, old: str, home: str, new: str) -> None:
    expected = getattr(importlib.import_module(home), new)
    with pytest.warns(
        FutureWarning, match=rf"{module}\.{old} is deprecated and will be removed in 0\.9\.0; use "
    ):
        assert getattr(importlib.import_module(module), old) is expected


def test_from_import_of_renamed_export_still_works() -> None:
    from cicerone.evaluation.tracking import annotate_impression_source

    with pytest.warns(FutureWarning):
        from cicerone.evaluation import _annotate_source
    assert _annotate_source is annotate_impression_source


@pytest.mark.parametrize("module", sorted({row[0] for row in RENAMED}))
def test_unknown_attribute_still_raises(module: str) -> None:
    with pytest.raises(AttributeError, match="has no attribute '_not_a_name'"):
        _ = importlib.import_module(module)._not_a_name


@pytest.mark.parametrize("module", sorted({row[0] for row in RENAMED if row[0] != "cicerone.locks"}))
def test_star_import_keeps_renamed_exports(module: str) -> None:
    old_names = {row[1] for row in RENAMED if row[0] == module}
    namespace: dict[str, object] = {}
    with pytest.warns(FutureWarning):
        exec(f"from {module} import *", namespace)
    assert old_names <= namespace.keys()
