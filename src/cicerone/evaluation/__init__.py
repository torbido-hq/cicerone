"""CTR/CVR attribution and production replay of served lists."""

from __future__ import annotations

from typing import Any

from cicerone.deprecation import renamed_attribute
from cicerone.evaluation.context import (
    conversion_events_for_settings,
    generated_ats_from_track,
    load_metric_events,
)
from cicerone.evaluation.metrics import OCCURRED_AT, SliceMetrics, merge_asof_events
from cicerone.evaluation.served import (
    ServedEvalReport,
    evaluate_served,
    filter_events_to_recommended,
    filter_recs_to_assigned,
    recs_from_history,
    recs_from_impressions,
    replay_ks,
)
from cicerone.evaluation.tracking import (
    DEFAULT_CONVERSION_TYPE,
    TrackEvalReport,
    annotate_impression_source,
    conversion_event_types,
    conversion_events,
    evaluate_tracking,
    filter_events_by_types,
    user_track_outcomes,
)

__all__ = [
    "DEFAULT_CONVERSION_TYPE",
    "OCCURRED_AT",
    "ServedEvalReport",
    "SliceMetrics",
    "TrackEvalReport",
    "annotate_impression_source",
    "conversion_event_types",
    "conversion_events",
    "conversion_events_for_settings",
    "evaluate_served",
    "evaluate_tracking",
    "filter_events_by_types",
    "filter_events_to_recommended",
    "filter_recs_to_assigned",
    "generated_ats_from_track",
    "load_metric_events",
    "merge_asof_events",
    "recs_from_history",
    "recs_from_impressions",
    "replay_ks",
    "user_track_outcomes",
]


_RENAMED = {
    "_annotate_source": "cicerone.evaluation.annotate_impression_source",
    "_merge_asof_events": "cicerone.evaluation.merge_asof_events",
    "_recs_from_history": "cicerone.evaluation.recs_from_history",
}
__all__ += list(_RENAMED)


def __getattr__(name: str) -> Any:
    return renamed_attribute(__name__, _RENAMED, name)
