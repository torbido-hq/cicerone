"""Load events and recommendation snapshots used by track eval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from sqlalchemy import bindparam, create_engine, text

from cicerone.config.settings import Settings
from cicerone.evaluation.tracking import conversion_events, filter_events_by_types
from cicerone.io.db_store import DEFAULT_EVENTS_TABLE
from cicerone.io.factory import build_input_source
from cicerone.io.options import is_s3_not_found, read_parquet, require_option, sql_identifier
from cicerone.io.recommendation_schema import USER_COLUMN

EVENT_METRIC_COLUMNS = (USER_COLUMN, "item_id", "event_type", "quantity", "occurred_at")


def generated_ats_from_track(
    rows: Sequence[Mapping[str, Any]],
    *extra: str | None,
) -> set[str]:
    wanted = {str(row.get("generated_at") or "") for row in rows}
    wanted.discard("")
    for stamp in extra:
        if stamp:
            wanted.add(str(stamp))
    return wanted


def stamp_recommendations(recs: pd.DataFrame | None, generated_at: str | None) -> pd.DataFrame | None:
    if recs is None or not generated_at:
        return recs
    stamped = recs.copy()
    stamped["generated_at"] = generated_at
    return stamped


def concat_history(history: pd.DataFrame | None, recs: pd.DataFrame | None) -> pd.DataFrame | None:
    if history is None:
        return recs
    if recs is None:
        return history
    return pd.concat([history, recs], ignore_index=True)


def prefer_history(history: pd.DataFrame | None, recs: pd.DataFrame | None) -> pd.DataFrame | None:
    if history is not None and not history.empty:
        return history
    return recs


def conversion_events_for_settings(events: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    return conversion_events(
        events,
        settings.track.conversion_event_types,
        primary_metric=settings.experiment.primary_metric,
    )


def load_metric_events(settings: Settings, *, event_types: Sequence[str] | None = None) -> pd.DataFrame:
    inp = settings.input
    types = tuple(event_types) if event_types else None
    if inp.kind == "dataset":
        try:
            filters = [("event_type", "in", list(types))] if types else None
            frame = read_parquet(
                inp.options, "events.parquet", columns=list(EVENT_METRIC_COLUMNS), filters=filters
            )
        except FileNotFoundError:
            return pd.DataFrame(columns=list(EVENT_METRIC_COLUMNS))
        except Exception as exc:
            if is_s3_not_found(exc):
                return pd.DataFrame(columns=list(EVENT_METRIC_COLUMNS))
            try:
                frame = read_parquet(inp.options, "events.parquet")
            except Exception:
                frame = build_input_source(inp).read_events()
        keep = [column for column in EVENT_METRIC_COLUMNS if column in frame.columns]
        frame = frame.loc[:, keep] if keep else frame
        return filter_events_by_types(frame, types)
    if inp.kind == "db" and not inp.options.get("events_query"):
        table = sql_identifier(
            inp.options.get("events_table", DEFAULT_EVENTS_TABLE),
            option="events_table",
        )
        engine = create_engine(require_option(inp.options, "database_url", "db"), pool_pre_ping=True)
        quoted = ", ".join(f'"{column}"' for column in EVENT_METRIC_COLUMNS)
        try:
            if types:
                stmt = text(f'SELECT {quoted} FROM "{table}" WHERE "event_type" IN :types').bindparams(
                    bindparam("types", expanding=True)
                )
                return pd.read_sql(stmt, engine, params={"types": list(types)})
            return pd.read_sql(text(f'SELECT {quoted} FROM "{table}"'), engine)
        except Exception:
            frame = build_input_source(inp).read_events()
            keep = [column for column in EVENT_METRIC_COLUMNS if column in frame.columns]
            frame = frame.loc[:, keep] if keep else frame
            return filter_events_by_types(frame, types)
        finally:
            engine.dispose()
    frame = build_input_source(inp).read_events()
    keep = [column for column in EVENT_METRIC_COLUMNS if column in frame.columns]
    frame = frame.loc[:, keep] if keep else frame
    return filter_events_by_types(frame, types)
