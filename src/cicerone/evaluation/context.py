"""Load events and recommendation snapshots used by track eval."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from sqlalchemy import bindparam, create_engine, text

from cicerone.config.settings import Settings
from cicerone.evaluation.tracking import conversion_events, filter_events_by_types
from cicerone.io.db_store import DEFAULT_EVENTS_TABLE
from cicerone.io.factory import build_input_source
from cicerone.io.options import (
    is_s3_not_found,
    read_parquet,
    readonly_select,
    require_option,
    sql_identifier,
)
from cicerone.io.recommendation_schema import USER_COLUMN
from cicerone.track.store_common import since_date_floor

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


def _filter_events_since(frame: pd.DataFrame, since: str | None) -> pd.DataFrame:
    if not since or frame.empty:
        return frame
    if "occurred_at" not in frame.columns:
        return frame.iloc[0:0]
    start = pd.to_datetime(since, utc=True, errors="coerce")
    if pd.isna(start):
        return frame.iloc[0:0]
    stamps = pd.to_datetime(frame["occurred_at"], utc=True, errors="coerce")
    return frame.loc[stamps.notna() & (stamps >= start)].copy()


def _metric_event_sql(
    source: str,
    *,
    types: tuple[str, ...] | None,
    floor: str | None,
) -> tuple[Any, dict[str, Any]]:
    quoted = ", ".join(f'"{column}"' for column in EVENT_METRIC_COLUMNS)
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if types:
        clauses.append('"event_type" IN :types')
        params["types"] = list(types)
    if floor is not None:
        clauses.append('"occurred_at" >= :since')
        params["since"] = floor
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    stmt = text(f"SELECT {quoted} FROM {source}{where}")
    if types:
        stmt = stmt.bindparams(bindparam("types", expanding=True))
    return stmt, params


_QUOTED_SQL = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_SQL_PAGE = re.compile(r"\b(?:limit|offset)\b", re.IGNORECASE)


def _sql_has_page(sql: str) -> bool:
    stripped = _QUOTED_SQL.sub(" ", sql)
    while True:
        nxt = re.sub(r"\([^()]*\)", " ", stripped)
        if nxt == stripped:
            break
        stripped = nxt
    return _SQL_PAGE.search(stripped) is not None


def _parquet_since_bounds(floor: str) -> tuple[Any, str]:
    start = pd.to_datetime(floor, utc=True, errors="coerce")
    if pd.isna(start):
        return floor, floor
    return start.to_pydatetime(), floor


def _read_metric_parquet(
    options: dict[str, Any],
    *,
    types: tuple[str, ...] | None,
    floor: str | None,
) -> pd.DataFrame:
    base: list[Any] = []
    if types:
        base.append(("event_type", "in", list(types)))
    attempts: list[list[Any] | None] = []
    if floor is not None:
        stamp, text_floor = _parquet_since_bounds(floor)
        attempts.append([*base, ("occurred_at", ">=", stamp)])
        if text_floor != stamp:
            attempts.append([*base, ("occurred_at", ">=", text_floor)])
    else:
        attempts.append(base or None)
    last_error: Exception | None = None
    for filters in attempts:
        try:
            return read_parquet(
                options,
                "events.parquet",
                columns=list(EVENT_METRIC_COLUMNS),
                filters=filters,
            )
        except FileNotFoundError:
            raise
        except Exception as exc:
            if is_s3_not_found(exc):
                raise
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.DataFrame(columns=list(EVENT_METRIC_COLUMNS))


def load_metric_events(
    settings: Settings, *, event_types: Sequence[str] | None = None, since: str | None = None
) -> pd.DataFrame:
    inp = settings.input
    types = tuple(event_types) if event_types else None
    floor = since_date_floor(since) if since else None
    if since and floor is None:
        return pd.DataFrame(columns=list(EVENT_METRIC_COLUMNS))
    if inp.kind == "dataset":
        try:
            frame = _read_metric_parquet(inp.options, types=types, floor=floor)
        except FileNotFoundError:
            return pd.DataFrame(columns=list(EVENT_METRIC_COLUMNS))
        except Exception:
            if floor is not None:
                return pd.DataFrame(columns=list(EVENT_METRIC_COLUMNS))
            try:
                frame = read_parquet(inp.options, "events.parquet")
            except Exception:
                frame = build_input_source(inp).read_events()
        keep = [column for column in EVENT_METRIC_COLUMNS if column in frame.columns]
        frame = frame.loc[:, keep] if keep else frame
        return _filter_events_since(filter_events_by_types(frame, types), since)
    if inp.kind == "db":
        engine = create_engine(require_option(inp.options, "database_url", "db"), pool_pre_ping=True)
        query = inp.options.get("events_query")
        try:
            if query:
                cleaned = readonly_select(str(query), option="input.options.events_query")
                if _sql_has_page(cleaned):
                    frame = pd.read_sql(text(cleaned), engine)
                    keep = [column for column in EVENT_METRIC_COLUMNS if column in frame.columns]
                    frame = frame.loc[:, keep] if keep else frame
                    return _filter_events_since(filter_events_by_types(frame, types), since)
                source = f"({cleaned}) AS _cicerone_metric_events"
            else:
                table = sql_identifier(
                    inp.options.get("events_table", DEFAULT_EVENTS_TABLE),
                    option="events_table",
                )
                source = f'"{table}"'
            stmt, params = _metric_event_sql(source, types=types, floor=floor)
            frame = pd.read_sql(stmt, engine, params=params)
            return _filter_events_since(frame, since)
        except Exception:
            if floor is not None:
                return pd.DataFrame(columns=list(EVENT_METRIC_COLUMNS))
            frame = build_input_source(inp).read_events()
            keep = [column for column in EVENT_METRIC_COLUMNS if column in frame.columns]
            frame = frame.loc[:, keep] if keep else frame
            return _filter_events_since(filter_events_by_types(frame, types), since)
        finally:
            engine.dispose()
    frame = build_input_source(inp).read_events()
    keep = [column for column in EVENT_METRIC_COLUMNS if column in frame.columns]
    frame = frame.loc[:, keep] if keep else frame
    return _filter_events_since(filter_events_by_types(frame, types), since)
