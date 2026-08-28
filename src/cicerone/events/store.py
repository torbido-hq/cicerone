"""Load existing recommendation rows for incremental merge."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Collection, Sequence

import pandas as pd
from sqlalchemy import Engine, bindparam, create_engine, text

from cicerone.config import IOSettings
from cicerone.io.db_errors import is_missing_column_error, is_missing_table_error
from cicerone.io.db_store import (
    DEFAULT_RECOMMENDATION_ITEMS_TABLE,
    DEFAULT_RECOMMENDATIONS_TABLE,
    MISSING_TABLE_ERRORS,
)
from cicerone.io.options import is_s3_not_found, read_parquet, require_option, sql_identifier
from cicerone.io.recommendation_reader import ITEMS_SNAPSHOT_FILENAME
from cicerone.io.recommendation_schema import (
    ITEM_COLUMN,
    RECOMMENDATION_COLUMNS,
    SOURCE_COLUMN,
    USER_COLUMN,
    VARIANT_COLUMN,
    recommendation_output_columns,
    recommendations_sql_names,
)

logger = logging.getLogger(__name__)

GUARDRAIL_COLUMNS: tuple[str, ...] = (USER_COLUMN, ITEM_COLUMN, SOURCE_COLUMN, VARIANT_COLUMN)

_MAX_CACHED_ENGINES = 8
_engines: OrderedDict[str, Engine] = OrderedDict()
_engines_lock = threading.Lock()


def empty_recommendations_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(RECOMMENDATION_COLUMNS))


def dispose_recommendation_engines() -> None:
    """Dispose cached SQLAlchemy engines (tests / process shutdown)."""
    with _engines_lock:
        engines = list(_engines.values())
        _engines.clear()
    for engine in engines:
        engine.dispose()


def _engine_for(database_url: str) -> Engine:
    with _engines_lock:
        engine = _engines.get(database_url)
        if engine is not None:
            _engines.move_to_end(database_url)
            return engine
        engine = create_engine(database_url, pool_pre_ping=True)
        _engines[database_url] = engine
        while len(_engines) > _MAX_CACHED_ENGINES:
            _url, old = _engines.popitem(last=False)
            old.dispose()
        return engine


def _normalize_recommendation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in RECOMMENDATION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"recommendations frame missing columns: {missing}")
    return frame.loc[:, recommendation_output_columns(frame)]


def _empty_on_schema_mismatch(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        return _normalize_recommendation_columns(frame)
    except ValueError as exc:
        logger.warning("Recommendations schema mismatch; treating as empty: %s", exc)
        return empty_recommendations_frame()


def _read_parquet_columns(output: IOSettings, filename: str, columns: Sequence[str]) -> pd.DataFrame:
    try:
        return read_parquet(output.options, filename, columns=list(columns))
    except FileNotFoundError:
        raise
    except Exception as exc:
        if is_s3_not_found(exc):
            raise
        return read_parquet(output.options, filename)


def _project_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    keep = [column for column in columns if column in frame.columns]
    return frame.loc[:, keep] if keep else frame.iloc[0:0].copy()


def _db_table_and_columns(output: IOSettings) -> tuple[str, str, str]:
    return recommendations_sql_names(output.options, default_table=DEFAULT_RECOMMENDATIONS_TABLE)


def load_items_catalog_size(output: IOSettings) -> int | None:
    """Distinct items in the serve snapshot, or ``None`` when it is missing."""
    if output.kind == "dataset":
        try:
            frame = _read_parquet_columns(output, ITEMS_SNAPSHOT_FILENAME, (ITEM_COLUMN,))
        except FileNotFoundError:
            return None
        except Exception as exc:
            if is_s3_not_found(exc):
                return None
            logger.exception("Failed to read items snapshot for experiment catalog size")
            return None
        if frame.empty or ITEM_COLUMN not in frame.columns:
            return None
        return int(frame[ITEM_COLUMN].astype(str).nunique())
    if output.kind == "db":
        table = sql_identifier(
            output.options.get("recommendation_items_table", DEFAULT_RECOMMENDATION_ITEMS_TABLE),
            option="recommendation_items_table",
        )
        engine = _engine_for(require_option(output.options, "database_url", "db"))
        try:
            with engine.connect() as conn:
                value = conn.execute(text(f'SELECT COUNT(DISTINCT "{ITEM_COLUMN}") FROM "{table}"')).scalar()
        except MISSING_TABLE_ERRORS as exc:
            if is_missing_table_error(exc) or is_missing_column_error(exc):
                return None
            raise
        except Exception as exc:
            if is_missing_table_error(exc) or is_missing_column_error(exc):
                return None
            logger.exception("Failed to count items snapshot for experiment catalog size")
            return None
        return int(value or 0)
    return None


def load_recommendation_guardrail_rows(output: IOSettings) -> pd.DataFrame | None:
    """Project source/item/variant columns for catalog guardrails."""
    if output.kind == "dataset":
        try:
            frame = _read_parquet_columns(output, "recommendations.parquet", GUARDRAIL_COLUMNS)
        except FileNotFoundError:
            return pd.DataFrame(columns=list(GUARDRAIL_COLUMNS))
        except Exception as exc:
            if is_s3_not_found(exc):
                return pd.DataFrame(columns=list(GUARDRAIL_COLUMNS))
            raise
        if frame.empty:
            return pd.DataFrame(columns=list(GUARDRAIL_COLUMNS))
        return _project_columns(frame, GUARDRAIL_COLUMNS)
    if output.kind == "db":
        table, _required, _user = _db_table_and_columns(output)
        engine = _engine_for(require_option(output.options, "database_url", "db"))
        column_sets = (GUARDRAIL_COLUMNS, (USER_COLUMN, ITEM_COLUMN, SOURCE_COLUMN))
        loaded: pd.DataFrame | None = None
        last_exc: BaseException | None = None
        for columns in column_sets:
            quoted = ", ".join(f'"{column}"' for column in columns)
            try:
                loaded = pd.read_sql_query(text(f"SELECT {quoted} FROM {table}"), engine)
                break
            except MISSING_TABLE_ERRORS as exc:
                last_exc = exc
                mapped = _empty_frame_from_db_error(exc, table=table)
                if mapped is not None:
                    return mapped
            except Exception as exc:
                last_exc = exc
                if is_missing_column_error(exc):
                    continue
                raise
        if loaded is None:
            if last_exc is not None:
                logger.warning("Recommendations schema mismatch; treating as empty: %s", last_exc)
            return empty_recommendations_frame()
        if loaded.empty:
            return pd.DataFrame(columns=list(GUARDRAIL_COLUMNS))
        return _project_columns(loaded, GUARDRAIL_COLUMNS)
    raise ValueError(f"Unsupported output kind for incremental load: {output.kind!r}")


def _empty_frame_from_db_error(exc: BaseException, *, table: str) -> pd.DataFrame | None:
    """Map missing-table/column DB errors to an empty frame; ``None`` means re-raise."""
    if is_missing_column_error(exc):
        logger.warning("Recommendations schema mismatch; treating as empty: %s", exc)
        return empty_recommendations_frame()
    if is_missing_table_error(exc):
        logger.warning("Recommendations table %r missing; treating as empty", table)
        return empty_recommendations_frame()
    return None


def _zero_from_db_error(exc: BaseException, *, table: str) -> int | None:
    """Map missing-table/column DB errors to ``0`` users; ``None`` means re-raise."""
    if is_missing_column_error(exc):
        logger.warning("Recommendations schema mismatch while counting users; treating as empty: %s", exc)
        return 0
    if is_missing_table_error(exc):
        logger.warning("Recommendations table %r missing while counting users; treating as empty", table)
        return 0
    return None


def _load_dataset_recommendations(output: IOSettings) -> pd.DataFrame:
    try:
        frame = read_parquet(output.options, "recommendations.parquet")
    except FileNotFoundError:
        return empty_recommendations_frame()
    except Exception as exc:
        if is_s3_not_found(exc):
            return empty_recommendations_frame()
        raise
    if frame.empty:
        return empty_recommendations_frame()
    return _empty_on_schema_mismatch(frame)


def _load_dataset_recommendations_for_users(output: IOSettings, user_ids: list[str]) -> pd.DataFrame:
    """Load only rows for ``user_ids`` when the parquet engine can push down filters."""
    try:
        frame = read_parquet(
            output.options,
            "recommendations.parquet",
            filters=[(USER_COLUMN, "in", user_ids)],
        )
    except FileNotFoundError:
        return empty_recommendations_frame()
    except Exception as exc:
        if is_s3_not_found(exc):
            return empty_recommendations_frame()
        message = str(exc).lower()
        if USER_COLUMN in message or "fieldref" in message or "filter" in message:
            logger.warning("Filtered recommendations read failed; falling back to full-file load: %s", exc)
            frame = _load_dataset_recommendations(output)
            if frame.empty:
                return frame
            return frame.loc[frame[USER_COLUMN].astype(str).isin(user_ids)].reset_index(drop=True)
        raise
    if frame.empty:
        return empty_recommendations_frame()
    normalized = _empty_on_schema_mismatch(frame)
    if normalized.empty:
        return normalized
    # Defensive: keep only requested ids if the engine ignored/partial-applied filters.
    return normalized.loc[normalized[USER_COLUMN].astype(str).isin(user_ids)].reset_index(drop=True)


def _load_db_recommendations(output: IOSettings, *, user_ids: Collection[str] | None = None) -> pd.DataFrame:
    table, _required_columns, user_col = _db_table_and_columns(output)
    engine = _engine_for(require_option(output.options, "database_url", "db"))
    try:
        if user_ids is None:
            frame = pd.read_sql_query(text(f"SELECT * FROM {table}"), engine)
        else:
            ids = sorted({str(user_id) for user_id in user_ids})
            if not ids:
                return empty_recommendations_frame()
            stmt = text(f"SELECT * FROM {table} WHERE {user_col} IN :user_ids").bindparams(
                bindparam("user_ids", expanding=True)
            )
            frame = pd.read_sql_query(stmt, engine, params={"user_ids": ids})
    except MISSING_TABLE_ERRORS as exc:
        empty = _empty_frame_from_db_error(exc, table=table)
        if empty is not None:
            return empty
        raise
    if frame.empty:
        return empty_recommendations_frame()
    return _empty_on_schema_mismatch(frame)


def load_recommendations_frame(output: IOSettings) -> pd.DataFrame:
    if output.kind == "dataset":
        return _load_dataset_recommendations(output)

    if output.kind == "db":
        return _load_db_recommendations(output)

    raise ValueError(f"Unsupported output kind for incremental load: {output.kind!r}")


def load_recommendations_for_users(output: IOSettings, user_ids: Collection[str]) -> pd.DataFrame:
    ids = sorted({str(user_id) for user_id in user_ids})
    if not ids:
        return empty_recommendations_frame()

    if output.kind == "dataset":
        return _load_dataset_recommendations_for_users(output, ids)

    if output.kind == "db":
        return _load_db_recommendations(output, user_ids=ids)

    raise ValueError(f"Unsupported output kind for incremental load: {output.kind!r}")


def count_recommendation_users(output: IOSettings) -> int:
    if output.kind == "dataset":
        try:
            # Project only user_id — avoids materializing the full recommendations frame.
            frame = read_parquet(output.options, "recommendations.parquet", columns=[USER_COLUMN])
        except FileNotFoundError:
            return 0
        except Exception as exc:
            if is_s3_not_found(exc):
                return 0
            message = str(exc).lower()
            if USER_COLUMN in message or "fieldref" in message:
                logger.warning(
                    "Recommendations schema mismatch while counting users; treating as empty: %s", exc
                )
                return 0
            raise
        if frame.empty:
            return 0
        return int(frame[USER_COLUMN].astype(str).nunique())

    if output.kind == "db":
        table, _columns, user_col = _db_table_and_columns(output)
        engine = _engine_for(require_option(output.options, "database_url", "db"))
        try:
            with engine.connect() as conn:
                value = conn.execute(text(f"SELECT COUNT(DISTINCT {user_col}) FROM {table}")).scalar()
        except MISSING_TABLE_ERRORS as exc:
            zero = _zero_from_db_error(exc, table=table)
            if zero is not None:
                return zero
            raise
        return int(value or 0)

    raise ValueError(f"Unsupported output kind for incremental load: {output.kind!r}")
