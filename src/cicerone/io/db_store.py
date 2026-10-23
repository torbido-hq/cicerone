"""Database input/output via SQLAlchemy.

Options: database_url (required); optional table names / raw SQL overrides
(events_query, users_query, items_query). Identifiers come from trusted
deploy-time config only.

NOTE: upgrading an existing recommendations table for optional ``reasons``
needs ``ALTER TABLE … ADD COLUMN reasons TEXT``; pandas to_sql(append) will
not add the column. Experiments similarly need ``ALTER TABLE … ADD COLUMN variant TEXT``.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Engine,
    Float,
    Index,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    bindparam,
    delete,
    insert,
    inspect,
    select,
    text,
)

from cicerone.config.constants import DEFAULT_MAX_ARTIFACT_BYTES
from cicerone.io.applied_events import DEFAULT_APPLIED_EVENTS_TABLE, applied_cutoff, chunked
from cicerone.io.db_errors import MISSING_TABLE_ERRORS, is_missing_column_error
from cicerone.io.engines import engine_for, quote_identifier, sql_source
from cicerone.io.options import (
    readonly_select,
    reject_oversize,
    require_max_bytes,
    require_option,
    sql_identifier,
    table_option,
)
from cicerone.io.recommendation_schema import (
    REASONS_COLUMN,
    VARIANT_COLUMN,
    recommendations_sql_names,
)
from cicerone.io.replace_users import RecommendationSchemaError, normalize_replace_user_ids
from cicerone.io.user_lookup import OCCURRED_AT_COLUMN, filter_rows_for_user, newest_events
from cicerone.locks.hold import WriteFence

logger = logging.getLogger(__name__)

_SQL_HISTORY_OVERFETCH = 8


DEFAULT_EVENTS_TABLE = "events"
DEFAULT_USERS_TABLE = "users"
DEFAULT_ITEMS_TABLE = "items"
DEFAULT_RECOMMENDATIONS_TABLE = "recommendations"
DEFAULT_MANIFEST_TABLE = "recommendation_runs"
DEFAULT_MODEL_ARTIFACT_TABLE = "model_artifacts"
# Separate from DEFAULT_ITEMS_TABLE so shared in/out DBs do not clobber source items.
DEFAULT_RECOMMENDATION_ITEMS_TABLE = "recommendation_items"
DEFAULT_EXPOSURES_TABLE = "recommendation_exposures"
DEFAULT_EXPERIMENT_STATE_TABLE = "experiment_state"
DEFAULT_TRACK_TABLE = "recommendation_track"
DEFAULT_EVAL_TABLE = "recommendation_eval"
DEFAULT_HISTORY_TABLE = "recommendation_history"

DEFAULT_DB_TABLES = frozenset(
    {
        DEFAULT_EVENTS_TABLE,
        DEFAULT_USERS_TABLE,
        DEFAULT_ITEMS_TABLE,
        DEFAULT_RECOMMENDATIONS_TABLE,
        DEFAULT_MANIFEST_TABLE,
        DEFAULT_MODEL_ARTIFACT_TABLE,
        DEFAULT_RECOMMENDATION_ITEMS_TABLE,
        DEFAULT_EXPOSURES_TABLE,
        DEFAULT_EXPERIMENT_STATE_TABLE,
        DEFAULT_TRACK_TABLE,
        DEFAULT_EVAL_TABLE,
        DEFAULT_HISTORY_TABLE,
        DEFAULT_APPLIED_EVENTS_TABLE,
    }
)


def _db_manifest_newer(conn, table: str, started_at: str) -> bool:
    if not inspect(conn).has_table(table):
        return False
    runs = Table(table, MetaData(), autoload_with=conn)
    if "generated_at" not in runs.c:
        return False
    columns = [runs.c.generated_at]
    if "last_incremental_at" in runs.c:
        columns.append(runs.c.last_incremental_at)
    row = conn.execute(select(*columns).order_by(runs.c.generated_at.desc()).limit(1)).mappings().first()
    if not row:
        return False
    for key in ("generated_at", "last_incremental_at"):
        value = row.get(key)
        if value is None:
            continue
        stamp = value.isoformat() if hasattr(value, "isoformat") else str(value)
        if stamp > started_at:
            return True
    return False


def _add_missing_manifest_columns(engine: Any, table: str, manifest: dict) -> None:
    # The table takes its columns from the first manifest; an incremental one adds keys a job manifest lacks.
    # Each ALTER runs in its own transaction, before the insert: DDL commits implicitly on MySQL.
    inspector = inspect(engine)
    if not inspector.has_table(table):
        return
    existing = {column["name"] for column in inspector.get_columns(table)}
    for key, value in manifest.items():
        if key in existing:
            continue
        column = sql_identifier(key, option="manifest key")
        column_type = _manifest_column_type(value).compile(dialect=engine.dialect)
        logger.info("Adding column %r to manifest table %r", column, table)
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        f"ALTER TABLE {quote_identifier(engine, table)} "
                        f"ADD COLUMN {quote_identifier(engine, column)} {column_type}"
                    )
                )
        except MISSING_TABLE_ERRORS:
            # A concurrent writer may have added the same column first.
            if column not in {column_info["name"] for column_info in inspect(engine).get_columns(table)}:
                raise


def _manifest_column_type(value: Any) -> Any:
    if isinstance(value, bool):
        return Boolean()
    if isinstance(value, int):
        return BigInteger()
    if isinstance(value, float):
        return Float()
    return Text()


def _clear_table_for_replace(conn, table: str) -> None:
    """Empty ``table`` before a full rewrite; prefer TRUNCATE, fall back to DELETE.

    MySQL commits TRUNCATE implicitly (ending the replace transaction), so it uses DELETE; on
    SQLite TRUNCATE fails and DELETE follows. Rolling back only a failed savepoint keeps the
    rewrite atomic; if the table does not exist yet, every statement fails and ``to_sql`` creates it.
    """
    verbs = ("DELETE FROM",) if conn.dialect.name == "mysql" else ("TRUNCATE TABLE", "DELETE FROM")
    for verb in verbs:
        savepoint = conn.begin_nested()
        try:
            conn.execute(text(f"{verb} {quote_identifier(conn, table)}"))
            savepoint.commit()
            return
        except MISSING_TABLE_ERRORS:
            savepoint.rollback()


def _missing_optional_recommendation_columns(engine: Engine, table: str, frame: pd.DataFrame) -> list[str]:
    wanted = [column for column in (REASONS_COLUMN, VARIANT_COLUMN) if column in frame.columns]
    if not wanted:
        return []
    inspector = inspect(engine)
    if not inspector.has_table(table):
        return []
    existing = {column["name"] for column in inspector.get_columns(table)}
    return [column for column in wanted if column not in existing]


def _require_optional_recommendation_columns(engine: Engine, table: str, frame: pd.DataFrame) -> None:
    missing = _missing_optional_recommendation_columns(engine, table, frame)
    if not missing:
        return
    alters = "; ".join(f"ALTER TABLE … ADD COLUMN {column} TEXT" for column in missing)
    raise RecommendationSchemaError(
        f"Recommendations table {table!r} is missing column(s) {missing}; {alters}"
    )


class DatabaseInputSource:
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._engine = engine_for(require_option(options, "database_url", "db"))

    def _configured_query(self, key: str) -> str | None:
        query = self._options.get(key)
        if query is None:
            return None
        return readonly_select(query, option=f"input.options.{key}")

    def _read(self, query: str | None, table: str) -> pd.DataFrame:
        sql = query or f"SELECT * FROM {quote_identifier(self._engine, table)}"
        logger.info("Reading from database: %s", "configured query" if query else f'table "{table}"')
        return pd.read_sql(text(sql), self._engine)

    def read_events(self) -> pd.DataFrame:
        return self._read(
            self._configured_query("events_query"),
            table_option(self._options, "events_table", DEFAULT_EVENTS_TABLE),
        )

    def _read_optional(self, query: str | None, table: str, label: str) -> pd.DataFrame | None:
        if query is None and not inspect(self._engine).has_table(table):
            logger.warning(
                "Optional %s source (table %r) does not exist — continuing without %s features.",
                label,
                table,
                label,
            )
            return None
        try:
            return self._read(query, table)
        except MISSING_TABLE_ERRORS:
            logger.warning("Optional %s source unavailable — continuing without %s features.", label, label)
            return None

    def read_users(self) -> pd.DataFrame | None:
        return self._read_optional(
            self._configured_query("users_query"),
            table_option(self._options, "users_table", DEFAULT_USERS_TABLE),
            "users",
        )

    def read_items(self) -> pd.DataFrame | None:
        return self._read_optional(
            self._configured_query("items_query"),
            table_option(self._options, "items_table", DEFAULT_ITEMS_TABLE),
            "items",
        )

    def _select_user_rows(
        self,
        query: str | None,
        table: str,
        user_id: str,
        *,
        limit: int | None = None,
        order_occurred_at: bool = False,
    ) -> pd.DataFrame:
        source = sql_source(self._engine, query, table, "_cicerone_user_rows")
        sql = f"SELECT * FROM {source} WHERE {quote_identifier(self._engine, 'user_id')} = :user_id"
        params: dict[str, Any] = {"user_id": user_id}
        if order_occurred_at:
            sql += f" ORDER BY {quote_identifier(self._engine, OCCURRED_AT_COLUMN)} DESC NULLS LAST"
        if limit is not None:
            sql += " LIMIT :limit"
            params["limit"] = int(limit)
        logger.info(
            "Reading from database: user-filtered %s",
            "configured query" if query else f'table "{table}"',
        )
        return pd.read_sql(text(sql), self._engine, params=params)

    def get_events_for_user(self, user_id: str, limit: int) -> pd.DataFrame:
        table = table_option(self._options, "events_table", DEFAULT_EVENTS_TABLE)
        query = self._configured_query("events_query")
        sql_limit = max(int(limit) * _SQL_HISTORY_OVERFETCH, int(limit))
        try:
            frame = self._select_user_rows(query, table, user_id, limit=sql_limit, order_occurred_at=True)
        except MISSING_TABLE_ERRORS:
            frame = self._select_user_rows(query, table, user_id, limit=sql_limit)
        return newest_events(filter_rows_for_user(frame, user_id), limit)

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        query = self._configured_query("users_query")
        table = table_option(self._options, "users_table", DEFAULT_USERS_TABLE)
        if query is None and not inspect(self._engine).has_table(table):
            return None
        try:
            frame = self._select_user_rows(query, table, user_id)
        except MISSING_TABLE_ERRORS:
            return None
        matched = filter_rows_for_user(frame, user_id)
        if matched.empty:
            return None
        return matched.iloc[0].to_dict()


def _artifact_fingerprint(written: datetime | None) -> str | None:
    return None if written is None else f"db:{written.isoformat()}"


class DatabaseOutputSink:
    def __init__(
        self,
        options: dict[str, Any],
        *,
        fence: WriteFence | None = None,
    ):
        self._options = options
        self._engine = engine_for(require_option(options, "database_url", "db"))
        self._fence = fence or WriteFence()
        self._applied_table_seen = False
        self._applied_table_checked = False
        self._check_applied_events_table_name()

    @contextmanager
    def recommendations_write(self) -> Iterator[None]:
        with self._fence.lease():
            yield

    def ensure_writer_held(self) -> None:
        self._ensure_fence()

    def _ensure_fence(self) -> None:
        self._fence.ensure()

    def write_recommendations(self, df: pd.DataFrame) -> None:
        with self.recommendations_write():
            table, _columns, _user_col = recommendations_sql_names(
                self._options, default_table=DEFAULT_RECOMMENDATIONS_TABLE
            )
            logger.info("Writing %d rows to database table %r", len(df), table)
            _require_optional_recommendation_columns(self._engine, table, df)
            # Keys go first, in their own commit: MySQL's TRUNCATE commits implicitly, so a failed insert
            # after it could otherwise roll the key delete back while the rows stay gone.
            with self._engine.begin() as conn:
                self._ensure_fence()
                self._clear_applied_keys(conn, table)
            with self._engine.begin() as conn:
                self._ensure_fence()
                _clear_table_for_replace(conn, table)
                self._ensure_fence()
                df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=1000)
                self._ensure_fence()

    def replace_recommendations_for_users(
        self, df: pd.DataFrame, *, user_ids: Sequence[str], applied_keys: Sequence[str] = ()
    ) -> int:
        ids = normalize_replace_user_ids(df, user_ids)
        if not ids:
            return 0
        with self.recommendations_write():
            return self._replace_recommendations_for_users_unlocked(df, ids, applied_keys)

    def _applied_events_table(self) -> Table:
        name = table_option(self._options, "applied_events_table", DEFAULT_APPLIED_EVENTS_TABLE)
        # Keys are scoped to the recommendations table (see _applied_scope), so outputs sharing a database
        # don't see each other's.
        return Table(
            name,
            MetaData(),
            # 64 chars: a SQL identifier and a sha256 hex key; the composite key fits a 767-byte index.
            Column("recommendations_table", String(64), primary_key=True),
            Column("event_key", String(64), primary_key=True),
            Column("applied_at", String(64), nullable=False),
            # Hashed so a long table name still gives an index name within MySQL's 64 characters.
            Index(f"ix_applied_at_{hashlib.sha256(name.encode()).hexdigest()[:16]}", "applied_at"),
        )

    def _create_applied_events_table(self) -> None:
        table = self._applied_events_table()
        # CREATE TABLE commits implicitly on MySQL, so never inside the rows' transaction.
        try:
            with self._engine.begin() as ddl:
                table.create(ddl, checkfirst=True)
        except MISSING_TABLE_ERRORS:
            # Another output sharing the database may have created it first.
            if not inspect(self._engine).has_table(table.name):
                raise
        self._applied_table_seen = True

    def _check_applied_events_table_name(self) -> None:
        name = self._applied_events_table().name
        others = {
            str(value)
            for key, value in self._options.items()
            if key.endswith("_table") and key != "applied_events_table"
        }
        others |= DEFAULT_DB_TABLES - {DEFAULT_APPLIED_EVENTS_TABLE}
        if name.lower() in {other.lower() for other in others}:
            raise ValueError(f"applied_events_table {name!r} is already used by another output table")

    def _recommendations_table_name(self) -> str:
        table, _columns, _user_col = recommendations_sql_names(
            self._options, default_table=DEFAULT_RECOMMENDATIONS_TABLE
        )
        return table

    def _applied_scope(self, recommendations_table: str) -> str:
        # Unquoted names ignore case on SQLite and Postgres; MySQL table-name case depends on the server.
        name = (
            recommendations_table if self._engine.dialect.name == "mysql" else recommendations_table.lower()
        )
        # Hashed so a case-insensitive collation cannot merge two case-distinct table names.
        return hashlib.sha256(name.encode()).hexdigest()

    def applied_event_keys(self, keys: Sequence[str]) -> set[str]:
        wanted = sorted({str(key) for key in keys})
        table = self._applied_events_table()
        if not wanted:
            return set()
        if not self._applied_table_seen:
            if not inspect(self._engine).has_table(table.name):
                return set()
            self._applied_table_seen = True
        scope = self._applied_scope(self._recommendations_table_name())
        cutoff = applied_cutoff(datetime.now(UTC))
        found: dict[str, str] = {}
        try:
            with self._engine.connect() as conn:
                for part in chunked(wanted):
                    query = select(table.c.event_key, table.c.applied_at).where(
                        table.c.recommendations_table == scope,
                        table.c.event_key.in_(part),
                        table.c.applied_at >= cutoff,
                    )
                    found.update((str(key), str(applied_at)) for key, applied_at in conn.execute(query))
        except MISSING_TABLE_ERRORS:
            # The table may have been dropped; check again on the next lookup.
            self._applied_table_seen = False
            raise
        if not found:
            return set()
        # A full write by a version without the key table leaves keys behind; its manifest supersedes them.
        # Checked only when keys match, which is rare (a redelivery), so most flushes skip the manifest query.
        full_write_at = self._latest_full_write_at()
        if full_write_at is None:
            return set(found)
        return {key for key, applied_at in found.items() if applied_at >= full_write_at}

    def _latest_full_write_at(self) -> str | None:
        table = table_option(self._options, "manifest_table", DEFAULT_MANIFEST_TABLE)
        generated_at = quote_identifier(self._engine, "generated_at")
        triggered_by = quote_identifier(self._engine, "triggered_by")
        query = text(
            f"SELECT MAX({generated_at}) FROM {quote_identifier(self._engine, table)} "
            f"WHERE {triggered_by} IS NULL OR {triggered_by} <> 'incremental'"
        )
        try:
            with self._engine.connect() as conn:
                value = conn.execute(query).scalar()
        except MISSING_TABLE_ERRORS:
            # No manifest table (or no such columns) yet: no full write to supersede the keys.
            return None
        if value is None:
            return None
        return value.isoformat() if hasattr(value, "isoformat") else str(value)

    def _applied_events_table_exists(self) -> bool:
        if not self._applied_table_seen and not self._applied_table_checked:
            self._applied_table_checked = True
            self._applied_table_seen = inspect(self._engine).has_table(self._applied_events_table().name)
        return self._applied_table_seen

    def _prune_applied_keys(self, conn: Any, now: datetime) -> None:
        table = self._applied_events_table()
        conn.execute(delete(table).where(table.c.applied_at < applied_cutoff(now)))

    def _record_applied_keys(self, conn: Any, keys: Sequence[str]) -> None:
        wanted = sorted({str(key) for key in keys})
        now = datetime.now(UTC)
        table = self._applied_events_table()
        scope = self._applied_scope(self._recommendations_table_name())
        self._prune_applied_keys(conn, now)
        for part in chunked(wanted):
            conn.execute(
                delete(table).where(table.c.recommendations_table == scope, table.c.event_key.in_(part))
            )
        stamp = now.isoformat()
        # executemany: SQLAlchemy batches these within the dialect's parameter limit.
        conn.execute(
            insert(table),
            [{"recommendations_table": scope, "event_key": key, "applied_at": stamp} for key in wanted],
        )

    def _clear_applied_keys(self, conn: Any, recommendations_table: str) -> None:
        # A full write replaces every row, so earlier incremental keys no longer apply (as on dataset output).
        table = self._applied_events_table()
        if inspect(conn).has_table(table.name):
            scope = self._applied_scope(recommendations_table)
            conn.execute(delete(table).where(table.c.recommendations_table == scope))

    def _replace_recommendations_for_users_unlocked(
        self, df: pd.DataFrame, ids: list[str], applied_keys: Sequence[str] = ()
    ) -> int:
        table, _columns, user_col = recommendations_sql_names(
            self._options, default_table=DEFAULT_RECOMMENDATIONS_TABLE
        )
        logger.info(
            "Replacing recommendations for %d user(s) (%d row(s)) in %r",
            len(ids),
            len(df),
            table,
        )
        _require_optional_recommendation_columns(self._engine, table, df)
        # Quoted like to_sql creates the table, so an uppercase name hits the same table.
        quoted_table = quote_identifier(self._engine, table)
        quoted_user = quote_identifier(self._engine, user_col)
        delete_sql = text(f"DELETE FROM {quoted_table} WHERE {quoted_user} IN :user_ids").bindparams(
            bindparam("user_ids", expanding=True)
        )
        count_sql = text(f"SELECT COUNT(DISTINCT {quoted_user}) FROM {quoted_table}")
        if applied_keys:
            self._create_applied_events_table()
        with self._engine.begin() as conn:
            self._ensure_fence()
            savepoint = conn.begin_nested()
            try:
                conn.execute(delete_sql, {"user_ids": ids})
                savepoint.commit()
            except MISSING_TABLE_ERRORS as exc:
                savepoint.rollback()
                logger.warning(
                    "Recommendations delete skipped for table %r (missing table or schema issue): %s",
                    table,
                    exc,
                )
                # Schema drift (e.g. missing user_id): do not append into a broken table.
                if is_missing_column_error(exc):
                    raise RecommendationSchemaError(
                        f"Recommendations schema mismatch for table {table!r}; refusing replace"
                    ) from exc
            if not df.empty:
                self._ensure_fence()
                df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=1000)
            if applied_keys:
                # Same transaction as the rows: a replay after a crash finds the keys only if the rows landed.
                self._ensure_fence()
                self._record_applied_keys(conn, applied_keys)
            elif self._applied_events_table_exists():
                # Batches without durable keys still prune, so expired rows never pile up.
                self._ensure_fence()
                prune = conn.begin_nested()
                try:
                    self._prune_applied_keys(conn, datetime.now(UTC))
                    prune.commit()
                except MISSING_TABLE_ERRORS:
                    # The table was dropped: skip pruning, never the rows write; check again later.
                    prune.rollback()
                    self._applied_table_seen = self._applied_table_checked = False
            self._ensure_fence()
            count_savepoint = conn.begin_nested()
            try:
                value = conn.execute(count_sql).scalar()
                count_savepoint.commit()
            except MISSING_TABLE_ERRORS as exc:
                count_savepoint.rollback()
                logger.warning(
                    "Recommendations count skipped for table %r (missing table or schema issue): %s",
                    table,
                    exc,
                )
                return 0
        return int(value or 0)

    def write_manifest(self, manifest: dict, *, skip_if_newer_than: str | None = None) -> bool:
        table = table_option(self._options, "manifest_table", DEFAULT_MANIFEST_TABLE)
        logger.info("Appending run manifest to database table %r", table)
        with self.recommendations_write():
            self._ensure_fence()
            _add_missing_manifest_columns(self._engine, table, manifest)
            with self._engine.begin() as conn:
                if skip_if_newer_than is not None and _db_manifest_newer(conn, table, skip_if_newer_than):
                    return False
                self._ensure_fence()
                pd.DataFrame([manifest]).to_sql(table, conn, if_exists="append", index=False)
                self._ensure_fence()
        return True

    def write_model_artifact(self, payload: bytes) -> None:
        """Replace the single-row model_artifacts table with the latest blob."""
        artifacts = self._model_artifacts_table()
        logger.info("Writing model artifact (%d bytes) to database table %r", len(payload), artifacts.name)
        with self.recommendations_write(), self._engine.begin() as conn:
            self._ensure_fence()
            artifacts.create(conn, checkfirst=True)
            self._ensure_fence()
            conn.execute(artifacts.delete())
            self._ensure_fence()
            conn.execute(insert(artifacts).values(payload=payload, written_at=datetime.now(UTC)))
            self._ensure_fence()

    def _model_artifacts_table(self) -> Table:
        return Table(
            table_option(self._options, "model_artifact_table", DEFAULT_MODEL_ARTIFACT_TABLE),
            MetaData(),
            Column("payload", LargeBinary, nullable=False),
            Column("written_at", DateTime(timezone=True), nullable=False),
        )

    def replace_model_artifact_if(self, payload: bytes, expected_fingerprint: str) -> bool:
        artifacts = self._model_artifacts_table()
        with self.recommendations_write(), self._engine.begin() as conn:
            self._ensure_fence()
            artifacts.create(conn, checkfirst=True)
            row = conn.execute(select(artifacts.c.written_at).limit(1).with_for_update()).first()
            if row is None or _artifact_fingerprint(row[0]) != expected_fingerprint:
                return False
            written = row[0]
            self._ensure_fence()
            deleted = conn.execute(artifacts.delete().where(artifacts.c.written_at == written))
            if deleted.rowcount < 1:
                return False
            self._ensure_fence()
            conn.execute(insert(artifacts).values(payload=payload, written_at=datetime.now(UTC)))
            self._ensure_fence()
        return True

    def read_model_artifact(self, *, max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES) -> bytes | None:
        require_max_bytes(max_bytes)
        table_name = table_option(self._options, "model_artifact_table", DEFAULT_MODEL_ARTIFACT_TABLE)
        if not inspect(self._engine).has_table(table_name):
            return None
        quoted_table, payload = (
            quote_identifier(self._engine, table_name),
            quote_identifier(self._engine, "payload"),
        )
        if self._engine.dialect.name == "sqlite":
            sql = f"SELECT substr({payload}, 1, :cap), length({payload}) FROM {quoted_table} LIMIT 1"
        else:
            sql = (
                f"SELECT substring({payload} from 1 for :cap), octet_length({payload}) "
                f"FROM {quoted_table} LIMIT 1"
            )
        with self._engine.connect() as conn:
            row = conn.execute(text(sql), {"cap": max_bytes + 1}).first()
        if row is None or row[1] is None:
            return None
        reject_oversize(int(row[1]), max_bytes)
        if row[0] is None:
            return None
        return bytes(row[0])

    def model_artifact_fingerprint(self) -> str | None:
        artifacts = self._model_artifacts_table()
        if not inspect(self._engine).has_table(artifacts.name):
            return None
        with self._engine.connect() as conn:
            row = conn.execute(select(artifacts.c.written_at).limit(1)).first()
        return None if row is None else _artifact_fingerprint(row[0])

    def write_items_snapshot(self, df: pd.DataFrame) -> None:
        table = table_option(self._options, "recommendation_items_table", DEFAULT_RECOMMENDATION_ITEMS_TABLE)
        logger.info("Writing %d item snapshot rows to database table %r", len(df), table)
        with self.recommendations_write(), self._engine.begin() as conn:
            self._ensure_fence()
            _clear_table_for_replace(conn, table)
            self._ensure_fence()
            df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=1000)
            self._ensure_fence()
