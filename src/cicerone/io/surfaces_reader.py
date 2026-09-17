"""Read popular / latest / item-neighbor snapshots written by the job."""

from __future__ import annotations

import io
import json
import logging
import threading
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from cicerone.blending import LATEST_SOURCE, POPULAR_SOURCE
from cicerone.io.db_store import (
    DEFAULT_LATEST_TABLE,
    DEFAULT_NEIGHBORS_TABLE,
    DEFAULT_POPULAR_TABLE,
    MISSING_TABLE_ERRORS,
)
from cicerone.io.options import (
    build_s3_client,
    is_s3_not_found,
    object_key,
    read_parquet,
    require_option,
    sql_identifier,
    validate_storage_options,
)
from cicerone.io.recommendation_schema import ITEM_COLUMN, RANK_COLUMN, SCORE_COLUMN, SOURCE_COLUMN
from cicerone.io.surfaces import (
    LATEST_FILENAME,
    NEIGHBOR_ITEM_COLUMN,
    NEIGHBORS_FILENAME,
    POPULAR_FILENAME,
    SURFACE_FILES,
    SURFACES_STAMP_FILENAME,
    empty_neighbors_frame,
    empty_surface_frame,
    surfaces_stamp_sha256,
)

logger = logging.getLogger(__name__)


class SurfacesReader:
    def get_popular(self, k: int) -> pd.DataFrame:
        raise NotImplementedError

    def get_latest(self, k: int) -> pd.DataFrame:
        raise NotImplementedError

    def get_similar(self, item_id: str, k: int) -> pd.DataFrame:
        raise NotImplementedError

    def refresh(self) -> None:
        return


class EmptySurfacesReader(SurfacesReader):
    def get_popular(self, k: int) -> pd.DataFrame:
        del k
        return empty_surface_frame(source=POPULAR_SOURCE)

    def get_latest(self, k: int) -> pd.DataFrame:
        del k
        return empty_surface_frame(source=LATEST_SOURCE)

    def get_similar(self, item_id: str, k: int) -> pd.DataFrame:
        del item_id, k
        return empty_neighbors_frame()


class DatasetSurfacesReader(SurfacesReader):
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._backend = validate_storage_options(options)
        self._lock = threading.Lock()
        self._popular = empty_surface_frame(source=POPULAR_SOURCE)
        self._latest = empty_surface_frame(source=LATEST_SOURCE)
        self._neighbors = empty_neighbors_frame()
        self._have_stamp = False
        self._s3_client = None
        self.refresh()

    def _get_s3_client(self):
        if self._s3_client is None:
            self._s3_client = build_s3_client(self._options)
        return self._s3_client

    def _read_bytes(self, filename: str) -> bytes | None:
        try:
            if self._backend == "local":
                path = Path(require_option(self._options, "path", "local")) / filename
                return path.read_bytes()
            bucket = require_option(self._options, "bucket", "s3")
            key = object_key(self._options, filename)
            response = self._get_s3_client().get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            try:
                return body.read()
            finally:
                body.close()
        except FileNotFoundError:
            return None
        except Exception as exc:
            if is_s3_not_found(exc):
                return None
            logger.exception("Failed to load surface file %s", filename)
            return None

    def _read(self, filename: str) -> pd.DataFrame | None:
        try:
            if self._backend == "local":
                path = Path(require_option(self._options, "path", "local")) / filename
                if not path.exists():
                    return None
            return read_parquet(
                self._options,
                filename,
                s3_client=self._get_s3_client() if self._backend == "s3" else None,
            )
        except FileNotFoundError:
            return None
        except Exception as exc:
            if is_s3_not_found(exc):
                return None
            logger.exception("Failed to load surface file %s", filename)
            return None

    def refresh(self) -> None:
        stamp_bytes = self._read_bytes(SURFACES_STAMP_FILENAME)
        if stamp_bytes is not None:
            files: list[tuple[str, bytes]] = []
            for filename in SURFACE_FILES:
                payload = self._read_bytes(filename)
                if payload is None:
                    return
                files.append((filename, payload))
            try:
                expected = json.loads(stamp_bytes.decode()).get("sha256")
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                return
            if expected != surfaces_stamp_sha256(files):
                return
            try:
                popular = pd.read_parquet(io.BytesIO(files[0][1]))
                latest = pd.read_parquet(io.BytesIO(files[1][1]))
                neighbors = pd.read_parquet(io.BytesIO(files[2][1]))
            except Exception:
                logger.exception("Failed to parse stamped surface snapshots")
                return
            with self._lock:
                self._popular = popular
                self._latest = latest
                self._neighbors = neighbors
                self._have_stamp = True
            return
        if self._have_stamp:
            return
        popular = self._read(POPULAR_FILENAME)
        latest = self._read(LATEST_FILENAME)
        neighbors = self._read(NEIGHBORS_FILENAME)
        if popular is None or latest is None or neighbors is None:
            return
        with self._lock:
            self._popular = popular
            self._latest = latest
            self._neighbors = neighbors

    def get_popular(self, k: int) -> pd.DataFrame:
        with self._lock:
            return self._popular.head(k).reset_index(drop=True)

    def get_latest(self, k: int) -> pd.DataFrame:
        with self._lock:
            return self._latest.head(k).reset_index(drop=True)

    def get_similar(self, item_id: str, k: int) -> pd.DataFrame:
        with self._lock:
            frame = self._neighbors
            if frame.empty or ITEM_COLUMN not in frame.columns:
                return empty_neighbors_frame()
            rows = frame.loc[frame[ITEM_COLUMN].astype(str) == str(item_id)]
            if RANK_COLUMN in rows.columns:
                rows = rows.sort_values(RANK_COLUMN, kind="mergesort")
            return rows.head(k).reset_index(drop=True)


class DbSurfacesReader(SurfacesReader):
    def __init__(self, options: dict[str, Any]):
        self._options = options
        self._popular_table = sql_identifier(
            options.get("popular_table", DEFAULT_POPULAR_TABLE), option="popular_table"
        )
        self._latest_table = sql_identifier(
            options.get("latest_table", DEFAULT_LATEST_TABLE), option="latest_table"
        )
        self._neighbors_table = sql_identifier(
            options.get("neighbors_table", DEFAULT_NEIGHBORS_TABLE), option="neighbors_table"
        )
        self._engine = create_engine(require_option(options, "database_url", "db"), pool_pre_ping=True)

    def _read_ranked(self, table: str, k: int) -> pd.DataFrame:
        sql = text(f'SELECT * FROM "{table}" ORDER BY "{RANK_COLUMN}" ASC LIMIT :k')
        try:
            return pd.read_sql(sql, self._engine, params={"k": k})
        except MISSING_TABLE_ERRORS:
            return pd.DataFrame()

    def get_popular(self, k: int) -> pd.DataFrame:
        return self._read_ranked(self._popular_table, k)

    def get_latest(self, k: int) -> pd.DataFrame:
        return self._read_ranked(self._latest_table, k)

    def get_similar(self, item_id: str, k: int) -> pd.DataFrame:
        sql = text(
            f'SELECT * FROM "{self._neighbors_table}" WHERE "{ITEM_COLUMN}" = :item_id '
            f'ORDER BY "{RANK_COLUMN}" ASC LIMIT :k'
        )
        try:
            return pd.read_sql(sql, self._engine, params={"item_id": item_id, "k": k})
        except MISSING_TABLE_ERRORS:
            return empty_neighbors_frame()


def similar_as_surface(neighbors: pd.DataFrame) -> pd.DataFrame:
    """Map neighbor rows onto the recommendation item columns."""
    if neighbors.empty:
        return empty_surface_frame(source="item_based")
    out = neighbors.copy()
    if NEIGHBOR_ITEM_COLUMN in out.columns:
        out[ITEM_COLUMN] = out[NEIGHBOR_ITEM_COLUMN].astype(str)
    out[SOURCE_COLUMN] = "item_based"
    keep = [ITEM_COLUMN, RANK_COLUMN, SCORE_COLUMN, SOURCE_COLUMN]
    present = [column for column in keep if column in out.columns]
    return out[present].reset_index(drop=True)
