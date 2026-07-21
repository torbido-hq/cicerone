"""Protocols for pluggable input/output backends.

Both input and output are abstracted behind these two interfaces so the job
doesn't care whether it's reading/writing static files (S3-compatible or
local disk) or a database.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class InputSource(Protocol):
    def read_events(self) -> pd.DataFrame: ...

    def read_users(self) -> pd.DataFrame | None: ...

    def read_items(self) -> pd.DataFrame | None: ...


class OutputSink(Protocol):
    def write_recommendations(self, df: pd.DataFrame) -> None: ...

    def write_manifest(self, manifest: dict) -> None: ...
