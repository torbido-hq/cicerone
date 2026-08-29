"""Online refresh result types (no ML imports — safe for the serve worker)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cicerone.io.recommendation_schema import REASONS_COLUMN, RECOMMENDATION_COLUMNS


def empty_online_rows() -> pd.DataFrame:
    return pd.DataFrame(columns=[*RECOMMENDATION_COLUMNS, REASONS_COLUMN])


@dataclass(frozen=True)
class OnlineRefreshResult:
    rows: pd.DataFrame
    users_refreshed: int = 0
    fit_partial_epochs: int = 0
    events_dropped_unknown: int = 0
    events_known: int = 0
    sequential_skipped: bool = False
