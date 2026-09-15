"""In-process LRU of per-user recommendation frames."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Collection

import pandas as pd

from cicerone.config import IOSettings
from cicerone.events.store import (
    empty_recommendations_frame,
    load_recommendations_for_users,
)
from cicerone.io.recommendation_reader import USER_COLUMN


class UpdaterUserCache:
    _cached_by_user: OrderedDict[str, pd.DataFrame]
    _user_cache_max_size: int
    _output_settings: IOSettings

    def _cache_put(self, user_id: str, frame: pd.DataFrame, *, protect: Collection[str]) -> None:
        self._cached_by_user[user_id] = frame
        self._cached_by_user.move_to_end(user_id)
        self._trim_user_cache(protect=protect)

    def _trim_user_cache(self, *, protect: Collection[str]) -> None:
        protect_set = set(protect)
        while len(self._cached_by_user) > self._user_cache_max_size:
            victim = next((key for key in self._cached_by_user if key not in protect_set), None)
            if victim is None:
                # Active batch larger than the cap; allow temporary oversize.
                return
            self._cached_by_user.pop(victim)

    def _frames_by_user(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if frame.empty or USER_COLUMN not in frame.columns:
            return {}
        # One assign for str user ids; reset_index materializes each group (no extra .copy()).
        keyed = frame.assign(**{USER_COLUMN: frame[USER_COLUMN].astype(str)})
        return {
            user_id: group.reset_index(drop=True) for user_id, group in keyed.groupby(USER_COLUMN, sort=False)
        }

    def _load_users(self, user_ids: set[str]) -> pd.DataFrame:
        missing = sorted(user_id for user_id in user_ids if user_id not in self._cached_by_user)
        if missing:
            loaded = load_recommendations_for_users(self._output_settings, missing)
            by_user = self._frames_by_user(loaded)
            for user_id in missing:
                self._cache_put(
                    user_id,
                    by_user.get(user_id, empty_recommendations_frame()),
                    protect=user_ids,
                )
        for user_id in user_ids:
            if user_id in self._cached_by_user:
                self._cached_by_user.move_to_end(user_id)
        parts = [
            self._cached_by_user[user_id]
            for user_id in sorted(user_ids)
            if not self._cached_by_user[user_id].empty
        ]
        if not parts:
            return empty_recommendations_frame()
        return pd.concat(parts, ignore_index=True)

    def _store_users_in_cache(self, user_ids: set[str], merged: pd.DataFrame) -> None:
        # After a successful replace, every id in user_ids was written. Missing from
        # merged means cleared rows — store empty frames (key present = loaded).
        by_user = self._frames_by_user(merged)
        for user_id in user_ids:
            self._cache_put(
                user_id,
                by_user.get(user_id, empty_recommendations_frame()),
                protect=user_ids,
            )
