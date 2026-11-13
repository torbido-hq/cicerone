"""Steps that failed after an incremental write, retried until they succeed (the batch itself is acked)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

POST_WRITE_RETRY_SECONDS = 30.0


@dataclass
class PostWriteRepair:
    refresh: bool = False
    manifest: dict[str, Any] | None = None
    publish_user_ids: set[str] = field(default_factory=set)
    next_attempt_at: float = 0.0

    @property
    def pending(self) -> bool:
        return self.refresh or self.manifest is not None or bool(self.publish_user_ids)

    def due(self) -> bool:
        return self.pending and time.monotonic() >= self.next_attempt_at

    def back_off(self) -> None:
        self.next_attempt_at = time.monotonic() + POST_WRITE_RETRY_SECONDS
