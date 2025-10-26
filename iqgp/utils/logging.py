"""Lightweight progress logger."""

from __future__ import annotations

import dataclasses
import sys
from datetime import datetime
from typing import Any, Dict


@dataclasses.dataclass
class ProgressLogger:
    log_every: int = 10

    def log(self, step: int, message: str, extra: Dict[str, Any] | None = None) -> None:
        if step % self.log_every != 0:
            return
        now = datetime.utcnow().isoformat()
        payload = {"time": now, "step": step, "message": message}
        if extra:
            payload.update(extra)
        sys.stdout.write(str(payload) + "\n")
        sys.stdout.flush()


__all__ = ["ProgressLogger"]
