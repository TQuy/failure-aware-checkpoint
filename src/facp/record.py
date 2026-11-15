"""Execution record: one JSON object per line, flushed as it is written.

The record is what the accounting trusts. A policy's own estimates are
written here too, but lost time is computed only from the timestamps.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator


class Record:
    def __init__(self, path: str | os.PathLike, source: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.source = source
        self._f = open(self.path, "a", buffering=1)

    def write(self, event: str, **fields: Any) -> None:
        line = {"t": time.time(), "src": self.source, "event": event, **fields}
        self._f.write(json.dumps(line) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def read(path: str | os.PathLike) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # a process killed mid-write can leave half a line
                continue
