from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .config import CACHE_DIR as DEFAULT_CACHE_DIR


class JSONCache:

    def __init__(self, namespace: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self.namespace = namespace
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = cache_dir / f"{namespace}.json"
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _key(self, key: str) -> str:
        return key.lower().strip()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(self._key(key), default)

    def has(self, key: str) -> bool:
        return self._key(key) in self._data

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[self._key(key)] = value

    _FLUSH_ATTEMPTS = 5
    _FLUSH_BACKOFF = 0.4

    def flush(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(f".{os.getpid()}.json.tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False),
                           encoding="utf-8")
            for attempt in range(self._FLUSH_ATTEMPTS):
                try:
                    tmp.replace(self.path)
                    return
                except OSError as exc:
                    if attempt == self._FLUSH_ATTEMPTS - 1:
                        print(f"  [cache] could not write {self.path.name} "
                              f"({type(exc).__name__}); keeping the previous "
                              f"file and retrying on the next flush",
                              flush=True)
                        tmp.unlink(missing_ok=True)
                        return
                    time.sleep(self._FLUSH_BACKOFF * (attempt + 1))
