from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime


def println(msg: str = "") -> None:
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```", 2)
        if len(parts) >= 2:
            t = parts[1]
            if t.startswith("json"):
                t = t[4:]
            if t.endswith("```"):
                t = t[:-3]
    return t.strip()


def extract_json_objects(text: str) -> list[str]:
    results: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                results.append(text[start:i + 1])
    return results


def parse_json_report(text: str, *, require_key: str | None = None) -> dict | None:
    for candidate in extract_json_objects(strip_fences(text)):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (require_key is None or require_key in obj):
            return obj
    return None


@dataclass
class Progress:

    bar_len: int = 30
    _name: str = field(default="", init=False)
    _total: int = field(default=1, init=False)
    _done: int = field(default=0, init=False)
    _start: float = field(default=0.0, init=False)

    def start(self, name: str, total: int) -> None:
        self._name, self._total, self._done = name, max(1, int(total)), 0
        self._start = time.perf_counter()
        self._render("start")

    def update(self, inc: int = 1, note: str = "") -> None:
        self._done = min(self._total, self._done + int(inc))
        self._render(note)

    def finish(self, note: str = "done") -> None:
        self._done = self._total
        self._render(note)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self, note: str = "") -> None:
        elapsed = max(1e-6, time.perf_counter() - self._start)
        rate = self._done / elapsed
        eta = (self._total - self._done) / rate if rate > 0 else float("inf")
        pct = self._done / self._total
        bar = "#" * int(self.bar_len * pct) + "-" * (self.bar_len - int(self.bar_len * pct))
        txt = (f"[{bar}] {self._name} {self._done}/{self._total} "
               f"({pct * 100:5.1f}%) | {rate:4.1f}/s | ETA {eta:5.1f}s")
        if note:
            txt += f" | {note}"
        try:
            width = shutil.get_terminal_size().columns
        except OSError:
            width = 120
        sys.stdout.write("\r" + txt[:width].ljust(width))
        sys.stdout.flush()
