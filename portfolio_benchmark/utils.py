import os
import sys
import time
from pathlib import Path

# 保证目录存在
def ensure_dirs():
    Path("csv_save").mkdir(parents=True, exist_ok=True)
    Path("fig_save").mkdir(parents=True, exist_ok=True)

# 打印信息
def println(msg):
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()

# 简单进度显示（旧版）
def progress(stage_idx: int, note: str = ""):
    stages = {
        0: "Generate portfolios",
        1: "Fetch prices",
        2: "Compute metrics",
        3: "Aggregate results",
    }
    stage_name = stages.get(stage_idx, f"Stage {stage_idx}")
    sys.stdout.write(f"[{stage_idx+1}/4] {stage_name}... {note}\n")
    sys.stdout.flush()

# =========================
# Enhanced progress with ETA
# =========================
import shutil
from dataclasses import dataclass, field

@dataclass
class SmartProgress:
    bar_len: int = 30
    _stage_name: str = field(default="", init=False)
    _stage_total: int = field(default=0, init=False)
    _stage_done: int = field(default=0, init=False)
    _stage_start: float = field(default=0.0, init=False)

    def _term_width(self) -> int:
        try:
            return shutil.get_terminal_size().columns
        except Exception:
            return 120

    def start_stage(self, name: str, total: int):
        self._stage_name = name
        self._stage_total = max(1, int(total))
        self._stage_done = 0
        self._stage_start = time.perf_counter()
        self._render(note="start")

    def update(self, inc: int = 1, note: str = ""):
        self._stage_done = min(self._stage_total, self._stage_done + int(inc))
        self._render(note=note)

    def set(self, done: int, note: str = ""):
        self._stage_done = min(self._stage_total, max(0, int(done)))
        self._render(note=note)

    def finish_stage(self, note: str = "done"):
        self._stage_done = self._stage_total
        self._render(note=note)
        sys.stdout.write("\n"); sys.stdout.flush()

    def _render(self, note: str = ""):
        elapsed = max(1e-6, time.perf_counter() - self._stage_start)
        rate = self._stage_done / elapsed
        remain = max(0, self._stage_total - self._stage_done)
        eta = remain / rate if rate > 0 else float("inf")
        pct = self._stage_done / self._stage_total
        filled = int(self.bar_len * pct)
        bar = "#" * filled + "-" * (self.bar_len - filled)
        txt = (f"[{bar}] {self._stage_name} "
               f"{self._stage_done}/{self._stage_total} "
               f"({pct*100:5.1f}%) | {rate:4.1f}/s | ETA {eta:5.1f}s")
        if note:
            txt += f" | {note}"
        width = self._term_width()
        sys.stdout.write("\r" + txt[:width].ljust(width))
        sys.stdout.flush()
