from __future__ import annotations

import concurrent.futures
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

from .cache import JSONCache

SKIP = object()


def run_cached_pool(
    items: list,
    worker: Callable,
    cache: JSONCache,
    *,
    label: str,
    max_workers: int,
    stall_timeout: float,
    progress_every: int = 500,
    default_value=None,
    stall_note: str = "abandoned rest; un-cached items retry next run",
) -> tuple[int, bool, list]:
    if default_value is None:
        default_value = {}
    items = list(items)
    if not items:
        return 0, False, []

    def _cache_default(item) -> None:
        if default_value is SKIP:
            return
        value = dict(default_value) if isinstance(default_value, dict) \
            else default_value
        cache.set(item, value)

    t0 = time.time()
    done = 0
    stalled = False
    completed: set = set()
    pool = ThreadPoolExecutor(max_workers=max_workers)
    futures = {pool.submit(worker, it): it for it in items}
    pending = set(futures)
    while pending:
        done_set, pending = concurrent.futures.wait(
            pending, timeout=stall_timeout,
            return_when=concurrent.futures.FIRST_COMPLETED)
        if not done_set:
            stalled = True
            break
        for fut in done_set:
            item = futures[fut]
            try:
                result = fut.result()
            except Exception:
                result = None
            if result:
                cache.set(item, result)
            else:
                _cache_default(item)
            completed.add(item)
            done += 1
            if done % progress_every == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (len(items) - done) / rate if rate else 0
                print(f"  [{label}]   {done}/{len(items)} "
                      f"({rate:.1f}/s, {elapsed:.0f}s elapsed, "
                      f"~{eta:.0f}s remaining)", flush=True)
                cache.flush()
    pool.shutdown(wait=False, cancel_futures=True)
    cache.flush()
    note = f" (STALLED — {stall_note})" if stalled else ""
    print(f"  [{label}] fetch complete: {done}/{len(items)} in "
          f"{time.time() - t0:.1f}s{note}", flush=True)
    leftover = [it for it in items if it not in completed]
    return done, stalled, leftover
