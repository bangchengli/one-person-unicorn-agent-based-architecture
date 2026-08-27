from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from core.cache import JSONCache
from core.concurrent_fetch import SKIP, run_cached_pool
from core.config import ANALYST_OUT_DIR, MAX_WORKERS, STALL_TIMEOUT, ensure_dirs
from core.utils import println, timestamp

from .agent import run_analyst
from .evaluation import evaluate_support_resistance

COLUMNS = [
    "model", "ticker", "status", "latency_sec", "input_tokens", "output_tokens",
    "cost_usd", "cost_source", "token_source", "direction", "output_length",
    "support_breached", "resistance_breached", "mae_margin_pct",
    "coverage_ratio", "days_scored", "cache_hit", "error",
]


def _row_from_report(model: str, ticker: str, end: str, report: dict) -> dict:
    ops = report.get("_ops") or {}
    row = {c: None for c in COLUMNS}
    row.update({
        "model": model,
        "ticker": ticker,
        "status": ops.get("status", "failed"),
        "latency_sec": round(float(ops.get("llm_latency_sec") or 0.0), 2),
        "input_tokens": int(ops.get("input_tokens") or 0),
        "output_tokens": int(ops.get("output_tokens") or 0),
        "cost_usd": ops.get("cost_usd"),
        "cost_source": ops.get("cost_source"),
        "token_source": ops.get("token_source"),
        "cache_hit": bool(ops.get("cache_hit")),
        "error": ops.get("error"),
    })
    if ops.get("status") != "success":
        return row

    forecast = report.get("forecast") or {}
    row["direction"] = forecast.get("direction")
    row["output_length"] = len(report.get("full_assessment") or "")

    last_close = (report.get("market_snapshot") or {}).get("last_close")
    if last_close:
        scored = evaluate_support_resistance(ticker, report, end, float(last_close))
        if scored:
            report["_quant_eval"] = scored
            row.update({k: scored[k] for k in
                        ("support_breached", "resistance_breached",
                         "mae_margin_pct", "coverage_ratio", "days_scored")})
    return row


def run_batch(ticker: str, start: str, end: str, models: list[str], *,
              use_cache: bool = True,
              max_workers: int = MAX_WORKERS,
              stall_timeout: float = STALL_TIMEOUT,
              save: bool = True) -> pd.DataFrame:
    ensure_dirs()
    cache = JSONCache("analyst_batch_rows")
    rows: list[dict] = []

    def worker(model: str):
        report = run_analyst(ticker, start, end, model, use_cache=use_cache)
        row = _row_from_report(model, ticker, end, report)
        rows.append(row)

        if row["status"] == "success":
            out = ANALYST_OUT_DIR / f"{ticker}_{model.replace('/', '_')}_{timestamp()}.json"
            try:
                out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            except (OSError, TypeError) as e:
                println(f"[analyst] could not save {model} report: {type(e).__name__}")

        status = row["status"]
        cost = row["cost_usd"]
        println(f"  {model:<30} {status:<8} "
                f"{row['latency_sec']:>7.2f}s  "
                f"{'$%.6f' % cost if isinstance(cost, (int, float)) else 'n/a':>12}")
        return row if status == "success" else SKIP

    println(f"== analyst batch: {ticker} across {len(models)} models ==")
    _, stalled, leftover = run_cached_pool(
        models, worker, cache,
        label="analyst",
        max_workers=max_workers,
        stall_timeout=stall_timeout,
        progress_every=1,
        default_value=SKIP,
        stall_note="stuck models abandoned; rerun to retry them",
    )
    if stalled and leftover:
        println(f"[analyst] never completed: {', '.join(map(str, leftover))}")
        for model in leftover:
            row = {c: None for c in COLUMNS}
            row.update({"model": model, "ticker": ticker, "status": "stalled",
                        "error": f"no response within {stall_timeout:.0f}s"})
            rows.append(row)

    df = pd.DataFrame(rows, columns=COLUMNS)
    if save and not df.empty:
        path = ANALYST_OUT_DIR / f"final_comparison_{ticker}_{timestamp()}.csv"
        df.to_csv(path, index=False)
        println(f"Comparison table saved to: {path}")
    return df
