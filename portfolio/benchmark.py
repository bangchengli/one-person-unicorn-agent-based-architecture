from __future__ import annotations

from typing import Optional

import pandas as pd

from core.concurrent_fetch import SKIP, run_cached_pool
from core.cache import JSONCache
from core.config import CSV_DIR, MAX_WORKERS, STALL_TIMEOUT, ensure_dirs
from core.utils import println, timestamp
from core.market import fetch_price_matrix

from .generate import generate_portfolio
from .metrics import (
    annualized_return,
    annualized_volatility,
    equity_curve_from_equal_weight,
    hist_var_cvar,
    max_drawdown,
    sharpe_ratio,
)

NAN = float("nan")


def _safe_div(numer: float, denom: float) -> float:
    try:
        return float(numer) / float(denom) if denom else NAN
    except (TypeError, ValueError):
        return NAN


def generate_all(models: list[dict], *, use_web_search: bool = True,
                 max_workers: int = MAX_WORKERS,
                 stall_timeout: float = STALL_TIMEOUT) -> dict[str, dict]:
    cache = JSONCache("portfolio_generation")
    results: dict[str, dict] = {}

    def worker(spec: dict):
        out = generate_portfolio(spec["model"], provider=spec.get("provider"),
                                 use_web_search=use_web_search)
        row = out.as_row()
        row["name"] = spec.get("name", spec["model"])
        row["tickers"] = out.tickers
        results[row["name"]] = row
        return row if out.ok else SKIP

    run_cached_pool(
        models, worker, cache,
        label="portfolio-gen",
        max_workers=max_workers,
        stall_timeout=stall_timeout,
        progress_every=1,
        default_value=SKIP,
        stall_note="providers still in flight were abandoned; rerun to retry",
    )
    return results


def benchmark(models: list[dict], start: str, end: str, *,
              use_web_search: bool = True,
              save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()

    println(f"== generating portfolios for {len(models)} models ==")
    generated = generate_all(models, use_web_search=use_web_search)

    portfolio_rows = [
        {"model": name, "rank": i, "ticker": t}
        for name, row in generated.items()
        for i, t in enumerate(row.get("tickers") or [], start=1)
    ]
    portfolios_df = pd.DataFrame(portfolio_rows)

    union = sorted({t for row in generated.values() for t in (row.get("tickers") or [])})
    if not union:
        println("No tickers produced by any model; nothing to benchmark.")
        return pd.DataFrame(), portfolios_df

    println(f"== fetching prices for {len(union)} unique tickers ==")
    prices = fetch_price_matrix(union, start, end)
    if prices is None or prices.empty:
        println("Price fetch failed; cannot compute metrics.")
        return pd.DataFrame(), portfolios_df

    rows = []
    for name, gen in generated.items():
        tickers = [t for t in (gen.get("tickers") or []) if t in prices.columns]
        if not tickers:
            println(f"[benchmark] {name}: no tickers resolvable against Polygon; "
                    f"status={gen.get('status')}")
            continue

        df = prices[tickers].copy()
        equity = equity_curve_from_equal_weight(df)
        ann_ret = annualized_return(equity)
        sharpe = sharpe_ratio(df)
        var95, cvar95 = hist_var_cvar(df, level=0.95)

        in_tok = int(gen.get("input_tokens") or 0)
        out_tok = int(gen.get("output_tokens") or 0)
        total_tokens = in_tok + out_tok
        latency = float(gen.get("llm_latency_sec") or 0.0)
        cost = gen.get("est_api_cost_usd")
        cost = NAN if cost is None else float(cost)

        usd_per_1k = _safe_div(cost, total_tokens / 1000.0) if total_tokens else NAN

        rows.append({
            "model": name,
            "n_tickers": len(df.columns),
            "ann_return": ann_ret,
            "ann_vol": annualized_volatility(df),
            "sharpe": sharpe,
            "max_drawdown": max_drawdown(equity),
            "VaR_95": var95,
            "CVaR_95": cvar95,
            "llm_latency_sec": latency,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "est_api_cost_usd": cost,
            "usd_per_1k_tokens": usd_per_1k,
            "sharpe_per_dollar": _safe_div(sharpe, cost),
            "return_per_dollar": _safe_div(ann_ret, cost),
            "sharpe_per_1k_tokens": _safe_div(sharpe, total_tokens / 1000.0),
            "return_per_1k_tokens": _safe_div(ann_ret, total_tokens / 1000.0),
            "sharpe_per_usd_per_1k": _safe_div(sharpe, usd_per_1k),
            "return_per_usd_per_1k": _safe_div(ann_ret, usd_per_1k),
            "tokens_per_sec": _safe_div(total_tokens, latency),
            "cost_per_sec": _safe_div(cost, latency),
            "web_search_used": gen.get("web_search_used"),
            "max_tokens_used": gen.get("max_tokens_used"),
            "status": gen.get("status"),
        })

    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values("sharpe", ascending=False)

    if save:
        ts = timestamp()
        metrics_path = CSV_DIR / f"benchmark_{ts}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        println(f"Benchmark saved to: {metrics_path}")
        if not portfolios_df.empty:
            pf_path = CSV_DIR / f"portfolios_{ts}.csv"
            portfolios_df.to_csv(pf_path, index=False)
            println(f"Portfolios saved to: {pf_path}")

    return metrics_df, portfolios_df
