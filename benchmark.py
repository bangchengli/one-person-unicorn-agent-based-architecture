# benchmark.py (with portfolio tables: long & wide)
import json
import re
import time
import pandas as pd

from portfolio_benchmark.utils import ensure_dirs, println, progress
from portfolio_benchmark.llm_portfolio import generate_portfolio_with_model
from portfolio_benchmark.market_data import fetch_union_prices_polygon
from portfolio_benchmark.metrics import (
    equity_curve_from_equal_weight,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    hist_var_cvar,
)
from portfolio_benchmark.visualization import visualize_cost_efficiency
from portfolio_benchmark.pricing import load_pricing_config
from portfolio_benchmark.config import (
    OPENAI_API_KEY,
    POLYGON_API_KEY,
    CHART_SAVE_DIR,
    CSV_SAVE_DIR,
)


def benchmark_models(
    models,
    start: str,
    end: str,
    use_web_search: bool = True,
    use_unified_api: bool = False,
    unified_provider: str = None,
    unified_model_map: dict = None,
    *,
    return_portfolios: bool = False,   # NEW: optionally return the two new tables
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Benchmark pipeline (quality removed):
      1) Generate portfolios for all models
      2) Build union of tickers
      3) Fetch union prices once (cache + in-proc memo)
      4) Slice per model and compute metrics (+ cost efficiency)

    If return_portfolios=True, returns a tuple:
      (metrics_df, portfolios_long_df, portfolios_wide_df)
    """
    ensure_dirs()
    rows = []

    # 1) Generate all portfolios
    progress(0, note="generate portfolios for all models")
    portfolios = {}
    meta = {}
    union = set()

    for m in models:
        if use_unified_api and unified_provider and unified_model_map and m in unified_model_map:
            # unified path (Anthropic/xAI/DeepSeek/LLaMA etc.)
            tickers, gen_latency, in_tok, out_tok, est_cost = generate_portfolio_with_model(
                m, use_web_search=False, provider=unified_provider
            )
            label = unified_model_map[m]  # use real remote model id in output table
        else:
            # openai path (or explicit provider inside llm_portfolio)
            tickers, gen_latency, in_tok, out_tok, est_cost = generate_portfolio_with_model(
                m, use_web_search=use_web_search, provider="openai"
            )
            label = m

        tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
        portfolios[m] = tickers
        meta[m] = (label, gen_latency, in_tok, out_tok, est_cost)
        union.update(tickers)
        println(f"{label} -> {len(tickers)} tickers: {', '.join(tickers[:8])}{'...' if len(tickers) > 8 else ''}")

    # === NEW: build and save portfolio tables (long & wide) ===
    ts_tables = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")

    # Long-form: one row per (model, rank, ticker)
    long_rows = []
    for m, tick_list in portfolios.items():
        label, _, _, _, _ = meta[m]
        for i, t in enumerate(tick_list, start=1):
            long_rows.append({"model": label, "rank": i, "ticker": t})
    portfolios_long = pd.DataFrame(long_rows).sort_values(["model", "rank"])
    out_long = (CSV_SAVE_DIR / f"portfolios_long_{ts_tables}.csv")
    portfolios_long.to_csv(out_long, index=False)
    println(f"Portfolios (long) saved to: {out_long}")

    # Wide-form: one row per model, tickers joined with ';'
    wide_rows = []
    for m, tick_list in portfolios.items():
        label, _, _, _, _ = meta[m]
        joined = ";".join(tick_list)
        wide_rows.append({"model": label, "tickers": joined, "n_tickers": len(tick_list)})
    portfolios_wide = pd.DataFrame(wide_rows).sort_values(["model"])
    out_wide = (CSV_SAVE_DIR / f"portfolios_wide_{ts_tables}.csv")
    portfolios_wide.to_csv(out_wide, index=False)
    println(f"Portfolios (wide) saved to: {out_wide}")
    # === END NEW ===

    union = sorted(union)
    if not union:
        println("No tickers across all models. Abort.")
        empty = pd.DataFrame()
        return (empty, portfolios_long, portfolios_wide) if return_portfolios else empty

    # 2) Fetch union prices once
    progress(1, note=f"fetch union prices: {len(union)} tickers")
    df_all = fetch_union_prices_polygon(union, start=start, end=end, use_cache=True)
    if df_all is None or df_all.empty:
        println("Failed to fetch union prices.")
        empty = pd.DataFrame()
        return (empty, portfolios_long, portfolios_wide) if return_portfolios else empty

    # 3) Compute metrics
    progress(2, note="compute metrics")

    # 4) Slice per model and compute metrics
    for idx, m in enumerate(models, 1):
        label, gen_latency, in_tok, out_tok, est_cost = meta[m]
        tickers = [t for t in portfolios[m] if t in df_all.columns]
        if not tickers:
            println(f"{label}: no valid tickers after slicing union df, skip")
            continue

        # Slice prices
        df = df_all[tickers].copy()

        # Metrics
        equity = equity_curve_from_equal_weight(df)
        ann_ret = annualized_return(equity)
        ann_vol = annualized_volatility(df)
        sharpe = sharpe_ratio(df)
        var95, cvar95 = hist_var_cvar(df, level=0.95)
        mdd = max_drawdown(equity)

        # Efficiency / cost metrics
        total_tokens = float((in_tok or 0) + (out_tok or 0))
        cost = float(est_cost or 0.0)
        lat = float(gen_latency or 0.0)

        sharpe_per_dollar = float("nan") if cost == 0 else float(sharpe) / cost
        return_per_dollar = float("nan") if cost == 0 else float(ann_ret) / cost

        # Prefer per-1k token cost as the primary normalization
        usd_per_1k_tokens = float("nan") if total_tokens == 0 else cost / (total_tokens / 1000.0)
        tokens_per_sec = float("nan") if lat == 0 else total_tokens / lat
        cost_per_sec = float("nan") if lat == 0 else cost / lat
        sharpe_per_1k_tokens = (
            float("nan") if not (usd_per_1k_tokens and usd_per_1k_tokens > 0)
            else float(sharpe) / usd_per_1k_tokens
        )
        return_per_1k_tokens = (
            float("nan") if not (usd_per_1k_tokens and usd_per_1k_tokens > 0)
            else float(ann_ret) / usd_per_1k_tokens
        )

        rows.append({
            "model": label,
            "n_tickers": len(df.columns),
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "VaR_95": var95,
            "CVaR_95": cvar95,
            "llm_latency_sec": gen_latency,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "est_api_cost_usd": est_cost,
            # cost-efficiency
            "usd_per_1k_tokens": usd_per_1k_tokens,
            "sharpe_per_dollar": sharpe_per_dollar,
            "return_per_dollar": return_per_dollar,
            "sharpe_per_1k_tokens": sharpe_per_1k_tokens,
            "return_per_1k_tokens": return_per_1k_tokens,
            "tokens_per_sec": tokens_per_sec,
            "cost_per_sec": cost_per_sec,
        })
        progress(3, note=f"{label} | computed ({idx}/{len(models)})")

    res = pd.DataFrame(rows).sort_values(["sharpe"], ascending=[False])
    ts = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    out_csv = (CSV_SAVE_DIR / f"benchmark_{ts}.csv")
    res.to_csv(out_csv, index=False)
    println(f"\nBenchmark saved to: {out_csv}")

    if return_portfolios:
        return res, portfolios_long, portfolios_wide
    return res


def run_multi_models(
    model_specs,
    start: str,
    end: str,
    use_web_search: bool = True,
    *,
    return_portfolios: bool = True   # NEW: 默认就把组合表也返回/落盘
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    遍历多家模型，逐个跑 benchmark_models(..., return_portfolios=True)，
    汇总 metrics 以及 portfolios_long / portfolios_wide 到单一 DataFrame，并各自落盘。
    返回：
      - 仅 metrics DataFrame（当 return_portfolios=False）
      - 或 (metrics_df, portfolios_long_all, portfolios_wide_all)
    """
    ensure_dirs()

    metrics_list = []
    long_all_rows = []
    wide_all_rows = []

    for spec in model_specs:
        label = spec["name"]                 # 展示别名
        provider = spec["provider"].lower()  # 'openai' | 'anthropic' | 'xai' | 'deepseek' | 'llama'
        model_id = spec["model"]             # 真实 API model id

        # 统一都要 portfolios，因此都传 return_portfolios=True
        if provider == "openai":
            res_tuple = benchmark_models(
                models=[model_id],
                start=start, end=end,
                use_web_search=use_web_search,
                use_unified_api=False,
                return_portfolios=True
            )
        else:
            res_tuple = benchmark_models(
                models=[model_id],
                start=start, end=end,
                use_web_search=False,
                use_unified_api=True,
                unified_provider=provider,
                unified_model_map={model_id: model_id},
                return_portfolios=True
            )

        if not res_tuple or (isinstance(res_tuple, tuple) and (res_tuple[0] is None or res_tuple[0].empty)):
            println(f"{label}: empty benchmark result, skip")
            continue

        metrics_df, portfolios_long, portfolios_wide = res_tuple

        # 用别名覆盖显示（metrics 和两个组合表都一致）
        metrics_df = metrics_df.copy()
        metrics_df.loc[:, "model"] = label
        metrics_list.append(metrics_df)

        if portfolios_long is not None and not portfolios_long.empty:
            pl = portfolios_long.copy()
            pl.loc[:, "model"] = label
            long_all_rows.append(pl)

        if portfolios_wide is not None and not portfolios_wide.empty:
            pw = portfolios_wide.copy()
            pw.loc[:, "model"] = label
            wide_all_rows.append(pw)

    if not metrics_list:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()) if return_portfolios else pd.DataFrame()

    # —— 合并并落盘 ——
    combined_metrics = pd.concat(metrics_list, ignore_index=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    out_metrics = (CSV_SAVE_DIR / f"benchmark_all_models_{ts}.csv")
    combined_metrics.to_csv(out_metrics, index=False)
    println(f"\nCombined benchmark saved to: {out_metrics}")

    portfolios_long_all = pd.DataFrame()
    portfolios_wide_all = pd.DataFrame()

    if long_all_rows:
        portfolios_long_all = pd.concat(long_all_rows, ignore_index=True).sort_values(["model", "rank"])
        out_long_all = (CSV_SAVE_DIR / f"portfolios_long_all_models_{ts}.csv")
        portfolios_long_all.to_csv(out_long_all, index=False)
        println(f"Portfolios (long, ALL MODELS) saved to: {out_long_all}")

    if wide_all_rows:
        portfolios_wide_all = pd.concat(wide_all_rows, ignore_index=True).sort_values(["model"])
        out_wide_all = (CSV_SAVE_DIR / f"portfolios_wide_all_models_{ts}.csv")
        portfolios_wide_all.to_csv(out_wide_all, index=False)
        println(f"Portfolios (wide, ALL MODELS) saved to: {out_wide_all}")

    return (combined_metrics, portfolios_long_all, portfolios_wide_all) if return_portfolios else combined_metrics



if __name__ == "__main__":
    START = "2024-07-25"
    END   = "2025-08-07"

    # Make sure OPENAI_API_KEY and POLYGON_API_KEY are set.
    if (not OPENAI_API_KEY) or (not POLYGON_API_KEY):
        println("Please set OPENAI_API_KEY and POLYGON_API_KEY as environment variables.")
        raise SystemExit(0)

    load_pricing_config()

    ALL_MODELS = [
        {"name": "claude_sonnet",   "provider": "anthropic","model": "claude-sonnet-4-20250514"},
        {"name": "grok_4",          "provider": "xai",      "model": "grok-4"},
        {"name": "llama4_maverick", "provider": "llama",    "model": "meta-llama/llama-4-maverick"},
        {"name": "deepseek-r1",     "provider": "deepseek", "model": "deepseek-reasoner"},
        {"name": "deepseek-v3",     "provider": "deepseek", "model": "deepseek-chat"},
        {"name": "gpt-4o",          "provider": "openai",   "model": "gpt-4o"},
        {"name": "gpt-4o-mini",     "provider": "openai",   "model": "gpt-4o-mini"},
        {"name": "gpt-5-mini",      "provider": "openai",   "model": "gpt-5-mini"},
        {"name": "gpt-5-nano",      "provider": "openai",   "model": "gpt-5-nano"},
        {"name": "gpt-5",           "provider": "openai",   "model": "gpt-5"},
        {"name": "o3",              "provider": "openai",   "model": "o3"},
        {"name": "o4-mini",         "provider": "openai",   "model": "o4-mini"},
    ]

    res, portfolios_long_all, portfolios_wide_all = run_multi_models(
    ALL_MODELS, START, END, use_web_search=True, return_portfolios=True
)


    try:
        visualize_cost_efficiency(
            res, x_col="usd_per_1k_tokens", y_col="sharpe",
            title="Cost Performance Frontier (per-1k tokens)"
        )
        visualize_cost_efficiency(
            res, x_col="llm_latency_sec", y_col="sharpe",
            title="Latency Performance Frontier",
            outfile=str(CHART_SAVE_DIR / "frontier_latency_vs_sharpe.png")
        )
    except Exception as e:
        println(f"visualization failed: {e}")

    # Composite score (example; adjust weights as needed)
    res["composite_score_aggressive"] = (
        0.6  * res["sharpe"].fillna(0)
        - 0.15 * res["ann_vol"].fillna(0)
        - 10.0 * res["usd_per_1k_tokens"].fillna(0)   # cost per 1k tokens
        - 0.1 * res["llm_latency_sec"].fillna(0)
    )
    println(str(res.sort_values("composite_score_aggressive", ascending=False)))
