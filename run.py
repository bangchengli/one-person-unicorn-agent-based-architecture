from __future__ import annotations

import argparse
import sys

from core.config import (
    ANALYST_OUT_DIR,
    CSV_DIR,
    DATA_DIR,
    PROVIDER_ENV,
    configured_providers,
    ensure_dirs,
    provider_key,
)
from core.utils import println

ANALYST_MODELS = [
    "claude-sonnet-4-20250514",
    "meta-llama/llama-4-maverick",
    "grok-4",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5",
    "o3",
    "o4-mini",
]

PORTFOLIO_MODELS = [
    {"name": "claude_sonnet", "provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    {"name": "grok_4", "provider": "xai", "model": "grok-4"},
    {"name": "llama4_maverick", "provider": "llama", "model": "meta-llama/llama-4-maverick"},
    {"name": "deepseek-r1", "provider": "deepseek", "model": "deepseek-reasoner"},
    {"name": "deepseek-v3", "provider": "deepseek", "model": "deepseek-chat"},
    {"name": "gpt-4o", "provider": "openai", "model": "gpt-4o"},
    {"name": "gpt-4o-mini", "provider": "openai", "model": "gpt-4o-mini"},
    {"name": "gpt-5-mini", "provider": "openai", "model": "gpt-5-mini"},
    {"name": "gpt-5-nano", "provider": "openai", "model": "gpt-5-nano"},
    {"name": "gpt-5", "provider": "openai", "model": "gpt-5"},
    {"name": "o3", "provider": "openai", "model": "o3"},
    {"name": "o4-mini", "provider": "openai", "model": "o4-mini"},
]


def cmd_doctor(_args) -> int:
    println("== credentials ==")
    for provider, env_name in PROVIDER_ENV.items():
        key = provider_key(provider)
        mark = "ok " if key else "-- "
        detail = f"set ({len(key)} chars)" if key else f"not set ({env_name})"
        println(f"  {mark} {provider:<10} {detail}")

    from core.config import POLYGON_API_KEY
    println(f"  {'ok ' if POLYGON_API_KEY else '!! '} polygon    "
            f"{'set' if POLYGON_API_KEY else 'NOT SET - required for all market data'}")

    println("\n== dependencies ==")
    for mod in ("pandas", "numpy", "requests", "openai", "autogen",
                "pypfopt", "matplotlib", "pyarrow"):
        try:
            __import__(mod)
            println(f"  ok  {mod}")
        except ImportError:
            println(f"  --  {mod} (pip install -r requirements.txt)")

    println("\n== paths ==")
    ensure_dirs()
    for label, path in (("data", DATA_DIR), ("analyst out", ANALYST_OUT_DIR),
                        ("csv", CSV_DIR)):
        println(f"  {label:<12} {path}")

    usable = configured_providers()
    println(f"\n{len(usable)} of {len(PROVIDER_ENV)} providers usable: "
            f"{', '.join(usable) if usable else 'none'}")
    if not POLYGON_API_KEY:
        println("POLYGON_API_KEY is missing; nothing can run. See .env.example.")
        return 1
    return 0


def cmd_analyst(args) -> int:
    from analyst.batch import run_batch

    models = args.models or ANALYST_MODELS
    df = run_batch(args.ticker, args.start, args.end, models,
                   use_cache=not args.no_cache)
    if df.empty:
        println("No results produced.")
        return 1
    println("\n" + df.to_string(index=False))
    return 0


def cmd_portfolio(args) -> int:
    from portfolio.benchmark import benchmark

    specs = PORTFOLIO_MODELS
    if args.models:
        wanted = set(args.models)
        specs = [s for s in specs if s["name"] in wanted or s["model"] in wanted]
        if not specs:
            println(f"No known model matches {args.models}")
            return 1

    metrics, _ = benchmark(specs, args.start, args.end,
                           use_web_search=not args.no_web_search)
    if metrics.empty:
        println("No metrics produced.")
        return 1
    println("\n" + metrics.to_string(index=False))

    if not args.no_charts:
        try:
            from portfolio.visualization import visualize_cost_efficiency
            from core.config import CHART_DIR
            visualize_cost_efficiency(
                metrics, x_col="usd_per_1k_tokens", y_col="sharpe",
                title="Cost Performance Frontier (per-1k tokens)",
                outfile=str(CHART_DIR / "frontier_cost_vs_sharpe.png"))
            visualize_cost_efficiency(
                metrics, x_col="llm_latency_sec", y_col="sharpe",
                title="Latency Performance Frontier",
                outfile=str(CHART_DIR / "frontier_latency_vs_sharpe.png"))
        except Exception as e:
            println(f"chart generation failed: {type(e).__name__}: {e}")
    return 0


def cmd_stability(args) -> int:
    from portfolio.stability import stability_test_for_model

    model = args.model or "gpt-4o"
    result = stability_test_for_model(model, args.repeats, args.start, args.end)
    for k, v in result.items():
        println(f"  {k:<18} {v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-Person Unicorn: agent-based financial services experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check credentials, dependencies and paths")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("analyst", help="Financial Analysis sweep across models")
    p.add_argument("--ticker", default="NVDA")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--models", nargs="*", help="subset of the model roster")
    p.add_argument("--no-cache", action="store_true",
                   help="re-bill every model instead of reusing cached reports")
    p.set_defaults(func=cmd_analyst)

    p = sub.add_parser("portfolio", help="Portfolio Management benchmark")
    p.add_argument("--start", default="2024-07-25")
    p.add_argument("--end", default="2025-08-07")
    p.add_argument("--models", nargs="*")
    p.add_argument("--no-web-search", action="store_true")
    p.add_argument("--no-charts", action="store_true")
    p.set_defaults(func=cmd_portfolio)

    p = sub.add_parser("stability", help="repeat one model to measure drift")
    p.add_argument("--model")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--start", default="2024-07-25")
    p.add_argument("--end", default="2025-08-07")
    p.set_defaults(func=cmd_stability)

    args = parser.parse_args(argv)
    ensure_dirs()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
