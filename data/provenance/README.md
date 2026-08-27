# Provenance of the published tables

Which run produced each table in *"One-Person Unicorn: An Agent-Based
Architecture for Automated Financial Services"* (IJISSC 17(1), 2026,
DOI 10.4018/IJISSC.409374), and what evidence for it survives.

Compiled 2026-08-27 by reconciling the manuscript against run artifacts
recovered from the machine the experiments were originally run on.

## Summary

| Table | Run | Source artifact | Status |
|---|---|---|---|
| Table 1 (technical snapshot) | 2026-04-06 analyst | `NVDA_*_20260406_*.json` in `../analyst_outputs/` | verified, 11/11 fields |
| Table 2 (latency/tokens/cost/direction) | **2026-01-22 23:59 analyst** | `table2_analyst_20260122/` | verified, 10/10 rows |
| Table 3 (risk-level evaluation) | **2026-04-06 19:42 analyst** | `../analyst_outputs/final_comparison_NVDA_20260406_194227.csv` | verified, 10/10 rows |
| Table 4 (portfolio metrics) | **2026-01-09 portfolio** | `table45_portfolio_20260109/` | **recomputed exactly, 84/84 cells** |
| Table 5 (latency/cost/throughput) | 2026-01-09 portfolio (same run as Table 4) | ticker lists + pricing snapshots present; summary CSV lost | partially verifiable |

## Why Tables 2 and 3 come from different runs

They look like two halves of one CSV, because in the *final* version of
`analyst.py` a single execution writes both column groups to one file. That is
not how they were produced.

The January version of `analyst.py` (452 lines, recovered as
`table2_analyst_20260122/analyst_january.py`) contains **no**
`evaluate_quant_sr`, no `support_breached`, no `mae_margin`, no
`coverage_ratio`. Its CSV columns are exactly:

    Model, Ticker, Date, Status, Latency (s), Input Tokens,
    Output Tokens, Est. Cost ($), Direction, Output Length

which is exactly Table 2 and nothing else. The risk-level scoring that Table 3
reports **did not exist yet**. It was written later, and the analyst sweep was
re-run on 2026-04-06 to produce it.

So the two tables are from different runs because the second table's metrics
were not implementable at the time of the first. This is ordinary research
progression, not a discrepancy — but the manuscript does not say so, and a
reader comparing the two tables row-by-row will notice they disagree on
latency and direction. **A future revision should state that Tables 2 and 3
come from separate executions.**

## Table 2: the "no bearish prediction" claim

`final_comparison_NVDA_20260122_235928.csv` reproduces all ten Table 2 rows
exactly (latency, input tokens, output tokens, cost, direction, output length).

Its Direction column reads, in full:

    neutral_to_bullish, up, Up, moderately bullish, neutral,
    slightly_up, neutral, sideways, up, slightly bullish

No bearish forecast appears. The manuscript's statement on p.8 — *"No model in
the evaluation produces a bearish prediction"* — **is true of its source run**,
as are *"GPT-5 is the slowest at 50.71 seconds"* and *"Llama 4 Maverick is the
fastest, completing analysis in 3.08 seconds"*.

Note that the later 2026-04-06 run *does* contain bearish forecasts (grok-4
returned `bearish`). The claim is specific to the run it reports, and forecast
direction is not stable across repetitions — see "Known limitations" below.

## Table 4: recomputed exactly

Table 4's six metric columns are a deterministic function of (ticker list,
price window). Unlike anything an agent produces, they can be reproduced to the
digit — and were.

    Run:     2026-01-09 portfolio generation, 12 models
    Window:  2024-07-25 .. 2025-08-07 (260 trading days)
    Prices:  Yahoo chart endpoint, split-adjusted close
    Code:    portfolio/metrics.py, unchanged from the published implementation
    Script:  see table4_recomputed.csv for the full paper-vs-recomputed table

    Result:  84 of 84 cells match at 4 decimal places.
             N Tickers 12/12, Ann Return 12/12, Ann Vol 12/12,
             Sharpe 12/12, Max Drawdown 12/12, VaR 12/12, CVaR 12/12.
             No ticker was dropped for missing data.

**One methodological detail matters if anyone repeats this.** Use
split-adjusted close, NOT dividend-adjusted close. Polygon's `adjusted=true`
adjusts for splits only; Yahoo's `adjclose` additionally reinvests dividends.
Using `adjclose` adds a full dividend yield to every annualised return —
about +1.8 to +2.4 percentage points on this large-cap universe — and inflates
Sharpe by roughly 10%, while leaving volatility, drawdown, VaR and CVaR almost
untouched. That asymmetric signature is exactly what a dividend-convention
mismatch looks like, and it is what the first recomputation attempt produced
before the convention was corrected.

## Table 5: what can and cannot be checked

Table 5 comes from the same 2026-01-09 execution as Table 4 — the benchmark
writes both from one DataFrame. Since Table 4 reproduces exactly, Table 5's
rows describe the same verified portfolios.

- **Reproducible**: Sharpe/Dollar and Return/Dollar are Table 4's Sharpe and
  Ann Return divided by Table 5's API Cost; Cost/Sec is cost over latency.
  These are internally consistent.
- **Not reproducible**: raw `Latency`, `Tokens/Sec` and `API Cost` existed only
  in the summary CSV, which was not preserved. The per-call pricing snapshots
  in `table45_portfolio_20260109/effective_pricing_*.json` confirm which price
  table was in force, but not the token counts.

## Known limitations of the study, for the record

Documented here so they are on the record rather than discovered later.

1. **Every reported figure is n = 1.** No repetition count, variance, or seed
   is stated, and `cache_seed` is explicitly `None`. The repo's own archive
   shows the instability directly: four identical grok-4 runs on 2026-03-05 and
   2026-04-06 returned `neutral`, `bullish`, `bearish`, `bearish`, with latency
   spanning 38.25–57.71 s. Forecast direction and coverage ratio are draws from
   a distribution, not model constants. `stability_test_for_model` exists in the
   code but was never enabled (`REPEATS = 0`).

2. **Providers were not on equal footing in the portfolio benchmark.**
   `use_web_search=True` was passed only down the OpenAI branch; every other
   provider was hardcoded to `False`. The hosted web-search payload inflates
   input tokens roughly 60x. Output caps also differed: OpenAI uncapped,
   Anthropic/xAI/Llama 1024, DeepSeek 1000 — so DeepSeek-R1's reported
   highest-latency result was generated against its own truncation limit.
   This makes Table 5's Tokens/Sec, Cost/Sec and latency an *as-deployed*
   comparison rather than a controlled one. The refactored `core/llm.py` now
   records `web_search_used` and `max_tokens_used` on every row so the
   condition is visible in the output.

3. **"MAE margin" is misnamed.** p.3 expands it as "mean absolute error", but
   the quantity is a signed difference of two percentages — no mean, no
   absolute value. Table 3 reports −1.42% for two models, which no absolute
   error can be. The formula and the sign-convention sentence are both correct;
   only the expansion is wrong. "Maximum Adverse Excursion margin" fits the
   same acronym and the actual computation. See `analyst/evaluation.py`.

4. **The backtest window is not stated in the manuscript.** Table 4's
   annualised figures are uninterpretable without it. It is
   2024-07-25 to 2025-08-07, with `rf = 0.0` and `periods_per_year = 252`.

5. **N Tickers is post-validation.** It is `len(df.columns)` after the price
   fetch, so a ticker the data provider could not resolve is silently dropped
   rather than counted against the model. In the 2026-01-09 run this changed
   nothing — all 56 tickers across all 12 portfolios resolved.

## Files

    table2_analyst_20260122/
        final_comparison_NVDA_20260122_235928.csv   Table 2, exact
        final_table_NVDA_20260122_235928.tex        LaTeX as generated
        NVDA_{gpt-5,o3,o4-mini}_20260122_*.json     per-model reports
        analyst_january.py                          the 452-line version

    table45_portfolio_20260109/
        _tickers.json                               12 models -> ticker lists
        *_portfolio_20260109-*.json                 parsed portfolios
        *_portfolio_20260109-*.txt                  raw model responses
        effective_pricing_20260109-*.json           pricing in force per call

    table4_recomputed.csv                           paper vs recomputed, all columns
