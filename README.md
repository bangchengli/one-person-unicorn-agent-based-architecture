# One-Person Unicorn — Infrastructure

Companion code for **"One-Person Unicorn: An Agent-Based Architecture for
Automated Financial Services"**, *International Journal of Information Systems
and Social Change* 17(1), 2026.
DOI [10.4018/IJISSC.409374](https://doi.org/10.4018/IJISSC.409374).

An agent-based architecture for automated financial services, evaluated across
GPT/o-series, Claude, Grok, DeepSeek and Llama on cost, latency and output
quality.

| Paper Figure 1 | Code |
|---|---|
| Ticker Symbol / Universal Prompt | `run.py` arguments |
| Agents (GPT/o-series, Claude, Grok, DeepSeek, Llama) | `core/llm.py`, `analyst/agent.py` |
| Financial Analysis (snapshot, news, forecasting) | `analyst/` |
| Portfolio Management (tickers, prices, equal weight) | `portfolio/` |
| Model Comparison (cost, latency) | `core/pricing.py` |
| Risk Advisory (volatility, drawdown, S/R, VaR, CVaR) | `analyst/evaluation.py`, `portfolio/metrics.py` |

---

## API keys required

Every credential is read from a `.env` file in the repo root. **No key is
stored in any `.py`, `.ipynb`, or config file in this repository.** Copy
`.env.example` to `.env` and fill in the values you have.

### Required

| Variable | Service | What it is used for | Where to get it |
|---|---|---|---|
| `POLYGON_API_KEY` | [Polygon.io](https://polygon.io) | The only market-data source: OHLCV bars, close-price series, and company news. Nothing runs without it. | polygon.io → Dashboard → API Keys |

**Polygon free-tier limits, measured 2026-08:** 5 requests/minute, and history
capped at **exactly 2 years back from today**. Requesting older data returns
HTTP 403 (`"Your plan doesn't include this data timeframe"`). The paper's
portfolio window starts 2024-07-25, which a free tier can no longer reach — see
"Reproducing the published tables" below for the workaround.

### Optional — one per model provider

Each is optional and independent. A provider with no key is reported as
`skipped` and the run continues with the models you can actually call, so a
partial key set produces a partial benchmark rather than nothing.

| Variable | Provider | Models it enables | Where to get it |
|---|---|---|---|
| `OPENAI_API_KEY` | [OpenAI](https://platform.openai.com) | `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4o`, `gpt-4o-mini`, `o3`, `o4-mini` | platform.openai.com → API keys |
| `ANTHROPIC_API_KEY` | [Anthropic](https://console.anthropic.com) | `claude-sonnet-4-20250514` | console.anthropic.com → API Keys |
| `XAI_API_KEY` | [xAI](https://console.x.ai) | `grok-4` | console.x.ai |
| `DEEPSEEK_API_KEY` | [DeepSeek](https://platform.deepseek.com) | `deepseek-chat`, `deepseek-reasoner` | platform.deepseek.com |
| `LLAMA_API_KEY` | [OpenRouter](https://openrouter.ai) | `meta-llama/llama-4-maverick` | openrouter.ai → Keys |

### Optional — unused by the paper pipeline

`FINNHUB_API_KEY` and `FMP_API_KEY` are read by `core/config.py` because the
original configuration carried them. No code path uses them. Leave blank.

### Cost note

Every model call is billed by its provider. A full 12-model portfolio sweep or
10-model analyst sweep is a real amount of money. The pipeline caches every
successful result and never re-bills for one it already has — use `--no-cache`
only when you deliberately want fresh calls.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in your keys
python run.py doctor          # verifies credentials, dependencies, paths
```

`doctor` prints which providers are usable and which dependencies are missing,
without printing any secret.

## Running

```bash
python run.py analyst   --ticker NVDA --start 2024-01-01 --end 2025-12-31
python run.py portfolio --start 2024-07-25 --end 2025-08-07
python run.py stability --model gpt-4o --repeats 5
```

Flags: `--models` to run a subset, `--no-cache` to re-bill every model instead
of reusing cached results, `--no-charts` to skip plotting.

## Layout

```
core/            config, credentials, caching, LLM clients, market data, pricing
analyst/         Financial Analysis: snapshot -> agent -> support/resistance scoring
portfolio/       Portfolio Management: generation -> prices -> metrics -> benchmark
data/provenance/ which run produced each published table, and the evidence
docs/            design notes: why each module is built the way it is
legacy/          the original scripts and notebooks as published
```

The source files carry no comments. The reasoning behind each module — and the
failures that shaped it — is in [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md).
Read the relevant section before changing a module.

## How it handles money and failure

- **Caching is by outcome.** Only a genuine success is cached. A missing
  credential, an HTTP failure, and an empty model response are three distinct
  states and none is stored as an answer — otherwise an outage freezes into
  permanent empty data.
- **A stall cannot take down a run.** If *nothing* completes within
  `STALL_TIMEOUT`, stuck calls are abandoned and everything that did answer is
  still written out. Abandoned items stay un-cached, so the next run retries.
- **An unknown cost is `null`, never `0.0`.** Coercing an unpriced model to
  zero would make it look like the cheapest model in a cost-performance study.
- **Estimated tokens are labelled** with `token_source`, so an estimate can
  never be mistaken for a metered measurement.

## Reproducing the published tables

[`data/provenance/README.md`](data/provenance/README.md) records which run
produced each table and what evidence survives.

Table 4 reproduces exactly — **84 of 84 cells at 4 decimal places** — from the
archived ticker lists. Two things matter if you repeat it:

1. Polygon's free tier cannot reach the window start (2024-07-25) any more.
   Yahoo's chart endpoint covers it, needs no key, and is what
   `data/provenance/recompute_table4.py` uses.
2. **Use split-adjusted close, not dividend-adjusted close.** Polygon's
   `adjusted=true` adjusts for splits only; Yahoo's `adjclose` also reinvests
   dividends, which adds a full dividend yield to every annualised return
   (+1.8 to +2.4 pp here) and inflates Sharpe by ~10%, while leaving
   volatility, drawdown, VaR and CVaR nearly untouched.

## License

Code released under the MIT License. The article itself is Open Access under
[CC BY 4.0](http://creativecommons.org/licenses/by/4.0/).
