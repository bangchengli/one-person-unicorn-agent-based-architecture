# Design notes

The source files carry no comments. Everything that would have been a comment
lives here instead, organised by module. Most of it records a failure that
actually happened — read the relevant section before changing a module, or the
same bug comes back.

---

## core/config.py

Single source of configuration. Every credential, path and tuning knob is
resolved here from the environment; nothing else in the codebase calls
`os.getenv`, and no key appears in a `.py` or `.ipynb`.

This replaced four disagreeing configuration sites:

- plaintext keys in `KEY.txt`, `OAI_CONFIG_LIST`, `config_api_keys`
- `D:/my-fin-project/...` output paths hardcoded in two modules
- `C:/Users/<user>/...` absolute paths in `analyst.py` and `debug.py`

The `D:` paths pointed at a drive that did not exist on the machine the code
was later run from, so those entry points could not execute at all.

`_load_dotenv` is written out by hand rather than imported from `python-dotenv`
so that a fresh checkout with no dependencies installed still runs
`python run.py doctor` and gets a real diagnostic instead of an ImportError.

Credentials default to `""`, never `None`, so callers can test truthiness and
skip a provider they have no key for. `require()` reports every missing
credential at once — discovering them one exception at a time costs a full run
per key.

## core/cache.py

On-disk JSON cache keyed by namespace, ported from a sibling project along
with two Windows failure modes it encodes:

**Atomic replace fails while any process holds a handle.** A `git add` of the
cache, an editor, or an antivirus scan is enough. This killed a five-hour
enrichment run at 51.9% complete: the flush raised `PermissionError` out of a
worker pool and took the process with it. The fix is a short retry loop
(5 attempts, 0.4s linear backoff) plus the rule that **a flush which still
cannot land must never be fatal** — the in-memory data is intact and the next
flush writes it.

**The scratch filename must carry the pid.** A fixed `.json.tmp` is shared by
every process using that namespace, so two concurrent passes race: one
replaces the file, the other's replace raises `FileNotFoundError` because its
own scratch file is gone. That killed a social-links pass at 41,400 of 77,818
lookups.

In this project the cache exists to stop re-billing API calls: an LLM report
or a news fetch that already succeeded is never paid for twice across runs.

## core/concurrent_fetch.py

Stall-guarded thread pool. The guard is **throughput-based, not per-item**:
`concurrent.futures.wait` returns whenever anything completes; if *nothing*
completes within `stall_timeout`, everything still in flight is assumed stuck,
the pool is abandoned with `shutdown(wait=False, cancel_futures=True)`, and the
run moves on. Items that never completed are returned to the caller and, being
un-cached, are simply re-fetched next run.

This matters more here than in the project it came from, because the items are
billed LLM calls. A batch of ten models where one provider hangs used to mean
the whole benchmark produced nothing — and the nine models that did answer
were paid for and thrown away.

`SKIP` is the sentinel for "do not cache anything for this outcome".

## core/llm.py

One client for every provider. It replaced two parallel implementations of the
same thing: a `ModelAPI` class that nothing imported, and five
`_call_<provider>` functions each re-reading `os.getenv` in its own body with
its own usage-extraction logic.

**Caching is by outcome.** A provider with no key is not an exception — the
result carries `skipped=True` so a ten-model run with six keys produces six
rows rather than dying on the seventh. A transport failure and an empty model
response are distinguished, because they look identical from a return value
and caching the first as if it were the second freezes an outage into
permanent empty data. Only `ok` results may be cached.

### Provider asymmetry — read before comparing models on cost or latency

The published benchmark did not put providers on equal footing:

- **Web search.** `use_web_search=True` was passed only down the OpenAI branch;
  every other provider was hardcoded to `False`. The hosted web-search payload
  inflates input tokens roughly 60x — back-solving the published Table 5 gives
  ~368–4493 input tokens for OpenAI models against ~73–85 for the others. That
  is the tool, not the model.
- **Output caps.** OpenAI uncapped; Anthropic/xAI/Llama 1024; DeepSeek 1000.
  DeepSeek-R1's published highest-latency result was generated against its own
  truncation limit.

`max_tokens` is now one explicit argument for every provider, and every result
records `web_search_used` and `max_tokens_used`, so an asymmetric comparison is
visible in the output table rather than buried in a branch. Making the
conditions uniform is a decision for the experiment, not something this layer
does silently — but it can no longer happen unnoticed.

## core/pricing.py

Single pricing table. Three disagreeing tables existed before, and two real
costing bugs came out of the split:

1. The analyst's table had **no entry** for grok-4, o3, o4-mini, the gpt-5
   family or Llama — yet the batch runner billed exactly those models. The
   lookup returned `None` and the cost function then returned `0.0`. Every one
   of those models was recorded as free whenever AutoGen did not report a cost.
2. The lookup fell back to **substring matching**, so `"gpt-5"` matched a
   `"gpt-5.2"` entry ($1.75/$14.00) instead of gpt-5's own $1.25/$10.00.

Substring fallback is gone. A model either has an entry or is reported as
unpriced, loudly.

**An unknown cost is `None`, never `0.0`.** A zero silently pollutes a
cost-per-token comparison; a null shows up as a gap. In a study whose subject
is cost-performance trade-offs, an unpriced model rendered as free would look
like the cheapest model in the field.

Neither bug reached a published figure — all ten Table 2 costs reconstruct to
the microdollar from the AutoGen-reported path.

`load_pricing_overrides` writes no per-run snapshot file. The old
implementation did, and eight `effective_pricing_*.json` accumulated from a
single afternoon. The pricing that actually billed a result now travels inside
that result.

## core/market.py

Merges two independent Polygon clients: the analyst's bare `requests.get` with
no retry policy (a single 429 during a ten-model batch killed the run and lost
every model already billed), and the benchmark's pooled session with a proper
`Retry` policy and parquet cache that only handled close prices.

Everything now shares one session, one retry policy honouring `Retry-After`,
and one cache layer. News is cached so a batch running ten models against one
ticker fetches that ticker's news once rather than ten times. Only a
successful non-empty fetch is cached — a failure must stay un-cached so the
next run retries rather than inheriting the outage.

Functions return `None` rather than raising: a data gap should skip one
model/ticker pair, not abort the batch.

## analyst/snapshot.py

The indicator set — RSI(14), MA20/MA50, 5- and 20-day returns, annualised
volatility, max drawdown, ATR(14), 20-day VWAP, MA-crossover trend label — is
unchanged from the published implementation. **The published cost-performance
tables were produced with exactly this input; do not alter these formulas
without noting the change against the paper.**

`compute_snapshot` returns `None` instead of raising on an unusable frame. The
original raised `ValueError` here and took a whole ten-model run down with it.

## analyst/evaluation.py

Implements the four reliability metrics the paper defines. **The formulas are
deliberately unchanged from the published implementation.** Any change makes
the code disagree with a published paper.

- `s1 = max(supports)` and `r1 = min(resistances)` are the *nearest* levels —
  the tightest bracket around the current price is the strictest test.
- The paper's "one-week horizon" is 5 trading days. The 15-calendar-day request
  window exists only to guarantee 5 trading days survive weekends and holidays;
  only the first five rows are scored.

### The MAE naming defect

The paper expands MAE as "mean absolute error". The quantity is neither a mean
nor an absolute value — it is a signed difference of two percentages, and the
paper's own Table 3 reports −1.42% for two models, which no absolute error can
be. The formula is correct and the paper's sign-convention sentence describes
it correctly; only the expansion of the acronym is wrong. **"Maximum Adverse
Excursion margin"** fits the same acronym and the actual computation.

Secondary: `actual_max_drawdown` is measured from the report-date close to the
window low, not peak-to-trough. A reader implementing "maximum drawdown over
the subsequent five trading days" literally will get a different number.

## analyst/agent.py

The AutoGen config list is built from environment variables instead of being
read from a plaintext file that carried real API keys into git history.

Reports are cached by `(ticker, end date, model, prompt version)`. Bump
`PROMPT_VERSION` when the prompt changes, so cached reports from an older
prompt are not silently mixed into a new run.

**Token estimation is labelled.** Providers AutoGen does not instrument report
zero tokens. The published implementation silently substituted a
characters-over-four estimate, which then flowed into the cost column
indistinguishably from a real count. The estimate is kept — dropping it would
change which models have cost figures at all — but the row now carries
`token_source="estimated_chars_div_4"`.

`code_execution_config` is left as published. The agent only needs the
registered news tool, so local execution could be turned off as a hardening
step, but that would change the agent's behaviour relative to the runs behind
the paper's tables. It stays opt-in rather than silently flipped.

## portfolio/generate.py

The parsing side is unchanged because it earns its keep: models answer this
prompt with a bare JSON array, a fenced block, one object per line, or prose
with the array buried in it. `robust_json_parse` handles all four, and a regex
fallback catches a model that ignores JSON entirely.

## portfolio/benchmark.py

Generation runs concurrently under the stall guard. It used to be a serial
loop, so one hanging provider stalled every model behind it and an exception in
any single model aborted the sweep after earlier models had been paid for.

**A model with no price entry reports cost as NaN, not 0.0.** The old
`float(est_cost or 0.0)` turned an unknown cost into a free one.

`sharpe_per_1k_tokens` and `return_per_1k_tokens` were corrected: the published
code divided by `usd_per_1k_tokens`, yielding Sharpe per dollar-per-1k-tokens
rather than per 1k tokens as the prose describes. Neither column appears in any
published table, so the correction changes no reported number. The old values
are kept alongside as `*_per_usd_per_1k` so an old CSV still reconciles.

## Reproducing the published Table 4

Table 4's six metric columns are a deterministic function of (ticker list,
price window) and reproduce to the digit — 84 of 84 cells at 4 decimal places.
See `data/provenance/README.md`.

**Use split-adjusted close, not dividend-adjusted close.** Polygon's
`adjusted=true` adjusts for splits only; Yahoo's `adjclose` additionally
reinvests dividends. Using `adjclose` adds a full dividend yield to every
annualised return — about +1.8 to +2.4 percentage points on a large-cap
universe — and inflates Sharpe by roughly 10%, while leaving volatility,
drawdown, VaR and CVaR almost untouched. That asymmetric signature is exactly
what a dividend-convention mismatch looks like.
