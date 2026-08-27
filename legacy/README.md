# Legacy entry points

These are the scripts and notebooks as they stood when the paper was written.
They are kept for provenance — a reader tracing a published number back to the
code that produced it should be able to see the original — but they are **not**
the supported way to run anything. Use `python run.py` at the repo root.

What replaced each file:

| Legacy file | Replaced by |
|---|---|
| `analyst.py` | `analyst/` package + `python run.py analyst` |
| `benchmark.py` | `portfolio/benchmark.py` + `python run.py portfolio` |
| `model_portfolio_benchmark.py` | same as above (this file was a whole second copy of it) |
| `debug.py` | `python run.py doctor` |
| `forcasting.py` | standalone experiment, not wired into the paper pipeline |
| `deepseek_r1_smoketest.py` | `python run.py doctor` covers provider reachability |
| `notebooks/*.ipynb` | exploratory work; the `.py` pipeline is authoritative |

## Why they no longer run as-is

- **Hardcoded absolute paths.** `model_portfolio_benchmark.py` and
  `forcasting.py` write to `D:/my-fin-project/...`, a drive that does not exist
  on the current machine. `analyst.py` and `debug.py` hardcode
  `C:/Users/bangc/one-person-unicorn-infra/...`, so they only ever worked from
  one directory on one computer.
- **Credential files that are gone.** `analyst.py` and `forcasting.py` load
  `OAI_CONFIG_LIST` and `config_api_keys`, which carried plaintext API keys and
  have been removed from the repo. Everything now reads `.env`.
- **Notebook keys.** The three notebooks that hardcoded a Polygon and an OpenAI
  key as `os.getenv(..., "<literal key>")` defaults have had those defaults
  replaced with `""`. The notebooks are otherwise untouched, including outputs.

## Known defects preserved here

Two costing bugs live in `analyst.py` and are fixed in `core/pricing.py`. They
are left intact here so the original behaviour stays inspectable:

1. `PRICING_REGISTRY` (line ~14) has no entry for `grok-4`, `o3`, `o4-mini`,
   the `gpt-5` family, or the Llama model — yet the batch runner at the bottom
   of the file bills exactly those models. `_resolve_pricing` returns `None`
   and `calculate_cost` then returns `0.0`, recording those models as free
   whenever AutoGen did not report a cost of its own.
2. `_resolve_pricing` falls back to substring matching, so `"gpt-5"` matches
   the `"gpt-5.2"` entry ($1.75/$14.00 per 1M) instead of gpt-5's own
   $1.25/$10.00.

**Neither bug reached a published figure.** All ten cost values in the paper's
Table 2 reconstruct to the microdollar from the `price` fields in the old
`OAI_CONFIG_LIST`, which means the AutoGen-reported cost path was taken every
time and `PRICING_REGISTRY` was never consulted. It was dead code in the
published path — but it was one fallthrough away from billing `gpt-4o-mini` at
`gpt-4o` rates (16x too high). Fixed in `core/pricing.py`, where an unpriced
model returns `None` and says so instead of returning `0.0`.
