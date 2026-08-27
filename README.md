# One-Person Unicorn: An Agent-Based Architecture for Automated Financial Services

Companion code for the paper of the same name.

**Bangcheng Li** ([0009-0001-7959-1362](https://orcid.org/0009-0001-7959-1362)) — Carroll School of Management, Boston College<br>
*International Journal of Information Systems and Social Change*, 17(1), 2026

[**Read the paper**](https://doi.org/10.4018/IJISSC.409374) · [IGI Global](https://www.igi-global.com/article/one-person-unicorn/409374) · Open Access under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)

An agent-based architecture for automated financial services, evaluated across
GPT/o-series, Claude, Grok, DeepSeek and Llama on cost, latency and output
quality.

<details>
<summary><b>Cite this work</b></summary>

```bibtex
@article{li2026onepersonunicorn,
  author  = {Li, Bangcheng},
  title   = {One-Person Unicorn: An Agent-Based Architecture for Automated Financial Services},
  journal = {International Journal of Information Systems and Social Change},
  volume  = {17},
  number  = {1},
  year    = {2026},
  doi     = {10.4018/IJISSC.409374}
}
```

</details>

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
| `POLYGON_API_KEY` | [Polygon.io](https://polygon.io) | Main market data source | polygon.io → Dashboard → API Keys |

### Optional — one per model provider

Each is optional and independent. You may also leverage more models from other providers.

| Variable | Provider | Models it enables | Where to get it |
|---|---|---|---|
| `OPENAI_API_KEY` | [OpenAI](https://platform.openai.com) | `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4o`, `gpt-4o-mini`, `o3`, `o4-mini` | platform.openai.com → API keys |
| `ANTHROPIC_API_KEY` | [Anthropic](https://console.anthropic.com) | `claude-sonnet-4-20250514` | console.anthropic.com → API Keys |
| `XAI_API_KEY` | [xAI](https://console.x.ai) | `grok-4` | console.x.ai |
| `DEEPSEEK_API_KEY` | [DeepSeek](https://platform.deepseek.com) | `deepseek-chat`, `deepseek-reasoner` | platform.deepseek.com |
| `LLAMA_API_KEY` | [OpenRouter](https://openrouter.ai) | `meta-llama/llama-4-maverick` | openrouter.ai → Keys |

### Cost note

Every model call is billed by its provider. A testing of the architecture may generate costs.

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

The source files carry no comments. Information regarding files are located in [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md).
Read the relevant section before changing a module.

## License

Code released under the MIT License. The article itself is Open Access under
[CC BY 4.0](http://creativecommons.org/licenses/by/4.0/).
