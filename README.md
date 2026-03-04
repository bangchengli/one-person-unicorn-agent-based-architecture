# Portfolio Benchmark (Multi-Model Integrated)

English-only comments and output.

## What is new
- Unified API interface to call Claude (Anthropic), Grok (xAI), DeepSeek (R1/V3), and LLaMA (via OpenRouter/TogetherAI)
- Original OpenAI path is kept intact
- Optional cost estimation for unified models if you add them to `configs/model_pricing.json`

## Setup
```bash
pip install openai pandas numpy requests matplotlib PyPortfolioOpt
```

Set environment variables (PowerShell example on Windows):
```powershell
setx OPENAI_API_KEY "your_key"
setx POLYGON_API_KEY "your_key"
setx DEEPSEEK_API_KEY "your_key"       # if you call DeepSeek natively
setx ANTHROPIC_API_KEY "your_key"      # for unified API
setx XAI_API_KEY "your_key"            # for Grok via unified API
setx LLAMA_API_KEY "your_key"          # for LLaMA via OpenRouter/TogetherAI
```

## Run
```bash
python benchmark.py
```

To switch unified provider and model mapping, edit the block in `benchmark.py` under `__main__`.
Cost estimation for unified providers is `0` by default unless you add their model name to `configs/model_pricing.json` under `PRICE_PER_MILLION`.
