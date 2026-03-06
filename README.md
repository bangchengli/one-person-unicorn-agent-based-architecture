# Portfolio Benchmark (Multi-Model Integrated)

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

# Financial Analyst
Provide price and market news scraping and analysis, and an evaluation report of the performance of various models.
