import os
from pathlib import Path

# Base directory (change one place to move storage)
BASE_DIR = Path(r"D:/my-fin-project/data")

# Subdirectories
TXT_SAVE_DIR   = BASE_DIR / "txt"
JSON_SAVE_DIR  = BASE_DIR / "json"
CSV_SAVE_DIR   = BASE_DIR / "csv"
CHART_SAVE_DIR = BASE_DIR / "charts"
CONFIG_DIR     = BASE_DIR / "configs"
PER_TICKER_DIR = CSV_SAVE_DIR / "per_ticker"

# Auto-create folders
for p in [TXT_SAVE_DIR, JSON_SAVE_DIR, CSV_SAVE_DIR, CHART_SAVE_DIR, CONFIG_DIR, PER_TICKER_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# API keys from environment variables
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
POLYGON_API_KEY  = os.getenv("POLYGON_API_KEY", "")

# Optional: other providers for unified API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
XAI_API_KEY       = os.getenv("XAI_API_KEY", "")
LLAMA_API_KEY     = os.getenv("LLAMA_API_KEY", "")  # OpenRouter or TogetherAI

# Pricing config sources
MODEL_PRICING_JSON = os.getenv("MODEL_PRICING_JSON", str(CONFIG_DIR / "model_pricing.json"))
MODEL_PRICING_URL  = os.getenv("MODEL_PRICING_URL", "")  # optional remote JSON

# Default static pricing (fallback)
DEFAULT_PRICE_PER_MILLION = {
    "gpt-5":       {"input": 1.25, "output": 10.00},
    "gpt-5-mini":  {"input": 0.25, "output": 2.00},
    "gpt-5-nano":  {"input": 0.05, "output": 0.40},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o3":          {"input": 2.00, "output": 8.00},
    "o4-mini":     {"input": 1.10, "output": 4.40},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "grok-4": {"input": 3.0, "output": 15.0},
    "meta-llama/llama-4-maverick": {"input": 0.15, "output": 0.60},
    # Add third-party models here if you want cost estimates via unified API
    # Example keys could be: "claude-sonnet-4-20250514", "grok-4", "meta-llama/llama-4-maverick"
}

# Default DeepSeek tiered pricing per 1M tokens
DEFAULT_DEEPSEEK_PRICING = {
    "standard": {
        "deepseek-chat":     {"input_hit": 0.07,  "input_miss": 0.27, "output": 1.10},
        "deepseek-reasoner": {"input_hit": 0.14, "input_miss": 0.55, "output": 2.19},
    },
    "discount": {
        "deepseek-chat":     {"input_hit": 0.035, "input_miss": 0.135, "output": 0.550},
        "deepseek-reasoner": {"input_hit": 0.035, "input_miss": 0.135, "output": 0.550},
    }
}

# Polygon endpoints
BASE_URL = (
    "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/"
    "{multiplier}/{timespan}/{from_}/{to}"
)
REF_TICKER_URL = "https://api.polygon.io/v3/reference/tickers"
