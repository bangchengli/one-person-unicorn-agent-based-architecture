from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.split("#", 1)[0].strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "").strip()
XAI_API_KEY: str = os.getenv("XAI_API_KEY", "").strip()
LLAMA_API_KEY: str = os.getenv("LLAMA_API_KEY", "").strip()
POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "").strip()

PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "llama": "LLAMA_API_KEY",
}


def provider_key(provider: str) -> str:
    return os.getenv(PROVIDER_ENV.get(provider.lower(), ""), "").strip()


def configured_providers() -> list[str]:
    return [p for p in PROVIDER_ENV if provider_key(p)]


def require(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n, "").strip()]
    if missing:
        raise SystemExit(
            "Missing required credentials: " + ", ".join(missing) +
            "\nCopy .env.example to .env and fill them in."
        )


DATA_DIR: Path = Path(os.getenv("DATA_DIR", "").strip() or (ROOT / "data"))
CACHE_DIR: Path = DATA_DIR / "cache"
PRICE_CACHE_DIR: Path = DATA_DIR / "prices"
ANALYST_OUT_DIR: Path = DATA_DIR / "analyst_outputs"
PORTFOLIO_OUT_DIR: Path = DATA_DIR / "portfolio_outputs"
CSV_DIR: Path = ROOT / "csv"
CHART_DIR: Path = ROOT / "charts"
CONFIG_DIR: Path = ROOT / "configs"


def ensure_dirs() -> None:
    for p in (DATA_DIR, CACHE_DIR, PRICE_CACHE_DIR, ANALYST_OUT_DIR,
              PORTFOLIO_OUT_DIR, CSV_DIR, CHART_DIR, CONFIG_DIR):
        p.mkdir(parents=True, exist_ok=True)


MAX_WORKERS: int = _env_int("MAX_WORKERS", 8)
STALL_TIMEOUT: float = _env_float("STALL_TIMEOUT", 180.0)
HTTP_TIMEOUT: float = _env_float("HTTP_TIMEOUT", 30.0)
LLM_TIMEOUT: float = _env_float("LLM_TIMEOUT", 240.0)
DEEPSEEK_USE_DISCOUNT_WINDOW: bool = _env_flag("DEEPSEEK_USE_DISCOUNT_WINDOW", True)

MODEL_PRICING_JSON: str = os.getenv("MODEL_PRICING_JSON", "").strip()
MODEL_PRICING_URL: str = os.getenv("MODEL_PRICING_URL", "").strip()

POLYGON_AGGS_URL = ("https://api.polygon.io/v2/aggs/ticker/{ticker}/range/"
                    "{multiplier}/{timespan}/{start}/{end}")
POLYGON_NEWS_URL = "https://api.polygon.io/v2/reference/news"
POLYGON_REF_TICKER_URL = "https://api.polygon.io/v3/reference/tickers"
