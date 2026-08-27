from __future__ import annotations

import json
from datetime import datetime, time as dtime, timezone
from typing import Optional

import requests

from .config import (
    DEEPSEEK_USE_DISCOUNT_WINDOW,
    MODEL_PRICING_JSON,
    MODEL_PRICING_URL,
)
from .utils import println

PRICE_PER_MILLION: dict[str, dict[str, float]] = {
    "gpt-5":       {"input": 1.25, "output": 10.00, "cached_input": 0.125},
    "gpt-5-mini":  {"input": 0.25, "output": 2.00,  "cached_input": 0.025},
    "gpt-5-nano":  {"input": 0.05, "output": 0.40,  "cached_input": 0.005},
    "gpt-4o":      {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60,  "cached_input": 0.075},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4":       {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o3":          {"input": 2.00, "output": 8.00},
    "o4-mini":     {"input": 1.10, "output": 4.40},
    "claude-sonnet-4-20250514":  {"input": 3.00, "output": 15.00},
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    "grok-4": {"input": 3.00, "output": 15.00},
    "meta-llama/llama-4-maverick": {"input": 0.15, "output": 0.60},
}

DEEPSEEK_PRICING: dict[str, dict[str, dict[str, float]]] = {
    "standard": {
        "deepseek-chat":     {"input_hit": 0.07, "input_miss": 0.27, "output": 1.10},
        "deepseek-reasoner": {"input_hit": 0.14, "input_miss": 0.55, "output": 2.19},
    },
    "discount": {
        "deepseek-chat":     {"input_hit": 0.035, "input_miss": 0.135, "output": 0.550},
        "deepseek-reasoner": {"input_hit": 0.035, "input_miss": 0.135, "output": 0.550},
    },
}

_unpriced_seen: set[str] = set()


def _merge(base: dict, override: dict) -> dict:
    out = base.copy()
    for k, v in (override or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_pricing_overrides() -> None:
    global PRICE_PER_MILLION, DEEPSEEK_PRICING
    cfg: dict = {}
    if MODEL_PRICING_JSON:
        try:
            cfg = _merge(cfg, json.loads(open(MODEL_PRICING_JSON, encoding="utf-8").read()))
        except (OSError, json.JSONDecodeError) as e:
            println(f"[pricing] ignoring MODEL_PRICING_JSON: {type(e).__name__}")
    if MODEL_PRICING_URL:
        try:
            r = requests.get(MODEL_PRICING_URL, timeout=10)
            r.raise_for_status()
            cfg = _merge(cfg, r.json())
        except Exception as e:
            println(f"[pricing] ignoring MODEL_PRICING_URL: {type(e).__name__}")
    if cfg.get("PRICE_PER_MILLION"):
        PRICE_PER_MILLION = _merge(PRICE_PER_MILLION, cfg["PRICE_PER_MILLION"])
    if cfg.get("DEEPSEEK_PRICING"):
        DEEPSEEK_PRICING = _merge(DEEPSEEK_PRICING, cfg["DEEPSEEK_PRICING"])


def is_deepseek_discount_window(now_utc: Optional[datetime] = None) -> bool:
    if not DEEPSEEK_USE_DISCOUNT_WINDOW:
        return False
    t = (now_utc or datetime.now(timezone.utc)).time()
    return t >= dtime(16, 30) or t < dtime(0, 30)


def deepseek_cost(model: str, input_tokens: int, output_tokens: int,
                  *, cache_hit: bool = False) -> Optional[float]:
    tier = "discount" if is_deepseek_discount_window() else "standard"
    table = DEEPSEEK_PRICING.get(tier, {}).get(model)
    if not table:
        return _unpriced(model)
    in_price = table["input_hit" if cache_hit else "input_miss"]
    return (input_tokens / 1e6) * in_price + (output_tokens / 1e6) * table["output"]


def _unpriced(model: str) -> None:
    if model not in _unpriced_seen:
        _unpriced_seen.add(model)
        println(f"[pricing] no price entry for {model!r}; cost recorded as "
                f"unknown (null), not zero. Add it to core/pricing.py or "
                f"MODEL_PRICING_JSON.")
    return None


def estimate_cost(model: str, input_tokens: int, output_tokens: int,
                  *, cached_input_tokens: int = 0,
                  cache_hit: bool = False) -> Optional[float]:
    if model.startswith("deepseek-"):
        return deepseek_cost(model, input_tokens, output_tokens, cache_hit=cache_hit)

    price = PRICE_PER_MILLION.get(model)
    if not price:
        return _unpriced(model)

    cached = max(0, int(cached_input_tokens))
    billed_input = max(0, int(input_tokens) - cached)
    cost = (billed_input / 1e6) * price["input"] + (int(output_tokens) / 1e6) * price["output"]
    if cached and price.get("cached_input") is not None:
        cost += (cached / 1e6) * price["cached_input"]
    return round(cost, 6)


def price_of(model: str) -> Optional[dict]:
    if model.startswith("deepseek-"):
        tier = "discount" if is_deepseek_discount_window() else "standard"
        entry = DEEPSEEK_PRICING.get(tier, {}).get(model)
        return {"tier": tier, **entry} if entry else None
    return PRICE_PER_MILLION.get(model)
