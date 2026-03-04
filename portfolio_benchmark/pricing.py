import json
from datetime import datetime, timezone, time as dtime
from typing import Optional, Dict
import requests

from .config import (
    DEFAULT_PRICE_PER_MILLION,
    DEFAULT_DEEPSEEK_PRICING,
    MODEL_PRICING_JSON,
    MODEL_PRICING_URL,
    CONFIG_DIR,
)
from .utils import println

PRICE_PER_MILLION: Dict[str, dict] = DEFAULT_PRICE_PER_MILLION.copy()
DEEPSEEK_PRICING: Dict[str, dict] = DEFAULT_DEEPSEEK_PRICING.copy()

def _safe_load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _safe_load_json_from_url(url: str, timeout: int = 10) -> Optional[dict]:
    try:
        if not url:
            return None
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        println(f"Failed to fetch pricing JSON from URL: {e}")
        return None

def _merge(base: dict, override: dict) -> dict:
    out = base.copy()
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out

def load_pricing_config():
    global PRICE_PER_MILLION, DEEPSEEK_PRICING
    file_cfg = _safe_load_json(MODEL_PRICING_JSON) or {}
    url_cfg  = _safe_load_json_from_url(MODEL_PRICING_URL) or {}

    if file_cfg.get("PRICE_PER_MILLION") or url_cfg.get("PRICE_PER_MILLION"):
        PRICE_PER_MILLION = _merge(DEFAULT_PRICE_PER_MILLION, file_cfg.get("PRICE_PER_MILLION", {}))
        PRICE_PER_MILLION = _merge(PRICE_PER_MILLION, url_cfg.get("PRICE_PER_MILLION", {}))

    if file_cfg.get("DEEPSEEK_PRICING") or url_cfg.get("DEEPSEEK_PRICING"):
        DEEPSEEK_PRICING = _merge(DEFAULT_DEEPSEEK_PRICING, file_cfg.get("DEEPSEEK_PRICING", {}))
        DEEPSEEK_PRICING = _merge(DEEPSEEK_PRICING, url_cfg.get("DEEPSEEK_PRICING", {}))

    snapshot_path = CONFIG_DIR / f"effective_pricing_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    try:
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump({
                "PRICE_PER_MILLION": PRICE_PER_MILLION,
                "DEEPSEEK_PRICING": DEEPSEEK_PRICING,
                "source_file": MODEL_PRICING_JSON,
                "source_url": MODEL_PRICING_URL,
                "ts": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def is_deepseek_discount_window_utc(now_utc=None) -> bool:
    now_utc = now_utc or datetime.now(timezone.utc)
    t = now_utc.time()
    return (t >= dtime(16, 30)) or (t < dtime(0, 30))

def deepseek_cost(model: str, input_tokens: int, output_tokens: int, cache_hit: bool) -> float:
    tier = "discount" if is_deepseek_discount_window_utc() else "standard"
    if model not in DEEPSEEK_PRICING.get(tier, {}):
        return 0.0
    table = DEEPSEEK_PRICING[tier][model]
    in_price  = table.get("input_hit" if cache_hit else "input_miss", 0.0)
    out_price = table.get("output", 0.0)
    return (input_tokens / 1_000_000.0) * in_price + (output_tokens / 1_000_000.0) * out_price
