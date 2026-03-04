# portfolio_benchmark/llm_portfolio.py
import json
import time
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import requests  # for non-OpenAI providers

from .config import OPENAI_API_KEY, TXT_SAVE_DIR, JSON_SAVE_DIR
from .pricing import load_pricing_config, PRICE_PER_MILLION, deepseek_cost
from .utils import ensure_dirs

# ----------------------------
# Robust JSON parsing helpers
# ----------------------------
def _strip_trailing_commas(s: str) -> str:
    s = s.strip()
    out = []
    stack = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "{[":
            stack.append(ch); out.append(ch)
        elif ch in "}]":
            if out and out[-1] == ",": out.pop()
            if stack: stack.pop()
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip()

def robust_json_parse(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        return [raw]
    text = (raw or "").strip()
    if not text:
        return []
    try:
        obj = json.loads(_strip_trailing_commas(text))
        return obj if isinstance(obj, list) else [obj]
    except Exception:
        pass
    items = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("#"):
            continue
        if s.endswith(","):
            s = s[:-1].rstrip()
        try:
            items.append(json.loads(_strip_trailing_commas(s)))
        except Exception:
            pass
    if items:
        return items
    objs, level, buf = [], 0, []
    for ch in text:
        if ch == "{":
            level += 1
        if level > 0:
            buf.append(ch)
        if ch == "}":
            level -= 1
            if level == 0 and buf:
                try:
                    objs.append(json.loads(_strip_trailing_commas("".join(buf))))
                except Exception:
                    pass
                buf = []
    return objs

# ----------------------------
# Fallback ticker extractor
# ----------------------------
_TICKER_STOP = {"AND","THE","FOR","WITH","FROM","THIS","THAT","YOU","YOUR","A","AN","IN","ON","BY","TO","AS","AT","OF"}
def _extract_tickers_fallback(text: str, n_min: int = 15, n_max: int = 25):
    cands = re.findall(r"\b[A-Z]{1,5}\b", text or "")
    out, seen = [], set()
    for t in cands:
        if t in _TICKER_STOP: continue
        if not t.isalpha():   continue
        if t in seen:         continue
        seen.add(t); out.append(t)
        if len(out) >= n_max: break
    return out

# ----------------------------
# Provider-specific call paths
# ----------------------------
def _call_openai(model: str, prompt: str, use_web_search: bool):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    kwargs = {"model": model, "input": prompt}
    if use_web_search:
        kwargs["tools"] = [{"type": "web_search"}]; kwargs["tool_choice"] = "auto"
    t0 = time.perf_counter()
    resp = client.responses.create(**kwargs)
    latency = time.perf_counter() - t0
    raw = getattr(resp, "output_text", None) or str(resp)
    usage = getattr(resp, "usage", None) or {}
    in_tok  = getattr(usage, "input_tokens",  None) or usage.get("input_tokens", 0)
    out_tok = getattr(usage, "output_tokens", None) or usage.get("output_tokens", 0)
    if in_tok == 0 and hasattr(usage, "input_text_tokens"):
        in_tok = getattr(usage, "input_text_tokens", 0)
    if out_tok == 0 and hasattr(usage, "output_text_tokens"):
        out_tok = getattr(usage, "output_text_tokens", 0)
    return raw, float(latency), int(in_tok or 0), int(out_tok or 0)

def _call_deepseek(model: str, prompt: str):
    # OpenAI SDK with base_url -> DeepSeek
    from openai import OpenAI
    import os
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000
    )
    latency = time.perf_counter() - t0
    content = ""; reasoning = ""
    try:
        ch0 = resp.choices[0]; msg = ch0.message
        content = getattr(msg, "content", "") or (msg.get("content") if isinstance(msg, dict) else "")
        reasoning = getattr(msg, "reasoning_content", "") or (msg.get("reasoning_content") if isinstance(msg, dict) else "")
    except Exception:
        pass
    raw = content if content else reasoning
    in_tok = out_tok = 0
    try:
        u = getattr(resp, "usage", None)
        if isinstance(u, dict):
            in_tok = int(u.get("prompt_tokens") or 0); out_tok = int(u.get("completion_tokens") or 0)
        elif u:
            in_tok = int(getattr(u, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(u, "completion_tokens", 0) or 0)
    except Exception:
        pass
    return raw, float(latency), in_tok, out_tok

def _anthropic_count_tokens(model: str, prompt: str) -> int:
    """Use Anthropic count-tokens endpoint to estimate input tokens."""
    import os, requests
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return 0
    url = "https://api.anthropic.com/v1/messages/count_tokens"
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    try:
        r = requests.post(url, headers=headers, json=data, timeout=20)
        r.raise_for_status()
        j = r.json() or {}
        return int(j.get("input_tokens") or 0)
    except Exception:
        return 0


def _call_anthropic(model: str, prompt: str):
    import os, time, json, requests
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    data = {"model": model, "max_tokens": 1024, "messages": [{"role": "user", "content": prompt}]}
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    j = resp.json()
    raw = ""
    try:
        parts = j.get("content", [])
        texts = [p.get("text") for p in parts if isinstance(p, dict) and "text" in p]
        raw = "\n".join([t for t in texts if t])
    except Exception:
        raw = json.dumps(j, ensure_ascii=False)
    in_tok = out_tok = 0
    usage = j.get("usage", {}) if isinstance(j, dict) else {}
    if usage:
        in_tok = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0
        out_tok = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0
    if int(in_tok or 0) == 0:
        try:
            in_tok = _anthropic_count_tokens(model, prompt)
        except Exception:
            pass
    return raw, float(latency), int(in_tok or 0), int(out_tok or 0)


def _call_llama(model: str, prompt: str):
    import os, time, json, requests
    api_key = os.getenv("LLAMA_API_KEY", "")
    if not api_key:
        raise RuntimeError("LLAMA_API_KEY is not set")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "extra_body": {"usage": {"include": True}}
    }
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    j = resp.json()
    raw = ""
    try:
        raw = j["choices"][0]["message"]["content"]
    except Exception:
        raw = json.dumps(j, ensure_ascii=False)
    in_tok = out_tok = 0
    usage = j.get("usage", {}) if isinstance(j, dict) else {}
    if usage:
        in_tok = usage.get("prompt_tokens", 0) or 0
        out_tok = usage.get("completion_tokens", 0) or 0
    return raw, float(latency), int(in_tok or 0), int(out_tok or 0)


def _call_xai(model: str, prompt: str):
    import os, time, json, requests
    api_key = os.getenv("XAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("XAI_API_KEY is not set")
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024}
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    j = resp.json()
    raw = ""
    try:
        raw = j["choices"][0]["message"]["content"]
    except Exception:
        raw = json.dumps(j, ensure_ascii=False)
    in_tok = out_tok = 0
    usage = j.get("usage", {}) if isinstance(j, dict) else {}
    if usage:
        in_tok = usage.get("prompt_tokens", 0) or 0
        out_tok = usage.get("completion_tokens", 0) or 0
    return raw, float(latency), int(in_tok or 0), int(out_tok or 0)


# ----------------------------
# Public entry (now with provider)
# ----------------------------
def generate_portfolio_with_model(model: str,
                                  n_min: int = 15,
                                  n_max: int = 25,
                                  use_web_search: bool = True,
                                  use_unified_api: bool = False,  # kept for compatibility; not used now
                                  unified_provider: Optional[str] = None,  # kept for compatibility
                                  unified_model: Optional[str] = None,     # kept for compatibility
                                  provider: Optional[str] = None           # NEW: explicit provider
                                  ) -> Tuple[List[str], float, int, int, float]:
    """
    Returns: (tickers, latency_sec, input_tokens, output_tokens, est_cost_usd)
    We always call by provider+model. 'name' is only for display outside this function.
    """
    prompt = (
        "You are an expert portfolio construction advisor. "
        f"Generate a diversified US equities portfolio with {n_min}-{n_max} tickers ONLY. "
        "Respond as a pure JSON array. Each element must be an object with the key 'name' containing the ticker symbol. "
        "No commentary. Example: [{\"name\": \"AAPL\"}, {\"name\": \"MSFT\"}]"
    )

    load_pricing_config()

    # Decide provider route
    prov = (provider or "").lower().strip()
    if not prov and (model.startswith("deepseek-")):
        prov = "deepseek"
    if not prov:
        prov = "openai"  # default

    if prov == "openai":
        raw, latency, in_tok, out_tok = _call_openai(model, prompt, use_web_search)
        price = PRICE_PER_MILLION.get(model, {"input": 0.0, "output": 0.0})
        est_cost = (in_tok / 1_000_000.0) * price.get("input", 0.0) + (out_tok / 1_000_000.0) * price.get("output", 0.0)

    elif prov == "deepseek":
        raw, latency, in_tok, out_tok = _call_deepseek(model, prompt)
        est_cost = deepseek_cost(model, in_tok, out_tok, cache_hit=False)

    elif prov == "anthropic":
        raw, latency, in_tok, out_tok = _call_anthropic(model, prompt)
        price = PRICE_PER_MILLION.get(model, {"input": 0.0, "output": 0.0})
        est_cost = (in_tok / 1_000_000.0) * price.get("input", 0.0) + (out_tok / 1_000_000.0) * price.get("output", 0.0)

    elif prov == "xai":
        raw, latency, in_tok, out_tok = _call_xai(model, prompt)
        price = PRICE_PER_MILLION.get(model, {"input": 0.0, "output": 0.0})
        est_cost = (in_tok / 1_000_000.0) * price.get("input", 0.0) + (out_tok / 1_000_000.0) * price.get("output", 0.0)

    elif prov == "llama":
        raw, latency, in_tok, out_tok = _call_llama(model, prompt)
        price = PRICE_PER_MILLION.get(model, {"input": 0.0, "output": 0.0})
        est_cost = (in_tok / 1_000_000.0) * price.get("input", 0.0) + (out_tok / 1_000_000.0) * price.get("output", 0.0)

    else:
        raise ValueError(f"Unsupported provider: {prov}")

    # Save raw text
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name_tag = (unified_model or model).replace(":", "_")
    txt_file = (TXT_SAVE_DIR / f"{name_tag}_portfolio_{ts}.txt")
    raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(raw_text)

    # Parse tickers
    data = robust_json_parse(raw_text)
    tickers = [x.get("name", "").strip().upper() for x in data if isinstance(x, dict) and x.get("name")]
    tickers = list(dict.fromkeys([t for t in tickers if t]))
    if not tickers:
        tickers = _extract_tickers_fallback(raw_text, n_min=n_min, n_max=n_max)

    # Save parsed JSON
    json_file = (JSON_SAVE_DIR / f"{name_tag}_portfolio_{ts}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump([{"name": t} for t in tickers], f, ensure_ascii=False, indent=2)

    return tickers, float(latency), int(in_tok or 0), int(out_tok or 0), float(est_cost)
