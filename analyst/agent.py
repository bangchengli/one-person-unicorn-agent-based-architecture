from __future__ import annotations

import json
import time
from typing import Any, Optional

from core.cache import JSONCache
from core.config import LLM_TIMEOUT, provider_key
from core.llm import provider_for
from core.market import fetch_news, fetch_ohlcv
from core.pricing import estimate_cost, price_of
from core.utils import parse_json_report, println

from .snapshot import compute_snapshot

PROMPT_VERSION = "v1"

_report_cache = JSONCache("analyst_reports")

_PROVIDER_AUTOGEN = {
    "openai": {},
    "deepseek": {"api_type": "openai", "base_url": "https://api.deepseek.com"},
    "xai": {"api_type": "openai", "base_url": "https://api.x.ai/v1"},
    "llama": {"api_type": "openai", "base_url": "https://openrouter.ai/api/v1"},
    "anthropic": {"api_type": "anthropic"},
}


def build_config_list(model: str) -> list[dict]:
    prov = provider_for(model)
    key = provider_key(prov)
    if not key:
        return []
    entry: dict[str, Any] = {"model": model, "api_key": key}
    entry.update(_PROVIDER_AUTOGEN.get(prov, {}))
    return [entry]


def _usage_bucket(bucket: dict, model_hint: str = "") -> tuple[int, int, Optional[float]]:
    if not isinstance(bucket, dict) or not bucket:
        return 0, 0, None
    total_cost = bucket.get("total_cost")
    model_keys = [k for k in bucket if k != "total_cost"]

    for k in ([model_hint] if model_hint in bucket else []) + \
             [k for k in model_keys if model_hint and (model_hint in k or k in model_hint)]:
        d = bucket.get(k) or {}
        if isinstance(d, dict):
            cost = d.get("cost", total_cost)
            return (int(d.get("prompt_tokens") or 0),
                    int(d.get("completion_tokens") or 0),
                    float(cost) if cost is not None else None)

    prompt = completion = 0
    cost_sum, counted = 0.0, 0
    for k in model_keys:
        d = bucket.get(k) or {}
        prompt += int(d.get("prompt_tokens") or 0)
        completion += int(d.get("completion_tokens") or 0)
        if d.get("cost") is not None:
            cost_sum += float(d["cost"])
            counted += 1
    resolved = cost_sum if counted else total_cost
    return prompt, completion, float(resolved) if resolved is not None else None


def extract_usage(chat_result, model_hint: str = "") -> dict:
    cost = getattr(chat_result, "cost", None)
    if not isinstance(cost, dict) or not cost:
        return {"input_tokens": 0, "output_tokens": 0,
                "cached_input_tokens": 0, "cached_output_tokens": 0,
                "autogen_reported_cost": None}

    incl_p, incl_c, _ = _usage_bucket(cost.get("usage_including_cached_inference") or {}, model_hint)
    excl_p, excl_c, excl_cost = _usage_bucket(cost.get("usage_excluding_cached_inference") or {}, model_hint)
    return {
        "input_tokens": excl_p,
        "output_tokens": excl_c,
        "cached_input_tokens": max(0, incl_p - excl_p),
        "cached_output_tokens": max(0, incl_c - excl_c),
        "autogen_reported_cost": excl_cost,
    }


def _build_prompt(ticker: str, as_of: str, snapshot: dict, model: str) -> str:
    return f"""
You are a trader-oriented financial analyst.
Ticker: {ticker}
Date: {as_of}

Technical Snapshot:
{json.dumps(snapshot, indent=2)}

Task:
1. Call 'get_market_news' to see what is driving the market currently.
2. Analyze the correlation between the news sentiment and the technical trend.
3. Output a SINGLE JSON object matching this schema exactly (No Markdown):
{{
  "meta": {{ "ticker": "...", "as_of": "...", "model": "{model}" }},
  "market_snapshot": {{...}},
  "news_analysis": [ {{ "headline": "...", "sentiment": "...", "impact": "..." }} ],
  "forecast": {{ "horizon": "1w", "direction": "...", "reasoning": "..." }},
  "risk_levels": {{ "support": [...], "resistance": [...] }},
  "full_assessment": "..."
}}
"""


def run_analyst(ticker: str, start: str, end: str, model: str, *,
                use_cache: bool = True) -> dict:
    cache_key = f"{ticker.upper()}|{end}|{model}|{PROMPT_VERSION}"
    if use_cache and _report_cache.has(cache_key):
        cached = _report_cache.get(cache_key)
        if cached:
            cached.setdefault("_ops", {})["cache_hit"] = True
            return cached

    def _failed(step: str, detail: str) -> dict:
        println(f"[analyst] {model}: failed at {step}: {detail}")
        return {"_ops": {"status": "failed", "model": model, "ticker": ticker,
                         "failed_step": step, "error": detail}}

    config_list = build_config_list(model)
    if not config_list:
        return {"_ops": {"status": "skipped", "model": model, "ticker": ticker,
                         "error": f"no credential for provider "
                                  f"{provider_for(model)!r}"}}

    prices = fetch_ohlcv(ticker, start, end)
    if prices is None or prices.empty:
        return _failed("fetch prices", f"no Polygon bars for {ticker} {start}..{end}")

    snapshot = compute_snapshot(prices)
    if snapshot is None:
        return _failed("compute snapshot", "price frame unusable")
    snapshot = json.loads(json.dumps(snapshot).replace("NaN", "null"))

    try:
        import autogen
    except ImportError:
        return _failed("import autogen", "pyautogen is not installed "
                                         "(pip install -r requirements.txt)")

    as_of = str(prices.index.max().date())
    prompt = _build_prompt(ticker, as_of, snapshot, model)

    try:
        assistant = autogen.AssistantAgent(
            name="Market_Analyst",
            llm_config={"config_list": config_list, "timeout": LLM_TIMEOUT,
                        "cache_seed": None},
            system_message=("You are a professional financial analyst. Use the "
                            "available tools to fetch REAL news, then correlate "
                            "it with the technical snapshot to generate a JSON "
                            "report."),
        )
        user_proxy = autogen.UserProxyAgent(
            name="User_Executor",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=6,
            is_termination_msg=lambda x: "full_assessment" in (x.get("content") or ""),
            code_execution_config={"work_dir": "coding", "use_docker": False},
        )

        def get_market_news(ticker_symbol: str) -> str:
            return json.dumps(fetch_news(ticker_symbol, as_of=end))

        autogen.register_function(
            get_market_news, caller=assistant, executor=user_proxy,
            name="get_market_news",
            description="Fetch REAL recent financial news for a specific "
                        "ticker from Polygon.io.")

        t0 = time.time()
        chat_result = user_proxy.initiate_chat(assistant, message=prompt)
        latency = time.time() - t0
    except Exception as e:
        return _failed("agent conversation", f"{type(e).__name__}: {str(e)[:200]}")

    usage = extract_usage(chat_result, model_hint=model)

    token_source = "provider"
    if usage["input_tokens"] == 0 and usage["output_tokens"] == 0:
        in_chars = out_chars = 0
        for msg in getattr(chat_result, "chat_history", []) or []:
            content = str(msg.get("content") or "")
            if msg.get("role") == "assistant":
                out_chars += len(content)
            else:
                in_chars += len(content)
        in_chars = in_chars or len(prompt)
        usage["input_tokens"] = max(1, in_chars // 4)
        usage["output_tokens"] = max(1, out_chars // 4)
        token_source = "estimated_chars_div_4"

    report = None
    for msg in reversed(getattr(chat_result, "chat_history", []) or []):
        report = parse_json_report(str(msg.get("content") or ""),
                                   require_key="full_assessment")
        if report:
            break
    if not report:
        return _failed("parse report", "no JSON object containing "
                                       "'full_assessment' in the transcript")

    cost = usage["autogen_reported_cost"]
    if cost is None:
        cost = estimate_cost(model, usage["input_tokens"], usage["output_tokens"],
                             cached_input_tokens=usage["cached_input_tokens"])

    report["_ops"] = {
        "status": "success",
        "model": model,
        "ticker": ticker,
        "as_of": as_of,
        "prompt_version": PROMPT_VERSION,
        "llm_latency_sec": float(latency),
        "cost_source": "autogen" if usage["autogen_reported_cost"] is not None else "pricing_table",
        "token_source": token_source,
        "cost_usd": cost,
        "pricing_used": price_of(model),
        "cache_hit": False,
        **{k: usage[k] for k in ("input_tokens", "output_tokens",
                                 "cached_input_tokens", "cached_output_tokens")},
    }

    _report_cache.set(cache_key, report)
    _report_cache.flush()
    return report
