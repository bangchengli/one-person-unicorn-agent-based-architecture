from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from .config import LLM_TIMEOUT, provider_key
from .pricing import estimate_cost, price_of
from .utils import println

_ENDPOINTS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "xai": "https://api.x.ai/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "llama": "https://openrouter.ai/api/v1/chat/completions",
}


@dataclass
class LLMResult:

    provider: str
    model: str
    text: str = ""
    latency_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: Optional[float] = None
    pricing_used: Optional[dict] = None
    error: Optional[str] = None
    skipped: bool = False
    web_search_used: bool = False
    max_tokens_used: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return not self.skipped and self.error is None and bool(self.text)

    def as_row(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "status": "skipped" if self.skipped else ("failed" if self.error else "success"),
            "latency_sec": round(self.latency_sec, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cost_usd": self.cost_usd,
            "web_search_used": self.web_search_used,
            "max_tokens_used": self.max_tokens_used,
            "error": self.error,
        }


def provider_for(model: str) -> str:
    m = model.lower()
    if m.startswith("deepseek-"):
        return "deepseek"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("grok"):
        return "xai"
    if m.startswith("meta-llama/") or "llama" in m:
        return "llama"
    return "openai"


def _usage_from_openai_shape(j: dict) -> tuple[int, int, int]:
    u = j.get("usage") or {}
    in_tok = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
    out_tok = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
    details = u.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0)
    return in_tok, out_tok, cached


def _anthropic_count_tokens(model: str, prompt: str, key: str) -> int:
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages/count_tokens",
            headers={"x-api-key": key, "content-type": "application/json",
                     "anthropic-version": "2023-06-01"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=20)
        r.raise_for_status()
        return int((r.json() or {}).get("input_tokens") or 0)
    except Exception:
        return 0


def call(model: str, prompt: str, *,
         provider: Optional[str] = None,
         max_tokens: int = 1024,
         temperature: float = 0.2,
         use_web_search: bool = False) -> LLMResult:
    prov = (provider or provider_for(model)).lower()
    res = LLMResult(provider=prov, model=model)

    key = provider_key(prov)
    if not key:
        res.skipped = True
        println(f"[llm] {prov}/{model}: no credential configured; skipped")
        return res

    try:
        t0 = time.perf_counter()
        if prov == "openai":
            text, in_tok, out_tok, cached = _call_openai(
                model, prompt, key, use_web_search=use_web_search,
                max_tokens=max_tokens)
        elif prov == "anthropic":
            text, in_tok, out_tok, cached = _call_anthropic(
                model, prompt, key, max_tokens=max_tokens)
        elif prov in ("xai", "deepseek", "llama"):
            text, in_tok, out_tok, cached = _call_openai_compatible(
                _ENDPOINTS[prov], model, prompt, key,
                max_tokens=max_tokens, temperature=temperature)
        else:
            res.error = f"unsupported provider: {prov}"
            return res
        res.latency_sec = time.perf_counter() - t0
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        body = e.response.text[:200] if e.response is not None else ""
        res.error = f"HTTP {code}: {body}"
        println(f"[llm] {prov}/{model} failed: {res.error}")
        return res
    except Exception as e:
        res.error = f"{type(e).__name__}: {str(e)[:200]}"
        println(f"[llm] {prov}/{model} failed: {res.error}")
        return res

    res.text = text or ""
    res.max_tokens_used = max_tokens
    res.web_search_used = bool(use_web_search and prov == "openai")
    res.input_tokens, res.output_tokens, res.cached_input_tokens = in_tok, out_tok, cached
    res.cost_usd = estimate_cost(model, in_tok, out_tok,
                                 cached_input_tokens=cached,
                                 cache_hit=bool(cached))
    res.pricing_used = price_of(model)
    if not res.text:
        println(f"[llm] {prov}/{model}: empty response")
    return res


def _call_openai(model: str, prompt: str, key: str, *,
                 use_web_search: bool, max_tokens: int):
    from openai import OpenAI
    client = OpenAI(api_key=key, timeout=LLM_TIMEOUT)
    kwargs: dict[str, Any] = {"model": model, "input": prompt}
    if use_web_search:
        kwargs["tools"] = [{"type": "web_search"}]
        kwargs["tool_choice"] = "auto"
    resp = client.responses.create(**kwargs)
    text = getattr(resp, "output_text", "") or ""
    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
    details = getattr(usage, "input_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0)
    return text, in_tok, out_tok, cached


def _call_anthropic(model: str, prompt: str, key: str, *, max_tokens: int):
    r = requests.post(
        _ENDPOINTS["anthropic"],
        headers={"x-api-key": key, "content-type": "application/json",
                 "anthropic-version": "2023-06-01"},
        json={"model": model, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=LLM_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    text = "\n".join(p.get("text", "") for p in j.get("content", [])
                     if isinstance(p, dict) and p.get("text"))
    usage = j.get("usage") or {}
    in_tok = int(usage.get("input_tokens") or 0)
    out_tok = int(usage.get("output_tokens") or 0)
    cached = int(usage.get("cache_read_input_tokens") or 0)
    if not in_tok:
        in_tok = _anthropic_count_tokens(model, prompt, key)
    return text, in_tok, out_tok, cached


def _call_openai_compatible(url: str, model: str, prompt: str, key: str, *,
                            max_tokens: int, temperature: float):
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "max_tokens": max_tokens, "temperature": temperature,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=LLM_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    text = ""
    choices = j.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        text = msg.get("content") or msg.get("reasoning_content") or ""
    in_tok, out_tok, cached = _usage_from_openai_shape(j)
    return text, in_tok, out_tok, cached
