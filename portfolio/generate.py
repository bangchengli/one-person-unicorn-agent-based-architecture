from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from core.config import PORTFOLIO_OUT_DIR, ensure_dirs
from core.llm import LLMResult, call
from core.pricing import load_pricing_overrides
from core.utils import println, timestamp

PROMPT = (
    "You are an expert portfolio construction advisor. "
    "Generate a diversified US equities portfolio with {n_min}-{n_max} tickers ONLY. "
    "Respond as a pure JSON array. Each element must be an object with the key "
    "'name' containing the ticker symbol. No commentary. "
    'Example: [{{"name": "AAPL"}}, {{"name": "MSFT"}}]'
)

_TICKER_STOP = {"AND", "THE", "FOR", "WITH", "FROM", "THIS", "THAT", "YOU",
                "YOUR", "A", "AN", "IN", "ON", "BY", "TO", "AS", "AT", "OF",
                "JSON", "USD", "ETF", "US"}


@dataclass
class PortfolioResult:

    model: str
    provider: str
    tickers: list[str] = field(default_factory=list)
    latency_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None
    error: Optional[str] = None
    skipped: bool = False
    web_search_used: bool = False
    max_tokens_used: int = 0
    raw_text: str = field(default="", repr=False)

    @property
    def ok(self) -> bool:
        return bool(self.tickers) and self.error is None and not self.skipped

    def as_row(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "status": "skipped" if self.skipped else ("failed" if self.error else "success"),
            "n_tickers": len(self.tickers),
            "llm_latency_sec": round(self.latency_sec, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "est_api_cost_usd": self.cost_usd,
            "web_search_used": self.web_search_used,
            "max_tokens_used": self.max_tokens_used,
            "error": self.error,
        }


def _strip_trailing_commas(s: str) -> str:
    out: list[str] = []
    for ch in s.strip():
        if ch in "}]" and out and out[-1] == ",":
            out.pop()
        out.append(ch)
    return "".join(out).strip()


def robust_json_parse(raw: Any) -> list[dict]:
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
    except json.JSONDecodeError:
        pass

    items = []
    for line in text.splitlines():
        s = line.strip().rstrip(",").strip()
        if not s or s.startswith(("//", "#")):
            continue
        try:
            parsed = json.loads(_strip_trailing_commas(s))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            items.append(parsed)
    if items:
        return items

    objs, depth, buf = [], 0, []
    for ch in text:
        if ch == "{":
            depth += 1
        if depth > 0:
            buf.append(ch)
        if ch == "}":
            depth -= 1
            if depth == 0 and buf:
                try:
                    parsed = json.loads(_strip_trailing_commas("".join(buf)))
                    if isinstance(parsed, dict):
                        objs.append(parsed)
                except json.JSONDecodeError:
                    pass
                buf = []
    return objs


def extract_tickers_fallback(text: str, n_max: int = 25) -> list[str]:
    out, seen = [], set()
    for t in re.findall(r"\b[A-Z]{1,5}\b", text or ""):
        if t in _TICKER_STOP or not t.isalpha() or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= n_max:
            break
    return out


def generate_portfolio(model: str, *,
                       provider: Optional[str] = None,
                       n_min: int = 15,
                       n_max: int = 25,
                       use_web_search: bool = True,
                       save: bool = True) -> PortfolioResult:
    load_pricing_overrides()
    res: LLMResult = call(
        model,
        PROMPT.format(n_min=n_min, n_max=n_max),
        provider=provider,
        max_tokens=1024,
        use_web_search=use_web_search,
    )

    out = PortfolioResult(
        model=model, provider=res.provider,
        latency_sec=res.latency_sec,
        input_tokens=res.input_tokens, output_tokens=res.output_tokens,
        cost_usd=res.cost_usd, error=res.error, skipped=res.skipped,
        web_search_used=res.web_search_used,
        max_tokens_used=res.max_tokens_used,
        raw_text=res.text,
    )
    if not res.ok:
        return out

    parsed = robust_json_parse(res.text)
    tickers = [str(x.get("name", "")).strip().upper()
               for x in parsed if isinstance(x, dict) and x.get("name")]
    tickers = list(dict.fromkeys(t for t in tickers if t))
    if not tickers:
        tickers = extract_tickers_fallback(res.text, n_max=n_max)
        if tickers:
            println(f"[portfolio] {model}: no JSON parsed; recovered "
                    f"{len(tickers)} tickers by regex fallback")
    out.tickers = tickers

    if not tickers:
        out.error = "no tickers recoverable from the response"
        return out

    if save:
        ensure_dirs()
        tag = model.replace("/", "_").replace(":", "_")
        stem = PORTFOLIO_OUT_DIR / f"{tag}_portfolio_{timestamp()}"
        try:
            stem.with_suffix(".txt").write_text(res.text, encoding="utf-8")
            stem.with_suffix(".json").write_text(
                json.dumps({"model": model, "provider": res.provider,
                            "tickers": tickers, "ops": res.as_row()},
                           ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as e:
            println(f"[portfolio] could not save {model} output: {type(e).__name__}")

    return out
