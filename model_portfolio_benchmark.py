import os
import re
import json
import time
import math
import pandas as pd
import numpy as np
import requests
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timezone, time as dtime, timedelta
from typing import List, Dict, Tuple, Any, Optional
from itertools import combinations

# =====================
# Config
# =====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

TXT_SAVE_DIR   = r"D:/my-fin-project/txt_save"
JSON_SAVE_DIR  = r"D:/my-fin-project/json_save"
CSV_SAVE_DIR   = r"D:/my-fin-project/csv_save"
CHART_SAVE_DIR = r"D:/my-fin-project/chart_save"
CONFIG_DIR     = r"D:/my-fin-project/configs"

MODEL_PRICING_JSON = os.getenv("MODEL_PRICING_JSON", os.path.join(CONFIG_DIR, "model_pricing.json"))
MODEL_PRICING_URL  = os.getenv("MODEL_PRICING_URL", "")  # optional remote JSON

BASE_URL = (
    "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/"
    "{multiplier}/{timespan}/{from_}/{to}"
)

REF_TICKER_URL = "https://api.polygon.io/v3/reference/tickers"

# Default static pricing (fallback)
DEFAULT_PRICE_PER_MILLION = {
    "gpt-5":       {"input": 1.25, "output": 10.00},
    "gpt-5-mini":  {"input": 0.25, "output": 2.00},
    "gpt-5-nano":  {"input": 0.05, "output": 0.40},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o3":          {"input": 2.00, "output": 8.00},
    "o4-mini":     {"input": 1.10, "output": 4.40},
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

# Global pricing dicts (can be updated by config)
PRICE_PER_MILLION = DEFAULT_PRICE_PER_MILLION.copy()
DEEPSEEK_PRICING  = DEFAULT_DEEPSEEK_PRICING.copy()

# =====================
# Progress display
# =====================
STAGES = [
    "1/4 Generate portfolio (LLM)",
    "2/4 Validate tickers (Polygon Ref)",
    "3/4 Fetch prices (Polygon)",
    "4/4 Compute and save"
]

def progress(stage_idx: int, note: str = ""):
    total_blocks = 30
    frac = (stage_idx + 1) / len(STAGES)
    filled = int(total_blocks * frac)
    bar = "#" * filled + "-" * (total_blocks - filled)
    stage_text = STAGES[stage_idx] if 0 <= stage_idx < len(STAGES) else ""
    if note:
        stage_text = f"{stage_text} | {note}"
    sys.stdout.write(f"\r[{bar}] {stage_text}".ljust(120))
    sys.stdout.flush()

def println(msg: str = ""):
    sys.stdout.write("\n" + msg + "\n")
    sys.stdout.flush()

def _ensure_dirs():
    for d in [TXT_SAVE_DIR, JSON_SAVE_DIR, CSV_SAVE_DIR, CHART_SAVE_DIR, CONFIG_DIR]:
        os.makedirs(d, exist_ok=True)

# =====================
# Pricing config loaders
# =====================

def _safe_load_json(path: str) -> Optional[dict]:
    try:
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        println(f"Failed to load pricing JSON from {path}: {e}")
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

def load_pricing_config():
    """
    Load pricing from optional local file and/or URL. Fallback to defaults.
    Merge order: defaults < local file < remote URL.
    """
    global PRICE_PER_MILLION, DEEPSEEK_PRICING

    cfg = _safe_load_json(MODEL_PRICING_JSON) or {}
    url_cfg = _safe_load_json_from_url(MODEL_PRICING_URL) or {}

    def merge(base: dict, override: dict) -> dict:
        out = base.copy()
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
        return out

    if cfg.get("PRICE_PER_MILLION") or url_cfg.get("PRICE_PER_MILLION"):
        PRICE_PER_MILLION = merge(DEFAULT_PRICE_PER_MILLION, cfg.get("PRICE_PER_MILLION", {}))
        PRICE_PER_MILLION = merge(PRICE_PER_MILLION, url_cfg.get("PRICE_PER_MILLION", {}))

    if cfg.get("DEEPSEEK_PRICING") or url_cfg.get("DEEPSEEK_PRICING"):
        DEEPSEEK_PRICING = merge(DEFAULT_DEEPSEEK_PRICING, cfg.get("DEEPSEEK_PRICING", {}))
        DEEPSEEK_PRICING = merge(DEEPSEEK_PRICING, url_cfg.get("DEEPSEEK_PRICING", {}))

    snapshot_path = os.path.join(CONFIG_DIR, f"effective_pricing_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
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

# =====================
# Robust JSON parsing
# =====================
def _strip_trailing_commas(s: str) -> str:
    s = s.strip()
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    s = s.rstrip(", \t\r\n")
    return s

def _robust_json_parse(raw: str) -> List[Dict[str, Any]]:
    raw = raw.strip()
    if not raw:
        return []
    try:
        obj = json.loads(_strip_trailing_commas(raw))
        return obj if isinstance(obj, list) else [obj]
    except Exception:
        pass

    items = []
    for line in raw.splitlines():
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

    objs = re.findall(r'\{(?:[^\{\}]|(?R))*\}', raw, flags=re.DOTALL)
    if objs:
        joined = "[" + ",".join(o.rstrip(", \t\r\n") for o in objs) + "]"
        try:
            return json.loads(_strip_trailing_commas(joined))
        except Exception:
            pass

    raise ValueError("Cannot parse portfolio JSON from model output.")

# =====================
# DeepSeek pricing helpers
# =====================
def _is_deepseek_discount_window_utc(now_utc: Optional[datetime] = None) -> bool:
    """
    Discount window: UTC 16:30 to 00:30 (cross-day)
    Corresponding Singapore time (UTC+8): 00:30 to 08:30
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    t = now_utc.time()
    return (t >= dtime(16, 30)) or (t < dtime(0, 30))

def _deepseek_cost(model: str, input_tokens: int, output_tokens: int, cache_hit: bool) -> float:
    tier = "discount" if _is_deepseek_discount_window_utc() else "standard"
    if model not in DEEPSEEK_PRICING.get(tier, {}):
        return 0.0
    table = DEEPSEEK_PRICING[tier][model]
    in_price  = table.get("input_hit" if cache_hit else "input_miss", 0.0)
    out_price = table.get("output", 0.0)
    return (input_tokens / 1_000_000.0) * in_price + (output_tokens / 1_000_000.0) * out_price

# =====================
# LLM: Generate portfolio
# =====================
def _openai_generate(model: str, prompt: str, use_web_search: bool):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    kwargs = {"model": model, "input": prompt}
    if use_web_search:
        kwargs["tools"] = [{"type": "web_search"}]
        kwargs["tool_choice"] = "auto"
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
    return raw, latency, int(in_tok or 0), int(out_tok or 0)

def _deepseek_generate(model: str, prompt: str):
    from openai import OpenAI
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    messages = [{"role": "user", "content": prompt}]
    t0 = time.perf_counter()
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=1024, temperature=0.3)
    latency = time.perf_counter() - t0
    try:
        raw = resp.choices[0].message.content
    except Exception:
        raw = str(resp)
    usage = getattr(resp, "usage", None) or {}
    in_tok  = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else getattr(usage, "prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0) if isinstance(usage, dict) else getattr(usage, "completion_tokens", 0)
    return raw, latency, int(in_tok or 0), int(out_tok or 0)

def generate_portfolio_with_model(model: str, n_min: int = 15, n_max: int = 25, use_web_search: bool = True):
    """
    Returns: (tickers, latency_sec, input_tokens, output_tokens, est_cost_usd)
    """
    prompt = (
        "You are an expert portfolio construction advisor. "
        f"Generate a diversified US equities portfolio with {n_min}-{n_max} tickers ONLY. "
        "Respond as a pure JSON array. Each element must be an object with the key 'name' containing the ticker symbol. "
        "No commentary. Example: [{\"name\": \"AAPL\"}, {\"name\": \"MSFT\"}]"
    )

    load_pricing_config()

    if model.startswith("deepseek-"):
        raw, latency, in_tok, out_tok = _deepseek_generate(model, prompt)
        est_cost = _deepseek_cost(model, in_tok, out_tok, cache_hit=False)
    else:
        raw, latency, in_tok, out_tok = _openai_generate(model, prompt, use_web_search=use_web_search)
        price = PRICE_PER_MILLION.get(model, {"input": 0.0, "output": 0.0})
        est_cost = (in_tok / 1_000_000.0) * price.get("input", 0.0) + \
                   (out_tok / 1_000_000.0) * price.get("output", 0.0)

    _ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    txt_file = os.path.join(TXT_SAVE_DIR, f"{model.replace(':','_')}_portfolio_{ts}.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(raw)

    data = _robust_json_parse(raw)
    tickers = [x.get("name", "").strip().upper() for x in data if isinstance(x, dict) and x.get("name")]
    tickers = list(dict.fromkeys([t for t in tickers if t]))

    json_file = os.path.join(JSON_SAVE_DIR, f"{model.replace(':','_')}_portfolio_{ts}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump([{"name": t} for t in tickers], f, ensure_ascii=False, indent=2)

    return tickers, float(latency), int(in_tok or 0), int(out_tok or 0), float(est_cost)

# =====================
# Polygon: ticker validation with caching
# =====================
TICKER_CACHE_PATH = os.path.join(JSON_SAVE_DIR, "polygon_ticker_cache.json")
CACHE_TTL_DAYS = 7

def _load_ticker_cache() -> dict:
    try:
        if not os.path.isfile(TICKER_CACHE_PATH):
            return {}
        with open(TICKER_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_ticker_cache(cache: dict):
    try:
        with open(TICKER_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _cache_fresh(ts_str: str) -> bool:
    try:
        ts = datetime.fromisoformat(ts_str)
        return datetime.now() - ts < timedelta(days=CACHE_TTL_DAYS)
    except Exception:
        return False

ACCEPT_TYPES = {"CS", "ADR", "ETF"}  # Common Stock, ADR, ETF

def polygon_ticker_info(ticker: str) -> Optional[dict]:
    """
    Lookup single ticker via Polygon v3 reference endpoint.
    """
    params = {
        "ticker": ticker,
        "active": "true",
        "market": "stocks",
        "limit": 1,
        "apiKey": POLYGON_API_KEY,
    }
    try:
        r = requests.get(REF_TICKER_URL, params=params, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])
        return results[0] if results else None
    except Exception as e:
        println(f"Reference lookup failed for {ticker}: {e}")
        return None

def validate_tickers_polygon(tickers: List[str]) -> Tuple[List[str], List[str]]:
    """
    Return (valid, invalid). Uses local cache for speed; accepts CS/ADR/ETF only.
    """
    cache = _load_ticker_cache()
    valid, invalid = [], []

    for t in tickers:
        entry = cache.get(t)
        if entry and _cache_fresh(entry.get("ts", "")):
            if entry.get("ok"):
                valid.append(t)
            else:
                invalid.append(t)
            continue

        info = polygon_ticker_info(t)
        ok = bool(info) and (info.get("type") in ACCEPT_TYPES)
        cache[t] = {"ok": ok, "ts": datetime.now().isoformat(), "meta": {"type": info.get("type") if info else None}}
        if ok:
            valid.append(t)
        else:
            invalid.append(t)
        time.sleep(0.2)

    _save_ticker_cache(cache)
    return valid, invalid

# =====================
# Market data
# =====================
def fetch_prices_polygon(tickers: List[str], start: str, end: str,
                         multiplier: int = 1, timespan: str = "day",
                         sleep_time: float = 0.8) -> Optional[str]:
    """
    Returns CSV path with wide-format prices (date index, columns=tickers).
    """
    _ensure_dirs()

    progress(1, note=f"validate {len(tickers)} tickers")
    valid, invalid = validate_tickers_polygon(tickers)
    if invalid:
        println(f"Invalid or inactive tickers removed: {', '.join(invalid)}")
    if not valid:
        println("No valid tickers after validation. Abort fetching.")
        return None

    records = []
    total = len(valid)
    t0 = time.perf_counter()
    for i, t in enumerate(valid, 1):
        progress(2, note=f"fetch {t} ({i}/{total})")
        url = BASE_URL.format(ticker=t, multiplier=multiplier, timespan=timespan, from_=start, to=end)
        params = {"adjusted": True, "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            results = r.json().get("results", [])
            if not results:
                println(f"{t}: no data")
                continue
            for item in results:
                records.append({
                    "date": pd.to_datetime(item["t"], unit="ms"),
                    "ticker": t,
                    "close": item["c"]
                })
        except Exception as e:
            println(f"{t} fetch failed: {e}")
            continue
        time.sleep(sleep_time)
    t1 = time.perf_counter()
    fetch_latency = t1 - t0

    if not records:
        println("Empty records")
        return None

    df = pd.DataFrame(records).pivot(index="date", columns="ticker", values="close").sort_index()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = os.path.join(CSV_SAVE_DIR, f"prices_{ts}.csv")
    df.to_csv(out_csv)
    with open(out_csv.replace(".csv", ".latency.txt"), "w", encoding="utf-8") as f:
        f.write(f"fetch_latency_sec={fetch_latency:.3f}\n")
    return out_csv

# =====================
# Metrics
# =====================
def equity_curve_from_equal_weight(df: pd.DataFrame) -> pd.Series:
    df = df.dropna(how="all")
    df = df.ffill().dropna(axis=1, how="all")
    rets = df.pct_change().dropna()
    if rets.empty:
        return pd.Series(dtype=float)
    eq_w = np.repeat(1.0 / rets.shape[1], rets.shape[1])
    port_ret = rets.dot(eq_w)
    equity = (1 + port_ret).cumprod()
    equity.name = "equity"
    return equity

def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return dd.min()

def annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    if equity.empty:
        return float("nan")
    n = len(equity)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = n / periods_per_year
    if years <= 0:
        return float("nan")
    return (1.0 + total_return) ** (1.0 / years) - 1.0

def annualized_volatility(df: pd.DataFrame, periods_per_year: int = 252) -> float:
    rets = df.pct_change().dropna()
    if rets.empty:
        return float("nan")
    eq_w = np.repeat(1.0 / rets.shape[1], rets.shape[1])
    port_ret = rets.dot(eq_w)
    return port_ret.std() * math.sqrt(periods_per_year)

def sharpe_ratio(df: pd.DataFrame, rf: float = 0.0, periods_per_year: int = 252) -> float:
    rets = df.pct_change().dropna()
    if rets.empty:
        return float("nan")
    eq_w = np.repeat(1.0 / rets.shape[1], rets.shape[1])
    port_ret = rets.dot(eq_w)
    excess = port_ret - (rf / periods_per_year)
    mu = excess.mean() * periods_per_year
    sigma = excess.std() * math.sqrt(periods_per_year)
    return float("nan") if sigma == 0 else mu / sigma

def hist_var_cvar(df: pd.DataFrame, level: float = 0.95) -> Tuple[float, float]:
    rets = df.pct_change().dropna()
    if rets.empty:
        return (float("nan"), float("nan"))
    eq_w = np.repeat(1.0 / rets.shape[1], rets.shape[1])
    port_ret = rets.dot(eq_w)
    q = np.quantile(port_ret, 1 - level)
    cvar = port_ret[port_ret <= q].mean()
    return q, cvar

# =====================
# Visualization
# =====================
def visualize_cost_efficiency(df: pd.DataFrame,
                              x_col: str = "est_api_cost_usd",
                              y_col: str = "sharpe",
                              label_col: str = "model",
                              title: str = "Cost Performance Frontier",
                              outfile: Optional[str] = None):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        print("visualize_cost_efficiency: missing columns, skip plot")
        return

    plt.figure()
    xs = df[x_col].values
    ys = df[y_col].values
    plt.scatter(xs, ys)
    labels = df[label_col].astype(str).tolist()
    for i, txt in enumerate(labels):
        plt.annotate(txt, (xs[i], ys[i]))
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True)
    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        print(f"Saved chart: {outfile}")
    else:
        plt.show()

# =====================
# Stability testing
# =====================
def jaccard_similarity(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def average_pairwise_jaccard(list_of_lists):
    pairs = list(combinations(range(len(list_of_lists)), 2))
    if not pairs:
        return float("nan")
    vals = []
    for i, j in pairs:
        vals.append(jaccard_similarity(list_of_lists[i], list_of_lists[j]))
    return float(np.mean(vals)) if vals else float("nan")

def stability_test_for_model(model: str, repeats: int, start: str, end: str,
                             use_web_search: bool = True) -> dict:
    sharpe_list = []
    annret_list = []
    mdd_list = []
    tickers_runs = []

    for r in range(repeats):
        tickers, _, _, _, _ = generate_portfolio_with_model(model, use_web_search=use_web_search)
        tickers_runs.append(tickers)

        csv_path = fetch_prices_polygon(tickers, start=start, end=end)
        if not csv_path:
            continue
        df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
        equity = equity_curve_from_equal_weight(df)
        sharpe_list.append(sharpe_ratio(df))
        annret_list.append(annualized_return(equity))
        mdd_list.append(max_drawdown(equity))

    return {
        "model": model,
        "repeats": repeats,
        "jaccard_mean": average_pairwise_jaccard(tickers_runs),
        "sharpe_mean": float(np.nanmean(sharpe_list)) if sharpe_list else float("nan"),
        "sharpe_std":  float(np.nanstd(sharpe_list)) if sharpe_list else float("nan"),
        "ann_return_mean": float(np.nanmean(annret_list)) if annret_list else float("nan"),
        "ann_return_std":  float(np.nanstd(annret_list)) if annret_list else float("nan"),
        "max_dd_mean": float(np.nanmean(mdd_list)) if mdd_list else float("nan"),
        "max_dd_std":  float(np.nanstd(mdd_list)) if mdd_list else float("nan"),
    }

# =====================
# Benchmark pipeline
# =====================
def benchmark_models(models: List[str], start: str, end: str, use_web_search: bool = True) -> pd.DataFrame:
    rows = []
    total_models = len(models)

    for idx, m in enumerate(models, 1):
        # Stage 1: generate
        progress(0, note=f"{m} | generate portfolio")
        t0 = time.perf_counter()
        tickers, gen_latency, in_tok, out_tok, est_cost = generate_portfolio_with_model(
            m, use_web_search=use_web_search
        )
        t1 = time.perf_counter()
        s1_elapsed = t1 - t0
        println(f"\n{m} -> {len(tickers)} tickers: {', '.join(tickers[:8])}{'...' if len(tickers) > 8 else ''}")
        println(f"LLM latency: {gen_latency:.3f}s | stage1_elapsed: {s1_elapsed:.3f}s | tokens: in={in_tok}, out={out_tok}, est_cost=${est_cost:.4f}")
        if not tickers:
            println(f"{m}: no tickers returned, skip")
            continue

        # Stage 2 + 3: validation + data
        csv_path = fetch_prices_polygon(tickers, start=start, end=end)
        if not csv_path:
            println(f"{m}: no CSV produced, skip")
            continue

        # Stage 4: metrics
        progress(3, note=f"{m} | compute metrics")
        t4 = time.perf_counter()
        df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
        equity = equity_curve_from_equal_weight(df)
        ann_ret = annualized_return(equity)
        ann_vol = annualized_volatility(df)
        sharpe = sharpe_ratio(df)
        var95, cvar95 = hist_var_cvar(df, level=0.95)
        mdd = max_drawdown(equity)
        t5 = time.perf_counter()
        s3_elapsed = t5 - t4

        rows.append({
            "model": m,
            "n_tickers": len(df.columns),
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "VaR_95": var95,
            "CVaR_95": cvar95,
            "llm_latency_sec": gen_latency,
            "stage1_elapsed_sec": s1_elapsed,
            "stage3_elapsed_sec": s3_elapsed,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "est_api_cost_usd": est_cost
        })

        progress(3, note=f"{m} | saving ({idx}/{total_models})")

    res = pd.DataFrame(rows).sort_values(["sharpe"], ascending=[False])
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    _ensure_dirs()
    out_csv = os.path.join(CSV_SAVE_DIR, f"benchmark_{ts}.csv")
    res.to_csv(out_csv, index=False)
    println(f"\nBenchmark saved to: {out_csv}")
    return res

# =====================
# Composite score
# =====================
def add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in ["sharpe", "ann_vol", "est_api_cost_usd", "llm_latency_sec"]:
        if col not in out.columns:
            out[col] = np.nan
    out["composite_score_aggressive"] = (
        0.6  * out["sharpe"].fillna(0)
        - 0.15 * out["ann_vol"].fillna(0)
        - 100.0 * out["est_api_cost_usd"].fillna(0)
        - 0.1 * out["llm_latency_sec"].fillna(0)
    )
    return out

# =====================
# Main
# =====================
if __name__ == "__main__":
    MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5-mini",
        "gpt-5",
        "o3",
        "o4-mini",
        "deepseek-chat",
        "deepseek-reasoner",
    ]
    START = "2024-07-25"
    END   = "2025-08-07"

    if (not OPENAI_API_KEY) or (not POLYGON_API_KEY):
        println("Please set OPENAI_API_KEY and POLYGON_API_KEY as environment variables.")
        sys.exit(0)
    if any(m.startswith("deepseek-") for m in MODELS) and not DEEPSEEK_API_KEY:
        println("Warning: DEEPSEEK_API_KEY not set; DeepSeek models will fail.")

    load_pricing_config()

    df = benchmark_models(MODELS, START, END, use_web_search=True)
    df = add_composite_score(df)
    println(str(df.sort_values("composite_score_aggressive", ascending=False)))

    try:
        _ensure_dirs()
        visualize_cost_efficiency(
            df, x_col="est_api_cost_usd", y_col="sharpe",
            title="Cost Performance Frontier",
            outfile=os.path.join(CHART_SAVE_DIR, "frontier_cost_vs_sharpe.png")
        )
        visualize_cost_efficiency(
            df, x_col="llm_latency_sec", y_col="sharpe",
            title="Latency Performance Frontier",
            outfile=os.path.join(CHART_SAVE_DIR, "frontier_latency_vs_sharpe.png")
        )
    except Exception as e:
        println(f"visualization failed: {e}")

    REPEATS = 0
    if REPEATS > 0:
        STAB_ROWS = []
        for m in MODELS:
            println(f"stability test for {m} (repeats={REPEATS})")
            stats = stability_test_for_model(m, repeats=REPEATS, start=START, end=END, use_web_search=True)
            STAB_ROWS.append(stats)

        stab_df = pd.DataFrame(STAB_ROWS)
        println(str(stab_df))
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        stab_path = os.path.join(CSV_SAVE_DIR, f"stability_{ts}.csv")
        stab_df.to_csv(stab_path, index=False)
        println(f"stability saved to: {stab_path}")
