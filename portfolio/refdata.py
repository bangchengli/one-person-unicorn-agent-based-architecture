from __future__ import annotations

import time
from typing import List, Dict

import pandas as pd
import requests

from core.config import POLYGON_API_KEY

_REF_BASE = "https://api.polygon.io/v3/reference/tickers/{ticker}"
_AGG_BASE = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"


def _get_ticker_sector(ticker: str, sleep_time: float = 0.0) -> str:
    if not POLYGON_API_KEY:
        return "Unknown"
    url = _REF_BASE.format(ticker=ticker)
    params = {"apiKey": POLYGON_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        j = r.json() or {}
        res = j.get("results", {}) if isinstance(j, dict) else {}
        sector = (
            res.get("sector")
            or res.get("industry")
            or res.get("sic_description")
            or res.get("market")
        )
        if isinstance(sector, str) and sector.strip():
            s = sector.strip().split(",")[0][:64]
        else:
            s = "Unknown"
        if sleep_time > 0:
            time.sleep(sleep_time)
        return s
    except Exception:
        return "Unknown"


def _avg_dollar_volume(ticker: str, start: str, end: str, sleep_time: float = 0.0) -> float:
    if not POLYGON_API_KEY:
        return 0.0
    url = _AGG_BASE.format(ticker=ticker, start=start, end=end)
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        results = (r.json() or {}).get("results", []) or []
        if not results:
            return 0.0
        vals = []
        for it in results:
            c = it.get("c")
            v = it.get("v")
            if c is None or v is None:
                continue
            try:
                vals.append(float(c) * float(v))
            except Exception:
                continue
        if sleep_time > 0:
            time.sleep(sleep_time)
        if not vals:
            return 0.0
        return float(pd.Series(vals).mean())
    except Exception:
        return 0.0


def build_meta_df(tickers: List[str], start: str, end: str,
                  per_req_sleep: float = 0.15) -> pd.DataFrame:
    uniq = [t for t in sorted({(t or '').strip().upper() for t in tickers}) if t]
    rows: List[Dict] = []
    total = len(uniq)
    for i, t in enumerate(uniq, 1):
        sector = _get_ticker_sector(t, sleep_time=per_req_sleep)
        adv = _avg_dollar_volume(t, start=start, end=end, sleep_time=per_req_sleep)
        rows.append({"ticker": t, "sector": sector, "avg_dollar_vol": adv})
    df = pd.DataFrame(rows, columns=["ticker", "sector", "avg_dollar_vol"])
    return df
