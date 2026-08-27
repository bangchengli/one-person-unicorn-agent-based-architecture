from __future__ import annotations

import concurrent.futures as cf
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

from .cache import JSONCache
from .config import (
    HTTP_TIMEOUT,
    MAX_WORKERS,
    POLYGON_AGGS_URL,
    POLYGON_API_KEY,
    POLYGON_NEWS_URL,
    PRICE_CACHE_DIR,
)
from .utils import println

_session = requests.Session()
_session.mount("https://", HTTPAdapter(
    max_retries=Retry(total=5, connect=3, read=3, backoff_factor=0.5,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET", "POST"),
                      respect_retry_after_header=True),
    pool_connections=50, pool_maxsize=50))
_session.headers.update({"Accept-Encoding": "gzip, deflate"})

_news_cache = JSONCache("polygon_news")
_series_memo: dict[str, pd.Series] = {}


def _require_key() -> bool:
    if not POLYGON_API_KEY:
        println("[market] POLYGON_API_KEY is not set; no market data available.")
        return False
    return True


def fetch_ohlcv(ticker: str, start: str, end: str, *,
                multiplier: int = 1, timespan: str = "day",
                adjusted: bool = True) -> Optional[pd.DataFrame]:
    if not _require_key():
        return None

    url = POLYGON_AGGS_URL.format(ticker=ticker.upper(), multiplier=multiplier,
                                  timespan=timespan, start=start, end=end)
    try:
        r = _session.get(url, params={
            "adjusted": "true" if adjusted else "false",
            "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY,
        }, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        results = (r.json() or {}).get("results") or []
    except Exception as e:
        println(f"[market] {ticker} OHLCV fetch failed: {type(e).__name__}: {e}")
        return None

    if not results:
        println(f"[market] {ticker}: no bars between {start} and {end}")
        return None

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume"})
    return df[["date", "open", "high", "low", "close", "volume"]] \
        .set_index("date").sort_index()


def _series_path(ticker: str, start: str, end: str, multiplier: int, timespan: str):
    safe = f"{ticker}_{start}_{end}_{multiplier}{timespan}".replace("/", "-")
    return PRICE_CACHE_DIR / f"{safe}.parquet"


def fetch_close_series(ticker: str, start: str, end: str, *,
                       multiplier: int = 1, timespan: str = "day",
                       use_cache: bool = True) -> Optional[pd.Series]:
    key = f"{ticker}|{start}|{end}|{multiplier}|{timespan}"
    if key in _series_memo:
        return _series_memo[key]

    path = _series_path(ticker, start, end, multiplier, timespan)
    if use_cache and path.exists():
        try:
            frame = pd.read_parquet(path)
            s = frame[ticker] if ticker in frame.columns else frame.iloc[:, 0]
            s.name = ticker
            _series_memo[key] = s
            return s
        except Exception:
            pass

    df = fetch_ohlcv(ticker, start, end, multiplier=multiplier, timespan=timespan)
    if df is None or df.empty:
        return None
    s = df["close"].astype("float32")
    s.name = ticker
    try:
        PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        s.to_frame().to_parquet(path, compression="zstd", index=True)
    except Exception:
        pass
    _series_memo[key] = s
    return s


def fetch_price_matrix(tickers: list[str], start: str, end: str, *,
                       use_cache: bool = True,
                       max_workers: int = MAX_WORKERS) -> Optional[pd.DataFrame]:
    unique = sorted({t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()})
    if not unique:
        println("[market] no tickers requested")
        return None

    frames: list[pd.Series] = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_close_series, t, start, end,
                             use_cache=use_cache): t for t in unique}
        for fut in cf.as_completed(futures):
            try:
                s = fut.result()
            except Exception as e:
                println(f"[market] {futures[fut]} failed: {type(e).__name__}: {e}")
                continue
            if s is not None:
                frames.append(s)

    if not frames:
        println("[market] every ticker fetch failed")
        return None
    return pd.concat(frames, axis=1).sort_index()


def fetch_news(ticker: str, *, as_of: Optional[str] = None,
               limit: int = 5, use_cache: bool = True) -> list[dict]:
    key = f"{ticker.upper()}|{as_of or 'latest'}|{limit}"
    if use_cache and _news_cache.has(key):
        return _news_cache.get(key) or []

    if not _require_key():
        return []

    params = {"ticker": ticker.upper(), "limit": limit, "sort": "published_utc",
              "order": "desc", "apiKey": POLYGON_API_KEY}
    if as_of:
        params["published_utc.lte"] = as_of

    try:
        r = _session.get(POLYGON_NEWS_URL, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        results = (r.json() or {}).get("results") or []
    except Exception as e:
        println(f"[market] {ticker} news fetch failed: {type(e).__name__}: {e}")
        return []

    items = [{
        "datetime": it.get("published_utc"),
        "title": it.get("title"),
        "summary": it.get("description"),
        "source": it.get("author"),
        "url": it.get("article_url"),
        "keywords": (it.get("keywords") or [])[:3],
    } for it in results]

    if items:
        _news_cache.set(key, items)
        _news_cache.flush()
    return items
