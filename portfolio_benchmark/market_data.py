import concurrent.futures as cf
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

from .config import POLYGON_API_KEY, BASE_URL, CSV_SAVE_DIR, PER_TICKER_DIR
from .utils import ensure_dirs, progress, println

# -----------------------------------------------------------------------------
# Shared HTTP session with retry & connection pooling
# -----------------------------------------------------------------------------
_session = requests.Session()
_retry = Retry(
    total=5,                 # total retry attempts (includes connect/read)
    connect=3,               # connection errors
    read=3,                  # read errors
    backoff_factor=0.5,      # exponential backoff: 0.5, 1.0, 2.0, ...
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET", "POST"),
    respect_retry_after_header=True,  # honor server Retry-After
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=50, pool_maxsize=50)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)
_session.headers.update({"Accept-Encoding": "gzip, deflate"})

# In-process memo to avoid duplicate HTTP calls within a single run
_INPROC_SERIES_CACHE: Dict[str, pd.Series] = {}

# Ensure per-ticker parquet directory exists
PER_TICKER_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _per_ticker_path(ticker: str, start: str, end: str, multiplier: int, timespan: str):
    safe = f"{ticker}_{start}_{end}_{multiplier}{timespan}".replace("/", "-")
    return PER_TICKER_DIR / f"{safe}.parquet"


def _fetch_single_series_from_api(
    ticker: str,
    start: str,
    end: str,
    multiplier: int = 1,
    timespan: str = "day",
    timeout: int = 30,
) -> Optional[pd.Series]:
    """Fetch one ticker's close price series from Polygon.

    Returns a pandas Series (index=datetime, name=ticker) or None when unavailable.
    """
    if not POLYGON_API_KEY:
        println("POLYGON_API_KEY is not set; cannot fetch data.")
        return None

    url = BASE_URL.format(ticker=ticker, multiplier=multiplier, timespan=timespan, from_=start, to=end)
    params = {"adjusted": True, "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}

    try:
        r = _session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        j = r.json() or {}
        results = j.get("results", []) or []
        if not results:
            println(f"{ticker}: no data")
            return None

        # Build float32 series to reduce memory/disk footprint
        idx = pd.to_datetime([it.get("t") for it in results], unit="ms")
        data = pd.Series([it.get("c") for it in results], dtype="float32")
        s = pd.Series(index=idx, data=data.values, name=ticker).sort_index()
        return s

    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else None
        if code == 429:
            println(f"{ticker}: hit 429; handled by retry policy")
        else:
            println(f"{ticker} HTTP error: {code} | {e}")
        return None
    except Exception as e:
        println(f"{ticker} fetch failed: {e}")
        return None


def get_series_from_cache_or_api(
    ticker: str,
    start: str,
    end: str,
    multiplier: int = 1,
    timespan: str = "day",
    use_cache: bool = True,
) -> Optional[pd.Series]:
    """Resolution order: in-proc memo → parquet cache → HTTP fetch → write cache.
    """
    key = f"{ticker}|{start}|{end}|{multiplier}|{timespan}"
    if key in _INPROC_SERIES_CACHE:
        return _INPROC_SERIES_CACHE[key]

    path = _per_ticker_path(ticker, start, end, multiplier, timespan)
    if use_cache and path.exists():
        try:
            dfp = pd.read_parquet(path)
            # Be robust to column name encoding; take first column when needed
            s = dfp.iloc[:, 0] if ticker not in dfp.columns else dfp[ticker]
            s.name = ticker
            _INPROC_SERIES_CACHE[key] = s
            return s
        except Exception:
            # Fall through to refetch on cache read errors
            pass

    s = _fetch_single_series_from_api(ticker, start, end, multiplier, timespan)
    if s is not None:
        try:
            s.to_frame().to_parquet(path, compression="zstd", index=True)
        except Exception:
            # Cache write failure should not break the pipeline
            pass
        _INPROC_SERIES_CACHE[key] = s
    return s


# -----------------------------------------------------------------------------
# Public fetchers
# -----------------------------------------------------------------------------

def fetch_union_prices_polygon(
    tickers: List[str],
    start: str,
    end: str,
    multiplier: int = 1,
    timespan: str = "day",
    use_cache: bool = True,
    max_workers: int = 8,
) -> Optional[pd.DataFrame]:
    """Fetch prices for the union of tickers with per-ticker cache and thread pool.

    Returns a wide DataFrame (index=date, columns=tickers). Does NOT write CSV.
    """
    ensure_dirs()

    unique = sorted({t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()})
    if not unique:
        println("No tickers provided for union fetch")
        return None

    frames = []
    total = len(unique)

    # Threaded fetching with bounded concurrency
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                get_series_from_cache_or_api,
                t, start, end, multiplier, timespan, use_cache
            ): t for t in unique
        }
        done = 0
        for fut in cf.as_completed(futures):
            t = futures[fut]
            try:
                s = fut.result()
                if s is not None:
                    frames.append(s)
            except Exception as e:
                println(f"{t} future failed: {e}")
            done += 1
            progress(1, note=f"fetch {done}/{total}")

    if not frames:
        println("Empty records from union fetch")
        return None

    df = pd.concat(frames, axis=1).sort_index()
    return df


def fetch_prices_polygon(
    tickers: List[str],
    start: str,
    end: str,
    multiplier: int = 1,
    timespan: str = "day",
    max_workers: int = 8,
) -> Optional[str]:
    """Backward-compatible adapter: fetches and writes a CSV like the old behavior."""
    df = fetch_union_prices_polygon(
        tickers,
        start,
        end,
        multiplier=multiplier,
        timespan=timespan,
        use_cache=True,
        max_workers=max_workers,
    )
    if df is None or df.empty:
        return None
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = CSV_SAVE_DIR / f"prices_{ts}.csv"
    df.to_csv(out_csv)
    return str(out_csv)
