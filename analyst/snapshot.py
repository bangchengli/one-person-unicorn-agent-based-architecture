from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if len(series) <= period:
        return pd.Series([float("nan")] * len(series), index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _annualised_vol(returns: pd.Series) -> Optional[float]:
    return float(returns.std(ddof=1) * math.sqrt(252)) if len(returns) > 1 else None


def _max_drawdown(equity: pd.Series) -> Optional[float]:
    if not len(equity):
        return None
    return float((equity / equity.cummax() - 1.0).min())


def compute_snapshot(prices: pd.DataFrame) -> Optional[dict]:
    if prices is None or prices.empty or "close" not in prices.columns:
        return None

    close = prices["close"].astype(float)
    high = prices["high"].astype(float)
    low = prices["low"].astype(float)
    volume = prices["volume"].astype(float)
    rets = close.pct_change().dropna()
    equity = (1.0 + rets).cumprod()

    n = len(prices)
    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None

    if n >= 15:
        prev_close = close.shift(1)
        tr = pd.concat([high - low,
                        (high - prev_close).abs(),
                        (low - prev_close).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])
    else:
        atr14 = None

    if n >= 20:
        typical = (high + low + close) / 3
        vwap20 = float((typical * volume).rolling(20).sum().iloc[-1]
                       / volume.rolling(20).sum().iloc[-1])
    else:
        vwap20 = None

    trend = "unknown"
    if ma20 is not None and ma50 is not None:
        trend = "bullish" if ma20 > ma50 else "bearish"

    return {
        "available": True,
        "last_close": float(close.iloc[-1]),
        "high_20d": float(high.rolling(20).max().iloc[-1]) if len(high) >= 20 else None,
        "low_20d": float(low.rolling(20).min().iloc[-1]) if len(low) >= 20 else None,
        "atr14": atr14,
        "vwap20": vwap20,
        "return_5d": float((1.0 + rets.tail(5)).prod() - 1.0) if len(rets) >= 5 else None,
        "return_20d": float((1.0 + rets.tail(20)).prod() - 1.0) if len(rets) >= 20 else None,
        "ann_vol": _annualised_vol(rets),
        "max_drawdown": _max_drawdown(equity),
        "rsi14": float(rsi(close, 14).iloc[-1]) if len(close) >= 15 else None,
        "ma20": ma20,
        "ma50": ma50,
        "trend_signal": trend,
        "n_obs": int(n),
        "start_date": str(prices.index.min().date()),
        "end_date": str(prices.index.max().date()),
    }
