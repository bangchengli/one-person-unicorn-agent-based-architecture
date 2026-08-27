from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from core.market import fetch_ohlcv

FORECAST_TRADING_DAYS = 5
FORECAST_CALENDAR_WINDOW = 15


def _parse_price(value: Any) -> Optional[float]:
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def evaluate_support_resistance(ticker: str, report: dict, report_date: str,
                                current_close: float) -> Optional[dict]:
    levels = report.get("risk_levels") or {}
    supports = [p for p in (_parse_price(s) for s in levels.get("support") or []) if p is not None]
    resistances = [p for p in (_parse_price(r) for r in levels.get("resistance") or []) if p is not None]
    if not supports or not resistances:
        return None

    s1 = max(supports)
    r1 = min(resistances)

    try:
        as_of = datetime.strptime(report_date, "%Y-%m-%d")
    except ValueError:
        return None

    future = fetch_ohlcv(
        ticker,
        (as_of + timedelta(days=1)).strftime("%Y-%m-%d"),
        (as_of + timedelta(days=FORECAST_CALENDAR_WINDOW)).strftime("%Y-%m-%d"),
    )
    if future is None or future.empty:
        return None

    window = future.head(FORECAST_TRADING_DAYS)
    if window.empty:
        return None

    actual_low = float(window["low"].min())
    actual_high = float(window["high"].max())

    predicted_drawdown_tolerance = (current_close - s1) / current_close
    actual_max_drawdown = (current_close - actual_low) / current_close
    mae_margin = predicted_drawdown_tolerance - actual_max_drawdown

    in_band = int(((window["close"] >= s1) & (window["close"] <= r1)).sum())

    return {
        "predicted_support": s1,
        "predicted_resistance": r1,
        "actual_1w_low": actual_low,
        "actual_1w_high": actual_high,
        "support_breached": bool(actual_low < s1),
        "resistance_breached": bool(actual_high > r1),
        "mae_margin_pct": round(mae_margin * 100, 2),
        "coverage_ratio": round(in_band / len(window), 4),
        "days_scored": int(len(window)),
    }
