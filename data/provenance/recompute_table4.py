"""Recompute the paper's Table 4 from the archived 2026-01-09 ticker lists.

Nothing here calls an LLM. Table 4's six metric columns are a deterministic
function of (ticker list, price window), so they can be reproduced exactly
given the same inputs - which is what makes this check meaningful in a way
that re-running the agent never could be.

Price source: Yahoo's chart endpoint (free, no key, covers the full
2024-07-25..2025-08-07 window). Polygon's free tier only reaches back to
2024-08-27, so it cannot cover the window start; it is used separately as a
cross-check over the overlapping segment.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from portfolio.metrics import (  # noqa: E402
    annualized_return,
    annualized_volatility,
    equity_curve_from_equal_weight,
    hist_var_cvar,
    max_drawdown,
    sharpe_ratio,
)

START, END = "2024-07-25", "2025-08-07"

# Published Table 4 (typeset PDF p.10).
PAPER = {
    "claude sonnet":   (24,  0.1475, 0.1714, 0.9397, -0.1713, -0.0143, -0.0252),
    "grok 4":          (20,  0.1601, 0.1708, 1.0113, -0.1690, -0.0153, -0.0255),
    "llama4 maverick": (20,  0.0348, 0.1681, 0.3760, -0.1663, -0.0146, -0.0243),
    "deepseek-r1":     (20,  0.1348, 0.1857, 0.8312, -0.1843, -0.0165, -0.0264),
    "deepseek-v3":     (23,  0.1322, 0.1806, 0.8309, -0.1660, -0.0147, -0.0260),
    "gpt-4o":          (25,  0.1041, 0.1720, 0.7156, -0.1564, -0.0152, -0.0249),
    "gpt-4o-mini":     (25, -0.0571, 0.0974, 0.1765, -0.0413, -0.0103, -0.0120),
    "gpt-5-mini":      (22,  0.1513, 0.1715, 0.9774, -0.1582, -0.0154, -0.0251),
    "gpt-5-nano":      (20,  0.1136, 0.1700, 0.7765, -0.1683, -0.0148, -0.0256),
    "gpt-5":           (22,  0.0723, 0.1674, 0.5641, -0.1566, -0.0136, -0.0244),
    "o3":              (22,  0.0847, 0.1603, 0.6596, -0.1551, -0.0140, -0.0231),
    "o4-mini":         (21,  0.0562, 0.1715, 0.4591, -0.1659, -0.0145, -0.0253),
}

_UA = {"User-Agent": "Mozilla/5.0"}


def yahoo_symbol(t: str) -> str:
    """Yahoo writes class shares with a hyphen: BRK.B -> BRK-B."""
    return t.replace(".", "-")


def fetch_yahoo(ticker: str) -> pd.Series | None:
    p1 = int(dt.datetime.fromisoformat(START).timestamp())
    p2 = int((dt.datetime.fromisoformat(END) + dt.timedelta(days=1)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol(ticker)}"
    try:
        r = requests.get(url, params={"period1": p1, "period2": p2,
                                      "interval": "1d", "events": "div,split"},
                         headers=_UA, timeout=30)
        if r.status_code != 200:
            return None
        res = (r.json().get("chart") or {}).get("result")
        if not res:
            return None
        node = res[0]
        # Split-adjusted close WITHOUT dividend reinvestment, matching what
        # Polygon returns for adjusted=true. Yahoo's adjclose additionally
        # reinvests dividends, which adds a full dividend yield to every
        # annualised return and inflates Sharpe by ~10% on large caps.
        adj = (node["indicators"].get("quote") or [{}])[0].get("close")
        if not adj:
            return None
        idx = pd.to_datetime([dt.datetime.utcfromtimestamp(t).date() for t in node["timestamp"]])
        s = pd.Series(adj, index=idx, name=ticker).dropna()
        return s if len(s) else None
    except Exception:
        return None


def main() -> None:
    tick_path = REPO / "data/provenance/table45_portfolio_20260109/_tickers.json"
    portfolios: dict[str, list[str]] = json.loads(tick_path.read_text(encoding="utf-8"))
    universe = sorted({t for v in portfolios.values() for t in v})

    print(f"fetching {len(universe)} tickers from Yahoo ({START} .. {END})")
    prices, missing = {}, []
    for i, t in enumerate(universe, 1):
        s = fetch_yahoo(t)
        if s is None:
            missing.append(t)
            print(f"  [{i:>2}/{len(universe)}] {t:<7} MISSING")
        else:
            prices[t] = s
            print(f"  [{i:>2}/{len(universe)}] {t:<7} {len(s)} bars", end="\r")
        time.sleep(0.25)   # be polite to a free endpoint
    print(f"\n  fetched {len(prices)}, missing {len(missing)}: {missing}")

    px = pd.DataFrame(prices).sort_index()
    print(f"  price matrix: {px.shape[0]} rows x {px.shape[1]} cols, "
          f"{px.index.min().date()} .. {px.index.max().date()}")

    rows = []
    for label, tickers in portfolios.items():
        cols = [t for t in tickers if t in px.columns]
        df = px[cols].copy()
        eq = equity_curve_from_equal_weight(df)
        var95, cvar95 = hist_var_cvar(df, level=0.95)
        got = (len(df.columns), annualized_return(eq), annualized_volatility(df),
               sharpe_ratio(df), max_drawdown(eq), var95, cvar95)
        exp = PAPER[label]
        rows.append({
            "model": label,
            "n_paper": exp[0], "n_recomputed": got[0],
            "dropped": [t for t in tickers if t not in px.columns],
            "ret_paper": exp[1], "ret_recomputed": round(got[1], 4),
            "vol_paper": exp[2], "vol_recomputed": round(got[2], 4),
            "sharpe_paper": exp[3], "sharpe_recomputed": round(got[3], 4),
            "sharpe_diff_pct": round(abs(got[3] - exp[3]) / abs(exp[3]) * 100, 2),
            "mdd_paper": exp[4], "mdd_recomputed": round(got[4], 4),
            "var_paper": exp[5], "var_recomputed": round(got[5], 4),
            "cvar_paper": exp[6], "cvar_recomputed": round(got[6], 4),
        })

    out = pd.DataFrame(rows)
    dest = REPO / "data/provenance/table4_recomputed.csv"
    out.to_csv(dest, index=False)

    pd.set_option("display.width", 200)
    print("\n=== Table 4: paper vs recomputed (Yahoo adjclose) ===")
    print(out[["model", "n_paper", "n_recomputed", "sharpe_paper", "sharpe_recomputed",
               "sharpe_diff_pct", "ret_paper", "ret_recomputed",
               "vol_paper", "vol_recomputed"]].to_string(index=False))
    print(f"\nsaved -> {dest}")

    within2 = (out["sharpe_diff_pct"] <= 2).sum()
    print(f"\nSharpe within 2% of published: {within2}/{len(out)}")
    print(f"median Sharpe deviation: {out['sharpe_diff_pct'].median():.2f}%")


if __name__ == "__main__":
    main()
