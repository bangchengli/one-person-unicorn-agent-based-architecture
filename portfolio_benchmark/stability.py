from itertools import combinations
import numpy as np
import pandas as pd

from .metrics import (
    equity_curve_from_equal_weight, max_drawdown, annualized_return, sharpe_ratio
)
from .market_data import fetch_prices_polygon
from .llm_portfolio import generate_portfolio_with_model

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

def stability_test_for_model(model: str, repeats: int, start: str, end: str, use_web_search: bool = True) -> dict:
    sharpe_list = []
    annret_list = []
    mdd_list = []
    tickers_runs = []

    for _ in range(repeats):
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
