from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

def prepare_ercov(df_prices: pd.DataFrame, returns_method: str = "log"):
    if returns_method == "log":
        mu = expected_returns.ema_historical_return(df_prices, compounding=True)
    else:
        mu = expected_returns.mean_historical_return(df_prices, compounding=True)
    S = risk_models.CovarianceShrinkage(df_prices).ledoit_wolf()
    return mu, S

def optimize_weights(df_prices: pd.DataFrame,
                     objective: str = "max_sharpe",
                     target_return: Optional[float] = None,
                     weight_bounds: Tuple[float, float] = (0.0, 0.15),
                     l2_reg: float = 0.001) -> Dict[str, float]:
    mu, S = prepare_ercov(df_prices)
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    if l2_reg and l2_reg > 0:
        ef.add_objective(lambda w: l2_reg * np.sum(w**2))
    if objective == "min_volatility":
        ef.min_volatility()
    elif objective == "efficient_risk" and target_return is not None:
        ef.efficient_return(target_return)
    else:
        ef.max_sharpe()
    return {k: float(v) for k, v in ef.clean_weights(cutoff=1e-4).items()}

def discrete_allocation(df_prices: pd.DataFrame, weights: Dict[str, float], total_portfolio_value: int = 100000):
    latest_prices = get_latest_prices(df_prices)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)
    allocation, leftover = da.lp_portfolio()
    return allocation, float(leftover)
