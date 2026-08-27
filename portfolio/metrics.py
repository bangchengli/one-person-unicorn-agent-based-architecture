import math
import numpy as np
import pandas as pd

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

def equity_curve_from_weights(df: pd.DataFrame, weights: dict) -> pd.Series:
    df = df.dropna(how="all").ffill().dropna(axis=1, how="all")
    cols = [c for c in df.columns if c in weights]
    if not cols:
        return pd.Series(dtype=float)
    w = np.array([weights[c] for c in cols], dtype=float)
    w = w / w.sum() if w.sum() > 0 else w
    rets = df[cols].pct_change().dropna()
    if rets.empty:
        return pd.Series(dtype=float)
    port_ret = rets.dot(w)
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

def hist_var_cvar(df: pd.DataFrame, level: float = 0.95):
    rets = df.pct_change().dropna()
    if rets.empty:
        return (float("nan"), float("nan"))
    eq_w = np.repeat(1.0 / rets.shape[1], rets.shape[1])
    port_ret = rets.dot(eq_w)
    q = np.quantile(port_ret, 1 - level)
    cvar = port_ret[port_ret <= q].mean()
    return q, cvar
