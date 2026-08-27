from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityConfig:
    min_list_len: int = 15
    max_list_len: int = 25
    min_liquidity_usd: float = 5_000_000.0
    target_sector_coverage: int = 8

    w_constraints: float = 0.45
    w_diversity:   float = 0.35
    w_llm_judge:   float = 0.20


def _safe_ratio(numer: float, denom: float) -> float:
    return 0.0 if denom == 0 else float(numer) / float(denom)


def score_constraints(
    tickers: Iterable[str],
    meta_df: pd.DataFrame,
    cfg: QualityConfig = QualityConfig(),
) -> float:
    tlist = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    n = len(tlist)
    if n == 0:
        return 0.0

    uniq = 1.0 if len(set(tlist)) == n else _safe_ratio(len(set(tlist)), n)

    real = meta_df.loc[meta_df["ticker"].isin(tlist)] if not meta_df.empty else pd.DataFrame()
    exist_ratio = _safe_ratio(len(real), n)

    if not real.empty and "avg_dollar_vol" in real.columns:
        liquid_ratio = float((real["avg_dollar_vol"] > cfg.min_liquidity_usd).mean())
    else:
        liquid_ratio = 0.0

    length_ok = 1.0 if (cfg.min_list_len <= n <= cfg.max_list_len) else 0.0

    return float(0.35 * uniq + 0.35 * exist_ratio + 0.20 * liquid_ratio + 0.10 * length_ok)


def score_diversity(
    tickers: Iterable[str],
    meta_df: pd.DataFrame,
    returns_df: Optional[pd.DataFrame] = None,
    cfg: QualityConfig = QualityConfig(),
) -> float:
    tset = {t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()}
    if not tset:
        return 0.0

    real = meta_df.loc[meta_df["ticker"].isin(tset)] if not meta_df.empty else pd.DataFrame()

    if not real.empty and "sector" in real.columns:
        sectors = int(real["sector"].nunique())
        sector_cov = min(1.0, _safe_ratio(sectors, cfg.target_sector_coverage))
    else:
        sector_cov = 0.0

    if returns_df is not None and not returns_df.empty:
        cols = [c for c in returns_df.columns if c in tset]
        if len(cols) >= 2:
            corr = returns_df[cols].pct_change().dropna().corr()
            if corr.shape[0] >= 2:
                tri = corr.values[np.triu_indices_from(corr, k=1)]
                avg_corr = float(np.nanmean(tri)) if tri.size else 1.0
            else:
                avg_corr = 1.0
        else:
            avg_corr = 1.0
    else:
        avg_corr = 1.0
    corr_score = 1.0 - np.clip((avg_corr + 1.0) / 2.0, 0.0, 1.0)

    n = len(tset)
    hhi = float(np.sum((np.ones(n) / n) ** 2))
    hhi_score = 1.0 - hhi

    return float(0.4 * sector_cov + 0.4 * corr_score + 0.2 * hhi_score)


JudgeFn = Callable[[str], Dict[str, float]]


def score_llm_judge(
    tickers: Iterable[str],
    judge_fn: Optional[JudgeFn] = None,
) -> float:
    if judge_fn is None:
        return 0.5

    tlist = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    prompt = (
        "You are a strict portfolio judge. Score each item in [0,1].\n"
        "Evaluate if this US equities list is diversified, coherent, and relevant.\n"
        f"Tickers: {tlist}\n"
        "Return pure JSON: {\"relevance\": x, \"coherence\": y, \"risk_control\": z}"
    )
    try:
        js = judge_fn(prompt) or {}
        rel = float(js.get("relevance", 0.0))
        coh = float(js.get("coherence", 0.0))
        risk = float(js.get("risk_control", 0.0))
        rel = float(np.clip(rel, 0.0, 1.0))
        coh = float(np.clip(coh, 0.0, 1.0))
        risk = float(np.clip(risk, 0.0, 1.0))
        return float(0.4 * rel + 0.4 * coh + 0.2 * risk)
    except Exception:
        return 0.5


def composite_quality(
    tickers: Iterable[str],
    meta_df: pd.DataFrame,
    returns_df: Optional[pd.DataFrame] = None,
    judge_fn: Optional[JudgeFn] = None,
    cfg: QualityConfig = QualityConfig(),
) -> Dict[str, float]:
    c = score_constraints(tickers, meta_df, cfg)
    d = score_diversity(tickers, meta_df, returns_df, cfg)
    j = score_llm_judge(tickers, judge_fn)
    overall = float(cfg.w_constraints * c + cfg.w_diversity * d + cfg.w_llm_judge * j)
    return {
        "quality_constraints": c,
        "quality_diversity": d,
        "quality_llm_judge": j,
        "quality_overall": overall,
    }


def build_meta_from_reference(ref_df: pd.DataFrame) -> pd.DataFrame:
    if ref_df is None or ref_df.empty:
        return pd.DataFrame(columns=["ticker", "sector", "avg_dollar_vol"])

    df = ref_df.copy()
    cols = {c.lower(): c for c in df.columns}

    tcol = cols.get("ticker") or cols.get("symbol")
    scol = cols.get("sector") or cols.get("industry")
    vcol = cols.get("avg_dollar_vol")

    out = pd.DataFrame()
    out["ticker"] = df[tcol].astype(str).str.upper() if tcol else ""
    out["sector"] = df[scol].astype(str) if scol else ""
    out["avg_dollar_vol"] = df[vcol].astype(float) if vcol else 0.0
    return out
