"""Classical quant & financial metrics used by agents and backtest analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> float:
    """Relative Strength Index (Wilder) on last `period` closes."""
    s = pd.to_numeric(close, errors="coerce").dropna()
    if len(s) < period + 1:
        return 50.0
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
    return float(100 - (100 / (1 + rs)))


def rate_of_change(close: pd.Series, period: int = 10) -> float:
    """Momentum ROC % over `period` bars."""
    s = pd.to_numeric(close, errors="coerce").dropna()
    if len(s) <= period:
        return 0.0
    return float((s.iloc[-1] / (s.iloc[-period - 1] + 1e-9) - 1) * 100)


def bollinger_pct_b(close: pd.Series, window: int = 20, num_std: float = 2.0) -> float:
    """%B position within Bollinger bands (0 = lower, 1 = upper)."""
    s = pd.to_numeric(close, errors="coerce").dropna()
    if len(s) < window:
        return 0.5
    mid = s.rolling(window).mean().iloc[-1]
    std = s.rolling(window).std().iloc[-1]
    upper = mid + num_std * std
    lower = mid - num_std * std
    if upper <= lower:
        return 0.5
    return float((s.iloc[-1] - lower) / (upper - lower + 1e-9))


def sharpe_proxy(returns_pct: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sharpe proxy from daily/simple return % series."""
    r = pd.to_numeric(returns_pct, errors="coerce").dropna()
    if len(r) < 3:
        return 0.0
    excess = r - risk_free
    std = excess.std()
    if std < 1e-9:
        return 0.0
    return float(excess.mean() / std * np.sqrt(252))


def max_drawdown_pct(close: pd.Series) -> float:
    """Peak-to-trough drawdown % on price series."""
    s = pd.to_numeric(close, errors="coerce").dropna()
    if len(s) < 2:
        return 0.0
    peak = s.cummax()
    dd = (s - peak) / (peak + 1e-9) * 100
    return float(dd.min())


def herfindahl_index(shares: pd.Series) -> float:
    """Broker concentration HHI (0–10000 scale → normalized 0–100)."""
    w = pd.to_numeric(shares, errors="coerce").fillna(0)
    total = w.sum()
    if total <= 0:
        return 0.0
    p = w / total
    hhi = float((p ** 2).sum() * 10000)
    return min(100.0, hhi / 100)


def z_score_reversion(series: pd.Series, window: int = 20) -> float:
    """Latest z-score vs rolling mean (mean-reversion signal)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < window:
        return 0.0
    roll = s.rolling(window)
    z = (s.iloc[-1] - roll.mean().iloc[-1]) / (roll.std().iloc[-1] + 1e-9)
    return float(z)


def analyze_price_series(close: pd.Series) -> dict:
    """Bundle TA metrics for agents / UI."""
    if close is None or len(pd.to_numeric(close, errors="coerce").dropna()) < 2:
        return {"bars": 0, "rsi": 50.0, "roc": 0.0, "bb_pct_b": 0.5, "max_dd": 0.0, "sharpe": 0.0}
    c = pd.to_numeric(close, errors="coerce").dropna()
    rets = c.pct_change().dropna() * 100
    return {
        "bars": len(c),
        "rsi": round(rsi(c), 1),
        "roc": round(rate_of_change(c), 2),
        "bb_pct_b": round(bollinger_pct_b(c), 3),
        "max_dd": round(max_drawdown_pct(c), 2),
        "sharpe": round(sharpe_proxy(rets), 2),
    }
