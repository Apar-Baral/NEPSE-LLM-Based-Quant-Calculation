from __future__ import annotations

import pandas as pd


def run_backtest(
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame,
    entry_tier: str = "Trigger",
    hold_days: int = 10,
) -> dict:
    if signals.empty or ohlcv.empty:
        return {"trades": 0, "win_rate": 0, "avg_return": 0, "cagr_proxy": 0, "details": []}

    entries = signals[signals["signal_tier"].isin([entry_tier, "Confirmed"])].copy()
    ohlcv = ohlcv.copy()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.normalize()

    trades = []
    for _, row in entries.iterrows():
        sym = row["symbol"]
        rd = pd.Timestamp(row["report_date"]).normalize()
        sym_px = ohlcv[(ohlcv["symbol"] == sym) & (ohlcv["date"] >= rd)].sort_values("date")
        if len(sym_px) < 2:
            continue
        entry_px = sym_px.iloc[0]["close"]
        exit_idx = min(hold_days, len(sym_px) - 1)
        exit_px = sym_px.iloc[exit_idx]["close"]
        if not entry_px or pd.isna(entry_px):
            continue
        ret = (exit_px - entry_px) / entry_px * 100
        trades.append({"symbol": sym, "entry_date": str(rd.date()), "return_pct": ret, "tier": row["signal_tier"]})

    if not trades:
        return {"trades": 0, "win_rate": 0, "avg_return": 0, "cagr_proxy": 0, "details": []}

    tdf = pd.DataFrame(trades)
    win_rate = (tdf["return_pct"] > 0).mean()
    avg_ret = tdf["return_pct"].mean()
    return {
        "trades": len(tdf),
        "win_rate": float(win_rate),
        "avg_return": float(avg_ret),
        "cagr_proxy": float(avg_ret * (252 / hold_days) / 100),
        "details": trades[:100],
    }
