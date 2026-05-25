from __future__ import annotations

import pandas as pd


def build_price_series_from_features(features: pd.DataFrame) -> pd.DataFrame:
    """Multi-day LTP proxy from feature matrix (one row per symbol per report_date)."""
    if features.empty or "ltp" not in features.columns:
        return pd.DataFrame()
    df = features[["symbol", "report_date", "ltp"]].copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["report_date"]).dt.normalize()
    df["close"] = pd.to_numeric(df["ltp"], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
    return df[["symbol", "date", "close"]].sort_values(["symbol", "date"])


def merge_ohlcv_sources(ohlcv: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
    frames = []
    if not ohlcv.empty and "close" in ohlcv.columns:
        o = ohlcv.copy()
        o["symbol"] = o["symbol"].astype(str).str.upper()
        o["date"] = pd.to_datetime(o["date"]).dt.normalize()
        frames.append(o[["symbol", "date", "close"]])
    feat_px = build_price_series_from_features(features) if features is not None else pd.DataFrame()
    if not feat_px.empty:
        frames.append(feat_px)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
    return combined.sort_values(["symbol", "date"])


def run_backtest(
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame,
    entry_tier: str = "Trigger",
    hold_days: int = 10,
    features: pd.DataFrame | None = None,
) -> dict:
    prices = merge_ohlcv_sources(ohlcv, features)
    meta = {
        "price_rows": len(prices),
        "price_symbols": int(prices["symbol"].nunique()) if not prices.empty else 0,
        "price_dates": int(prices["date"].nunique()) if not prices.empty else 0,
        "signal_rows": len(signals),
    }

    if signals.empty:
        return {
            "trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "cagr_proxy": 0,
            "details": [],
            "message": "No predictions — run pipeline first.",
            **meta,
        }

    if prices.empty:
        return {
            "trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "cagr_proxy": 0,
            "details": [],
            "message": "No price history. Upload OHLCV CSV or run pipeline on multiple report dates.",
            **meta,
        }

    tier_set = [entry_tier]
    if entry_tier == "Trigger":
        tier_set.append("Confirmed")

    entries = signals[signals["signal_tier"].isin(tier_set)].copy()
    entries["report_date"] = pd.to_datetime(entries["report_date"]).dt.normalize()
    entries["symbol"] = entries["symbol"].astype(str).str.upper()

    if entries.empty:
        return {
            "trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "cagr_proxy": 0,
            "details": [],
            "message": f"No signals with tier {entry_tier} / Confirmed on saved predictions.",
            **meta,
        }

    prices = prices.copy()
    prices["symbol"] = prices["symbol"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()

    trades = []
    skipped = {"no_prices": 0, "single_day": 0, "bad_entry": 0}

    for _, row in entries.iterrows():
        sym = row["symbol"]
        rd = pd.Timestamp(row["report_date"]).normalize()
        sym_px = prices[prices["symbol"] == sym].sort_values("date")
        if sym_px.empty:
            skipped["no_prices"] += 1
            continue

        on_or_after = sym_px[sym_px["date"] >= rd]
        if on_or_after.empty:
            on_or_after = sym_px.tail(1)
        if len(sym_px) < 2 and len(on_or_after) < 2:
            skipped["single_day"] += 1
            continue

        entry_px = float(on_or_after.iloc[0]["close"])
        if not entry_px or pd.isna(entry_px):
            skipped["bad_entry"] += 1
            continue

        future = on_or_after.iloc[1:]
        if future.empty:
            skipped["single_day"] += 1
            continue

        exit_idx = min(hold_days - 1, len(future) - 1)
        exit_px = float(future.iloc[exit_idx]["close"])
        if not exit_px or pd.isna(exit_px):
            skipped["bad_entry"] += 1
            continue

        ret = (exit_px - entry_px) / entry_px * 100
        trades.append(
            {
                "symbol": sym,
                "entry_date": str(on_or_after.iloc[0]["date"].date()),
                "exit_date": str(future.iloc[exit_idx]["date"].date()),
                "return_pct": round(ret, 2),
                "tier": row["signal_tier"],
                "hold_days_used": exit_idx + 1,
            }
        )

    if not trades:
        return {
            "trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "cagr_proxy": 0,
            "details": [],
            "message": (
                f"0 trades filled. Tier matches: {len(entries)}. "
                f"Need 2+ price dates per symbol (upload more daily files). "
                f"Skipped: {skipped}"
            ),
            "skipped": skipped,
            **meta,
        }

    tdf = pd.DataFrame(trades)
    win_rate = (tdf["return_pct"] > 0).mean()
    avg_ret = tdf["return_pct"].mean()
    return {
        "trades": len(tdf),
        "win_rate": float(win_rate),
        "avg_return": float(avg_ret),
        "cagr_proxy": float(avg_ret * (252 / max(hold_days, 1)) / 100),
        "details": trades[:100],
        "message": "ok",
        "skipped": skipped,
        **meta,
    }
