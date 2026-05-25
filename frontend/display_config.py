"""Human-readable labels and chart styling for Streamlit."""

from __future__ import annotations

TIER_ORDER = ["Confirmed", "Trigger", "Setup", "Watch", "Neutral", "Invalidated"]

TIER_COLORS = {
    "Confirmed": "#00c853",
    "Trigger": "#69f0ae",
    "Setup": "#ffd740",
    "Watch": "#40c4ff",
    "Neutral": "#78909c",
    "Invalidated": "#ef5350",
}

TIER_HELP = {
    "Confirmed": "High probability + strong floorsheet early score",
    "Trigger": "Actionable early setup — review for entry",
    "Setup": "Dist shakeout / broker skew building",
    "Watch": "Volume + early rank worth monitoring",
    "Neutral": "No clear edge yet",
    "Invalidated": "Heavy long-horizon distribution — avoid long",
}

COLUMN_LABELS = {
    "volume_rank": "Vol Rank",
    "symbol": "Symbol",
    "ltp": "LTP (Rs)",
    "daily_volume": "Daily Qty",
    "daily_turnover_lac": "Turnover (Lac)",
    "early_rank_score": "Early Rank",
    "signal_tier": "Signal",
    "p_long_momentum": "P(Long 10D)",
    "expected_return_10d": "Exp Return 10D (%)",
    "early_momentum_score": "Early Momentum",
    "smart_money_score": "Smart Money",
    "distribution_risk_score": "Dist Risk",
    "broker_pressure": "Broker Pressure",
    "float_turnover_1d_abs": "Float Turn 1D",
    "analog_hit_rate": "Analog Hit Rate",
    "confidence": "Confidence",
    "short_ofi": "Short OFI",
    "heavy_broker_share": "Heavy Broker %",
}


def format_scanner_table(df):
    import pandas as pd

    out = df.copy()
    pct_cols = ("early_rank_score", "p_long_momentum", "analog_hit_rate", "confidence", "short_ofi", "heavy_broker_share")
    for col in pct_cols:
        if col in out.columns:
            out[col] = (pd.to_numeric(out[col], errors="coerce").fillna(0) * 100).round(1)
    if "early_momentum_score" in out.columns:
        out["early_momentum_score"] = pd.to_numeric(out["early_momentum_score"], errors="coerce").fillna(0).round(0)
    if "daily_volume" in out.columns:
        out["daily_volume"] = pd.to_numeric(out["daily_volume"], errors="coerce").fillna(0).round(0)
    rename = {k: v for k, v in COLUMN_LABELS.items() if k in out.columns}
    return out.rename(columns=rename)
