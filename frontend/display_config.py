"""Human-readable labels and chart styling for Streamlit."""

from __future__ import annotations

import pandas as pd

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
    "Confirmed": "Strong effective P(long) + EMS + early rank (rare)",
    "Trigger": "Actionable early setup — review for entry",
    "Setup": "Dist shakeout / broker skew building",
    "Watch": "Volume + early rank worth monitoring",
    "Neutral": "No clear edge yet",
    "Invalidated": "Heavy long-horizon distribution — avoid long",
}

COLUMN_LABELS = {
    "turnover_rank": "Turnover Rank (1D)",
    "volume_rank": "Turnover Rank (1D)",
    "early_pick_rank": "Early Pick #",
    "symbol": "Symbol",
    "ltp": "LTP (Rs)",
    "daily_volume": "Daily Qty",
    "daily_turnover_lac": "Turnover (Lac)",
    "early_rank_score": "Early Rank %",
    "signal_tier": "Signal",
    "p_long_momentum": "P(Long 10D) %",
    "expected_return_10d": "Exp Return 10D %",
    "early_momentum_score": "Early Momentum",
    "floorsheet_momentum_score": "Floorsheet",
    "broker_pressure": "Broker Pressure",
    "distribution_risk_score": "Dist Risk",
    "top_broker_ids": "Top Brokers (58,49…)",
    "top_broker_net_lac": "Top Broker Net (Lac)",
    "circular_risk": "Circular Risk %",
    "circular_flag": "Circular?",
    "circular_confirmed": "Circular Confirmed",
    "wash_score": "Wash %",
    "directional_pct": "Directional %",
    "reciprocal_brokers": "Recip. Brokers",
    "float_turnover_1d_abs": "Float Turn 1D",
    "llm_p_long": "LLM P(Long) %",
    "llm_note": "LLM Note",
}


def format_scanner_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ("ltp", "daily_volume", "daily_turnover_lac", "broker_pressure", "early_momentum_score"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "early_rank_score" in out.columns:
        out["early_rank_score"] = (pd.to_numeric(out["early_rank_score"], errors="coerce").fillna(0) * 100).round(1)
    if "p_long_momentum" in out.columns:
        out["p_long_momentum"] = (pd.to_numeric(out["p_long_momentum"], errors="coerce").fillna(0) * 100).round(1)
    if "llm_p_long" in out.columns:
        out["llm_p_long"] = (pd.to_numeric(out["llm_p_long"], errors="coerce") * 100).round(1)
        out["llm_p_long"] = out["llm_p_long"].where(out["llm_p_long"].notna(), None)
    if "expected_return_10d" in out.columns:
        out["expected_return_10d"] = pd.to_numeric(out["expected_return_10d"], errors="coerce").round(2)
    if "llm_note" in out.columns:
        out["llm_note"] = out["llm_note"].fillna("").replace("", "—")
    if "top_broker_ids" in out.columns:
        out["top_broker_ids"] = out["top_broker_ids"].fillna("—")
    if "ltp" in out.columns:
        out["ltp"] = out["ltp"].round(2)

    rename = {k: v for k, v in COLUMN_LABELS.items() if k in out.columns}
    return out.rename(columns=rename)
