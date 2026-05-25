"""Scanner universe tier assignment (isolated module for reliable imports)."""

from __future__ import annotations

import pandas as pd

from backend.signals.momentum_rules import assign_signal_tier

TIER_RANK = {"Invalidated": 0, "Neutral": 1, "Watch": 2, "Setup": 3, "Trigger": 4, "Confirmed": 5}
RANK_TIER = {v: k for k, v in TIER_RANK.items()}


def _pick_tier(a: str, b: str) -> str:
    return RANK_TIER[max(TIER_RANK.get(a, 1), TIER_RANK.get(b, 1))]


def assign_universe_tiers(df: pd.DataFrame) -> pd.Series:
    """Rule tiers + percentile ranks within the scanner universe."""
    if df.empty:
        return pd.Series(dtype=str)
    rules = df.apply(assign_signal_tier, axis=1)
    rank_pct = (
        df["early_rank_score"].rank(pct=True, method="average")
        if "early_rank_score" in df.columns
        else pd.Series(0.5, index=df.index)
    )
    pct_tier = pd.Series("Neutral", index=df.index)
    pct_tier[rank_pct >= 0.93] = "Trigger"
    pct_tier[(rank_pct >= 0.82) & (rank_pct < 0.93)] = "Setup"
    pct_tier[(rank_pct >= 0.55) & (rank_pct < 0.82)] = "Watch"

    drs = df.get("distribution_risk_score", pd.Series(0, index=df.index)).fillna(0)
    shakeout = df.get("dist_shakeout_flag", pd.Series(False, index=df.index)).fillna(False)
    pct_tier[(drs >= 88) & (~shakeout)] = "Invalidated"

    merged = [_pick_tier(rules.iloc[i], pct_tier.iloc[i]) for i in range(len(df))]

    if "llm_tier" in df.columns:
        for i in range(len(df)):
            lt = df.iloc[i].get("llm_tier")
            if isinstance(lt, str) and lt in TIER_RANK:
                merged[i] = _pick_tier(merged[i], lt)

    return pd.Series(merged, index=df.index)
