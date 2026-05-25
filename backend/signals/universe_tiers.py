"""Scanner universe tier assignment (isolated module for reliable imports)."""

from __future__ import annotations

import pandas as pd

from backend.signals.momentum_rules import assign_signal_tier

TIER_RANK = {"Invalidated": 0, "Neutral": 1, "Watch": 2, "Setup": 3, "Trigger": 4, "Confirmed": 5}
RANK_TIER = {v: k for k, v in TIER_RANK.items()}


def _cap_tier(tier: str, max_tier: str) -> str:
    if TIER_RANK.get(tier, 1) > TIER_RANK.get(max_tier, 5):
        return max_tier
    return tier


def assign_universe_tiers(df: pd.DataFrame) -> pd.Series:
    """
    Rule-based tiers first; percentile only nudges Neutral/Watch upward (no mass Confirmed).
    """
    if df.empty:
        return pd.Series(dtype=str)

    rules = df.apply(assign_signal_tier, axis=1)
    rank_pct = (
        df["early_rank_score"].rank(pct=True, method="average")
        if "early_rank_score" in df.columns
        else pd.Series(0.5, index=df.index)
    )

    drs = df.get("distribution_risk_score", pd.Series(0, index=df.index)).fillna(0)
    shakeout = df.get("dist_shakeout_flag", pd.Series(False, index=df.index)).fillna(False)
    if "pattern_dist_shakeout" in df.columns:
        shakeout = shakeout | df["pattern_dist_shakeout"].fillna(False).astype(bool)

    merged = []
    for i in range(len(df)):
        rule = rules.iloc[i]
        rp = float(rank_pct.iloc[i])
        dr = float(drs.iloc[i])
        sh = bool(shakeout.iloc[i])

        if dr >= 88 and not sh:
            merged.append("Invalidated")
            continue

        tier = rule
        # Percentile nudge only when rules say Neutral/Watch — never inflate to Confirmed
        if tier == "Neutral":
            if rp >= 0.92:
                tier = "Setup"
            elif rp >= 0.75:
                tier = "Watch"
        elif tier == "Watch" and rp >= 0.95:
            tier = _cap_tier("Setup", "Setup")

        # LLM tier: can adjust one step, capped at Trigger unless rules already Confirmed
        lt = df.iloc[i].get("llm_tier")
        if isinstance(lt, str) and lt in TIER_RANK:
            if lt == "Confirmed" and tier in ("Trigger", "Setup", "Watch"):
                tier = "Trigger"
            elif TIER_RANK.get(lt, 1) > TIER_RANK.get(tier, 1):
                tier = _cap_tier(lt, "Trigger")

        merged.append(tier)

    return pd.Series(merged, index=df.index)
