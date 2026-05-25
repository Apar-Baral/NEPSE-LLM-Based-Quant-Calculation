from __future__ import annotations

import pandas as pd

from backend.config_signals import get_signal_config
from backend.signals.effective_scores import effective_scores


def analyze_momentum(row: pd.Series, universe: pd.DataFrame | None = None) -> dict:
    cfg = get_signal_config()
    p_raw = float(row.get("p_long_momentum_raw") or row.get("p_long_momentum") or 0)
    p, ems, broker_p = effective_scores(row, cfg)
    fs = float(row.get("floorsheet_momentum_score") or 0)
    rank = float(row.get("early_rank_score") or 0)
    mtf = float(row.get("mtf_convergence") or 0)
    drs = float(row.get("distribution_risk_score") or 50)
    turn = float(row.get("daily_turnover_lac") or 0)
    tier = str(row.get("signal_tier", "Neutral"))

    score = 35
    notes = []

    turn_pct = 50.0
    if universe is not None and not universe.empty and "daily_turnover_lac" in universe.columns:
        u = pd.to_numeric(universe["daily_turnover_lac"], errors="coerce").fillna(0)
        if u.max() > 0 and turn > 0:
            turn_pct = float((u < turn).mean() * 100)
    if turn_pct >= 92:
        score += 22
        notes.append(f"Top turnover name ({turn:,.0f} Lac — top {100-turn_pct:.0f}% of market)")
    elif turn_pct >= 75:
        score += 12
        notes.append(f"High liquidity ({turn:,.0f} Lac)")
    elif turn < 5:
        score -= 12
        notes.append(f"Very thin turnover ({turn:,.0f} Lac)")

    if tier in ("Trigger", "Confirmed"):
        score += 15
        notes.append(f"Scanner tier **{tier}**")
    elif tier == "Setup":
        score += 8
        notes.append(f"Scanner tier **{tier}**")
    elif tier == "Invalidated":
        score -= 20
        notes.append(f"Scanner tier **{tier}**")

    if p >= 0.55:
        score += 22
        notes.append(f"Effective P(long) {p:.0%}")
    elif p >= 0.38:
        score += 8
        notes.append(f"Neutral-positive P(long) {p:.0%}")
    else:
        score -= 8
        notes.append(f"Low effective P(long) {p:.0%} (raw {p_raw:.0%})")

    if fs >= 55:
        score += 20
        notes.append(f"Floorsheet momentum {fs:.0f}/100")
    elif fs >= 35:
        score += 10
        notes.append(f"Floorsheet building {fs:.0f}")

    if ems >= 35:
        score += 18
        notes.append(f"Early momentum score {ems:.0f}")
    elif ems >= 20:
        score += 8
        notes.append(f"EMS building {ems:.0f}")
    else:
        notes.append(f"EMS {ems:.0f} (floorsheet {fs:.0f}, rank {rank:.2f})")

    if rank >= 0.12:
        score += 12
        notes.append(f"Early rank {rank:.2f}")
    elif rank >= 0.08:
        score += 6

    if mtf >= 0.6:
        score += 10
        notes.append(f"Multi-timeframe convergence {mtf:.0%}")

    if drs >= 75:
        score -= 15
        notes.append(f"Distribution risk {drs:.0f}")

    shakeout = row.get("pattern_dist_shakeout") or row.get("dist_shakeout_flag")
    if shakeout in (True, 1, "True"):
        score += 12
        notes.append("Distribution shakeout pattern")

    return {
        "step": "Quant momentum",
        "score": int(min(100, max(0, score))),
        "pass": score >= 55,
        "p_long_effective": round(p, 4),
        "p_long_raw": round(p_raw, 4),
        "ems_effective": round(ems, 1),
        "floorsheet_score": round(fs, 1),
        "early_rank": round(rank, 4),
        "notes": notes,
    }
