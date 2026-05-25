from __future__ import annotations

import pandas as pd

from backend.config_signals import get_signal_config
from backend.signals.effective_scores import effective_scores


def analyze_momentum(row: pd.Series) -> dict:
    cfg = get_signal_config()
    p_raw = float(row.get("p_long_momentum") or 0)
    p, ems, broker_p = effective_scores(row, cfg)
    fs = float(row.get("floorsheet_momentum_score") or 0)
    rank = float(row.get("early_rank_score") or 0)
    mtf = float(row.get("mtf_convergence") or 0)
    drs = float(row.get("distribution_risk_score") or 50)

    score = 40
    notes = []

    if p >= 0.55:
        score += 22
        notes.append(f"Effective P(long) {p:.0%}")
    elif p >= 0.38:
        score += 8
        notes.append(f"Neutral-positive P(long) {p:.0%}")
    else:
        score -= 8
        notes.append(f"Low effective P(long) {p:.0%} (raw {p_raw:.0%})")

    if ems >= 35:
        score += 18
        notes.append(f"Early momentum score {ems:.0f}")
    elif ems >= 20:
        score += 8
        notes.append(f"EMS building {ems:.0f}")
    else:
        notes.append(f"EMS weak ({ems:.0f}) — floorsheet {fs:.0f}")

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
