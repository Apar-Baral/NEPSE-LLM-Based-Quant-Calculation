from __future__ import annotations

import pandas as pd

from backend.quant.broker_quant import analyze_brokers
from backend.quant.llm_verify import verify_with_llm
from backend.quant.momentum_quant import analyze_momentum
from backend.quant.price_action import detect_fair_value_gaps
from backend.quant.volumetric import analyze_volume


def run_quant_analysis(
    sym: str,
    row: pd.Series,
    panel_sym: pd.DataFrame,
    broker_panel: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    run_llm: bool = False,
) -> dict:
    """Full multi-step quant pipeline with composite confirmation."""
    sym = str(sym).strip().upper()

    steps = [
        analyze_volume(row, universe),
        analyze_brokers(sym, row, broker_panel),
        detect_fair_value_gaps(row, panel_sym),
        analyze_momentum(row, universe),
    ]

    if run_llm:
        steps.append(verify_with_llm(sym, row, steps))

    weights = [0.22, 0.22, 0.22, 0.24, 0.10] if run_llm else [0.25, 0.25, 0.25, 0.25]
    composite = sum(s["score"] * w for s, w in zip(steps, weights[: len(steps)]))

    turn = float(row.get("daily_turnover_lac") or 0)
    rank = float(row.get("early_rank_score") or 0)
    tier = str(row.get("signal_tier", "Neutral"))
    fs = float(row.get("floorsheet_momentum_score") or 0)
    if turn >= 400 and rank >= 0.10:
        composite = min(100, composite + 8)
    elif turn >= 150 and tier in ("Trigger", "Confirmed"):
        composite = min(100, composite + 6)
    if fs >= 60 and tier in ("Setup", "Trigger", "Confirmed"):
        composite = min(100, composite + 4)

    passes = sum(1 for s in steps if s.get("pass"))
    if composite >= 72 and passes >= 3 and tier in ("Trigger", "Confirmed"):
        verdict = "Strong long bias"
    elif composite >= 58 and (tier in ("Trigger", "Confirmed", "Setup") or turn >= 200):
        verdict = "Positive — early momentum candidate"
    elif composite >= 45:
        verdict = "Caution — mixed signals"
    else:
        verdict = "Weak / avoid long"

    return {
        "symbol": sym,
        "composite_score": int(round(composite)),
        "steps_passed": passes,
        "steps_total": len(steps),
        "verdict": verdict,
        "steps": steps,
        "p_long_display": float(steps[3].get("p_long_effective", row.get("p_long_momentum") or 0)),
        "ems_display": float(steps[3].get("ems_effective", row.get("early_momentum_score") or 0)),
    }
