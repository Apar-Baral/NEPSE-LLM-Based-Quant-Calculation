from __future__ import annotations

import pandas as pd


def analyze_volume(row: pd.Series, universe: pd.DataFrame | None = None) -> dict:
    turn = float(row.get("daily_turnover_lac") or 0)
    qty = float(row.get("daily_volume") or 0)
    ft = float(row.get("float_turnover_1d_abs") or row.get("float_turnover_zscore_hv") or 0)
    z = float(row.get("float_turnover_zscore") or 0)

    turn_pct = 50.0
    if universe is not None and not universe.empty and "daily_turnover_lac" in universe.columns:
        u_turn = pd.to_numeric(universe["daily_turnover_lac"], errors="coerce").fillna(0)
        if u_turn.max() > 0:
            turn_pct = float((u_turn < turn).mean() * 100)

    score = 40.0
    notes = []
    if turn >= 150:
        score += 25
        notes.append(f"High 1D turnover ({turn:,.0f} Lac) — top {100-turn_pct:.0f}% of scanner")
    elif turn >= 60:
        score += 12
        notes.append(f"Moderate turnover ({turn:,.0f} Lac)")
    else:
        score -= 10
        notes.append(f"Thin turnover ({turn:,.0f} Lac)")

    if z >= 1.5:
        score += 15
        notes.append(f"Float turnover z-score {z:.2f} — unusual activity")
    elif z >= 0.8:
        score += 5

    if qty >= 50000:
        score += 10
        notes.append(f"Daily qty {qty:,.0f}")

    return {
        "step": "Volumetric",
        "score": int(min(100, max(0, score))),
        "pass": score >= 55,
        "notes": notes,
        "turnover_lac": turn,
        "turnover_percentile": round(100 - turn_pct, 1),
        "float_z": z,
    }
