from __future__ import annotations

import pandas as pd

from backend.scanner.broker_desk import analyze_symbol_brokers


def analyze_brokers(sym: str, row: pd.Series, broker_panel: pd.DataFrame) -> dict:
    desk = analyze_symbol_brokers(sym, broker_panel) if not broker_panel.empty else {}
    bp = float(row.get("broker_pressure") or 0)
    score = 45
    notes = []

    if bp >= 22:
        score += 25
        notes.append(f"Strong 1D–1W broker skew ({bp:.0f}/100)")
    elif bp >= 15:
        score += 12
        notes.append(f"Moderate broker pressure ({bp:.0f})")
    else:
        score -= 5
        notes.append(f"Weak broker skew ({bp:.0f})")

    if desk.get("top_broker_ids"):
        notes.append(f"Watch brokers active: {desk['top_broker_ids']}")

    if desk.get("circular_confirmed"):
        score -= 30
        notes.append("Confirmed circular — broker flow unreliable")
    elif desk.get("circular_flag"):
        score -= 15
        notes.append("Suspect circular activity")

    if desk.get("directional_pct", 100) >= 25:
        score += 8
        notes.append(f"Directional broker flow {desk.get('directional_pct')}%")

    return {
        "step": "Broker analysis",
        "score": int(min(100, max(0, score))),
        "pass": score >= 50 and not desk.get("circular_confirmed"),
        "notes": notes,
        **desk,
    }
