from __future__ import annotations

import pandas as pd


def detect_order_blocks(row: pd.Series) -> dict:
    """Demand/supply zones as order-block proxies from floorsheet tech zones."""
    ltp = float(row.get("ltp") or 0)
    demand = row.get("tech_demand_zone")
    supply = row.get("tech_supply_zone")
    notes = []
    score = 50

    d = float(demand) if pd.notna(demand) and demand else None
    s = float(supply) if pd.notna(supply) and supply else None

    ob_bias = "neutral"
    if ltp > 0 and d is not None:
        dist_d = (ltp - d) / ltp * 100
        if -2 <= dist_d <= 6:
            score += 20
            ob_bias = "at_demand_ob"
            notes.append(f"Price near demand order block ({d:.2f}, {dist_d:+.1f}%)")
        elif dist_d < -2:
            notes.append(f"Below demand zone — breakdown risk")

    if ltp > 0 and s is not None:
        dist_s = (s - ltp) / ltp * 100
        if dist_s < 8:
            score -= 8
            notes.append(f"Supply overhead at {s:.2f} ({dist_s:.1f}% away)")
        if 0 < dist_s <= 4:
            ob_bias = "under_supply_ob"
            notes.append("Under supply block — breakout needed")

    return {
        "order_block_bias": ob_bias,
        "demand_zone": d,
        "supply_zone": s,
        "order_block_score": int(min(100, max(0, score))),
        "notes": notes,
    }


def detect_fair_value_gaps(row: pd.Series, panel_sym: pd.DataFrame) -> dict:
    """
    FVG proxy: short-horizon net flow discontinuity vs LTP.
    Bullish FVG hint when 1D acc/dist net positive with light dist power.
    """
    notes = []
    score = 50
    bull_fvg = False
    bear_fvg = False

    d1_net = float(row.get("dist_1D_net_amount") or row.get("acc_1D_net_amount") or 0)
    d1_power = float(row.get("dist_1D_power_score") or 3)
    ofi = float(row.get("ofi") or 0)

    if ofi > 0.15:
        score += 15
        bull_fvg = True
        notes.append(f"Positive order-flow imbalance (OFI {ofi:.2f})")
    elif ofi < -0.15:
        score -= 12
        bear_fvg = True
        notes.append(f"Negative OFI ({ofi:.2f})")

    if d1_net > 0 and d1_power <= 1.5:
        score += 12
        bull_fvg = True
        notes.append("1D light distribution with positive net — bullish FVG-style absorption")
    elif d1_net < 0 and d1_power >= 2:
        score -= 10
        bear_fvg = True
        notes.append("Heavy 1D distribution pressure")

    if not panel_sym.empty and "ltp" in panel_sym.columns:
        ltps = pd.to_numeric(panel_sym["ltp"], errors="coerce").dropna()
        if len(ltps) >= 2:
            gap_pct = abs(ltps.iloc[-1] - ltps.iloc[0]) / (ltps.iloc[0] + 1e-9) * 100
            if gap_pct >= 2:
                notes.append(f"LTP gap {gap_pct:.1f}% across horizons — price inefficiency")

    return {
        "step": "Price action",
        "score": int(min(100, max(0, score))),
        "pass": score >= 52,
        "bullish_fvg": bull_fvg,
        "bearish_fvg": bear_fvg,
        "notes": notes,
        **detect_order_blocks(row),
    }
