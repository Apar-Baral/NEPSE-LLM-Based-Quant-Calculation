"""NEPSE floorsheet broker flow semantics (accumulation vs distribution)."""

from __future__ import annotations

import pandas as pd


def _side_split(grp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "side" not in grp.columns:
        return grp.iloc[0:0], grp
    acc = grp[grp["side"].astype(str).str.lower() == "accumulation"]
    dist = grp[grp["side"].astype(str).str.lower() == "distribution"]
    return acc, dist


def broker_flow_metrics(grp: pd.DataFrame) -> dict:
    """
    Correct bias for NEPSE floorsheet:
    - Distribution rows: high sell_qty is normal (brokers distributing), not generic 'sell'.
    - Accumulation rows: net_qty = buy - sell is true accumulation bias.
    - Combined long_pressure uses acc net minus dist sell pressure.
    """
    buy = pd.to_numeric(grp.get("buy_qty", 0), errors="coerce").fillna(0).sum()
    sell = pd.to_numeric(grp.get("sell_qty", 0), errors="coerce").fillna(0).sum()
    net_qty = pd.to_numeric(grp.get("net_qty", 0), errors="coerce").fillna(0).sum()
    net_amt = pd.to_numeric(grp.get("net_amount", 0), errors="coerce").fillna(0).sum()
    activity = float(buy + sell)

    acc, dist = _side_split(grp)
    acc_net = pd.to_numeric(acc.get("net_qty", 0), errors="coerce").fillna(0).sum() if not acc.empty else 0.0
    dist_net = pd.to_numeric(dist.get("net_qty", 0), errors="coerce").fillna(0).sum() if not dist.empty else 0.0
    has_acc = not acc.empty and activity > 0
    has_dist = not dist.empty

    buy_share = (buy / (activity + 1e-9)) * 100 if activity > 0 else 0.0
    long_pressure = float(acc_net - dist_net) if has_acc or has_dist else float(net_qty)

    if has_acc and acc_net > 50:
        bias = "acc_buy"
        flow_label = f"Accumulation net +{acc_net:,.0f}"
        signal = "bullish"
    elif has_acc and acc_net < -50:
        bias = "acc_sell"
        flow_label = f"Accumulation net {acc_net:,.0f}"
        signal = "bearish"
    elif buy_share >= 52:
        bias = "absorption"
        flow_label = f"Buyer absorption {buy_share:.0f}% of desk flow"
        signal = "bullish"
    elif buy_share <= 38 and activity > 0:
        bias = "dist_heavy"
        flow_label = f"Distribution-heavy {100 - buy_share:.0f}% sell-side (not all brokers bearish)"
        signal = "bearish"
    elif activity <= 0:
        bias = "—"
        flow_label = "No activity"
        signal = "neutral"
    else:
        bias = "two_sided"
        flow_label = f"Two-sided flow ({buy_share:.0f}% buy share)"
        signal = "neutral"

    two_side = min(buy, sell) / (activity + 1e-9) * 100
    directional = abs(long_pressure) / (activity + 1e-9) * 100
    share = 0.0  # filled by caller

    return {
        "buy_qty": float(buy),
        "sell_qty": float(sell),
        "net_qty": float(net_qty),
        "net_amount_lac": float(net_amt),
        "long_pressure_qty": long_pressure,
        "buy_share_pct": round(buy_share, 1),
        "activity_qty": activity,
        "two_side_pct": round(two_side, 2),
        "directional_pct": round(directional, 2),
        "bias": bias,
        "flow_label": flow_label,
        "signal": signal,
        "has_accumulation": bool(has_acc),
        "has_distribution": bool(has_dist),
    }
