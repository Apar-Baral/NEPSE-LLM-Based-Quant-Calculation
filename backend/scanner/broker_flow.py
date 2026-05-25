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
    NEPSE floorsheet semantics:
    - **Distribution analytics** (Seller Broker rows): Sell Qty = stock distributed,
      Buy Qty = buy-side absorption. High sell_qty alone is NOT bearish.
    - **Accumulation analytics**: net_qty = buy - sell is true desk bias.
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
    has_dist = not dist.empty and not has_acc

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
    elif has_dist:
        # Distribution sheet: judge by absorption % (buy / buy+sell), not raw sell > buy
        if buy_share >= 48:
            bias = "absorption"
            flow_label = f"Strong absorption — buyers took {buy_share:.0f}% of distribution flow"
            signal = "bullish"
        elif buy_share >= 35:
            bias = "dist_absorption"
            flow_label = f"Distribution with dip-buying — {buy_share:.0f}% absorbed (sell qty = supply, not crash)"
            signal = "bullish"
        elif buy_share >= 25:
            bias = "two_sided"
            flow_label = f"Two-sided distribution — {buy_share:.0f}% buy absorption"
            signal = "neutral"
        else:
            bias = "dist_heavy"
            flow_label = f"Weak absorption {buy_share:.0f}% — most flow is pure distribution"
            signal = "bearish"
    elif buy_share >= 52:
        bias = "absorption"
        flow_label = f"Buyer absorption {buy_share:.0f}% of desk flow"
        signal = "bullish"
    elif buy_share <= 22 and activity > 0:
        bias = "dist_heavy"
        flow_label = f"Low buy absorption {buy_share:.0f}%"
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
        "distribution_only": bool(has_dist),
    }


def symbol_absorption_summary(sym: str, broker_panel: pd.DataFrame) -> dict:
    """Symbol-level buy vs sell on distribution sheet with correct interpretation."""
    sym = str(sym).strip().upper()
    if broker_panel.empty:
        return {}

    horizons = ("1D", "2D", "1W", "1M", "3M") if "horizon" in broker_panel.columns else ("1D",)
    by_h: dict[str, dict] = {}
    for h in horizons:
        sub = broker_panel[(broker_panel["symbol"] == sym) & (broker_panel["horizon"] == h)]
        if sub.empty:
            continue
        m = broker_flow_metrics(sub)
        by_h[h] = {
            "buy_qty": m["buy_qty"],
            "sell_qty": m["sell_qty"],
            "absorption_pct": m["buy_share_pct"],
            "bias": m["bias"],
            "signal": m["signal"],
        }

    primary = by_h.get("1D") or by_h.get("1W") or next(iter(by_h.values()), {})
    buy = float(primary.get("buy_qty", 0))
    sell = float(primary.get("sell_qty", 0))
    abs_pct = float(primary.get("absorption_pct", 0))

    note = (
        "On **distribution** floorsheets, **Sell Qty** = shares supplied to market; "
        "**Buy Qty** = buy-side absorption. Compare **absorption %**, not raw sell > buy."
    )
    if abs_pct >= 35:
        verdict = "Buyers absorbing distribution (can rally with high sell qty)"
    elif abs_pct >= 25:
        verdict = "Mixed — some absorption under distribution"
    else:
        verdict = "Thin absorption — distribution-led"

    return {
        "symbol": sym,
        "buy_qty": buy,
        "sell_qty": sell,
        "absorption_pct": abs_pct,
        "verdict": verdict,
        "note": note,
        "by_horizon": by_h,
    }
