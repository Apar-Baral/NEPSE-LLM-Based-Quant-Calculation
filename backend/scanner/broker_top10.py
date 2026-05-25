"""Mathematical analysis for top N brokers (market-wide + per symbol)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import load_yaml_config

SHORT_HORIZONS = ("1D", "2D", "3D", "4D", "1W")


def _broker_cfg() -> dict:
    cfg = load_yaml_config("settings.yaml").get("brokers", {})
    return {
        "analyze_top_n": int(cfg.get("analyze_top_n", 10)),
        "watch_list": [str(int(b)) for b in cfg.get("watch_list", [58, 49, 45, 55, 34, 6, 38, 43, 70])],
    }


def discover_top_brokers(broker_panel: pd.DataFrame, horizon: str = "1D", top_n: int | None = None) -> list[str]:
    """Rank brokers by total market activity (mathematical discovery)."""
    top_n = top_n or _broker_cfg()["analyze_top_n"]
    if broker_panel.empty:
        return _broker_cfg()["watch_list"][:top_n]

    sub = broker_panel[broker_panel["horizon"] == horizon].copy() if "horizon" in broker_panel.columns else broker_panel.copy()
    if sub.empty:
        sub = broker_panel[broker_panel["horizon"].isin(SHORT_HORIZONS)].copy() if "horizon" in broker_panel.columns else broker_panel

    sub["broker_id"] = sub["broker_id"].astype(str)
    sub["activity_qty"] = pd.to_numeric(sub.get("activity_qty", 0), errors="coerce").fillna(0)
    if "buy_qty" in sub.columns and "sell_qty" in sub.columns:
        sub["activity_qty"] = sub["activity_qty"].where(
            sub["activity_qty"] > 0,
            pd.to_numeric(sub["buy_qty"], errors="coerce").fillna(0)
            + pd.to_numeric(sub["sell_qty"], errors="coerce").fillna(0),
        )

    agg = (
        sub.groupby("broker_id")["activity_qty"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    discovered = list(agg.index.astype(str))
    # Merge with configured watch list (preserve order: discovered first, then watch extras)
    watch = _broker_cfg()["watch_list"]
    merged = discovered + [b for b in watch if b not in discovered]
    return merged[:top_n]


def analyze_broker_row(broker_id: str, grp: pd.DataFrame, symbol_activity: float) -> dict:
    buy = pd.to_numeric(grp.get("buy_qty", 0), errors="coerce").fillna(0).sum()
    sell = pd.to_numeric(grp.get("sell_qty", 0), errors="coerce").fillna(0).sum()
    net_qty = pd.to_numeric(grp.get("net_qty", 0), errors="coerce").fillna(0).sum()
    net_amt = pd.to_numeric(grp.get("net_amount", 0), errors="coerce").fillna(0).sum()
    activity = buy + sell
    two_side = min(buy, sell) / (buy + sell + 1e-9) * 100
    share = activity / (symbol_activity + 1e-9) * 100
    directional = abs(net_qty) / (activity + 1e-9) * 100
    # Broker conviction score 0-100
    conviction = (
        min(40, share * 0.4)
        + min(30, directional * 0.3)
        + min(20, abs(net_amt) / (abs(net_amt) + activity + 1e-9) * 100 * 0.2)
        - min(15, two_side * 0.15)
    )
    return {
        "broker_id": broker_id,
        "buy_qty": float(buy),
        "sell_qty": float(sell),
        "net_qty": float(net_qty),
        "net_amount_lac": float(net_amt),
        "activity_qty": float(activity),
        "share_pct": round(share, 2),
        "two_side_pct": round(two_side, 2),
        "directional_pct": round(directional, 2),
        "conviction_score": round(max(0, min(100, conviction)), 1),
        "bias": "buy" if net_qty > 0 else ("sell" if net_qty < 0 else "neutral"),
    }


def symbol_top_brokers_table(
    sym: str,
    broker_panel: pd.DataFrame,
    horizon: str = "1D",
    top_n: int | None = None,
) -> pd.DataFrame:
    """Full mathematical breakdown for top N brokers on a symbol."""
    top_n = top_n or _broker_cfg()["analyze_top_n"]
    if broker_panel.empty:
        return pd.DataFrame()

    sub = broker_panel[(broker_panel["symbol"] == sym) & (broker_panel["horizon"] == horizon)].copy()
    if sub.empty:
        sub = broker_panel[broker_panel["symbol"] == sym].copy()
        if "horizon" in sub.columns:
            sub = sub[sub["horizon"].isin(SHORT_HORIZONS)]

    if sub.empty:
        return pd.DataFrame()

    sub["broker_id"] = sub["broker_id"].astype(str)
    total_act = (
        pd.to_numeric(sub.get("buy_qty", 0), errors="coerce").fillna(0).sum()
        + pd.to_numeric(sub.get("sell_qty", 0), errors="coerce").fillna(0).sum()
    )

    top_ids = discover_top_brokers(broker_panel, horizon, top_n)
    rows = []
    for bid in top_ids:
        grp = sub[sub["broker_id"] == bid]
        if grp.empty:
            rows.append(
                {
                    "broker_id": bid,
                    "buy_qty": 0.0,
                    "sell_qty": 0.0,
                    "net_qty": 0.0,
                    "net_amount_lac": 0.0,
                    "activity_qty": 0.0,
                    "share_pct": 0.0,
                    "two_side_pct": 0.0,
                    "directional_pct": 0.0,
                    "conviction_score": 0.0,
                    "bias": "—",
                }
            )
        else:
            rows.append(analyze_broker_row(bid, grp, total_act))

    df = pd.DataFrame(rows)
    # Also include any other high-activity brokers on symbol not in top market list
    other = sub[~sub["broker_id"].isin(top_ids)]
    if not other.empty:
        for bid, grp in other.groupby("broker_id"):
            act = grp["buy_qty"].sum() + grp["sell_qty"].sum() if "buy_qty" in grp.columns else 0
            if act >= total_act * 0.05:
                rows.append(analyze_broker_row(str(bid), grp, total_act))
        df = pd.DataFrame(rows).sort_values("conviction_score", ascending=False).head(top_n + 5)

    return df.sort_values("conviction_score", ascending=False)


def aggregate_top_broker_scores(sym: str, broker_panel: pd.DataFrame) -> dict:
    """Summary metrics from top-10 broker math for scanner merge."""
    table = symbol_top_brokers_table(sym, broker_panel)
    if table.empty:
        return {
            "top_broker_ids": "",
            "top_broker_net_lac": 0.0,
            "broker_top10_buy_pressure": 0.0,
            "broker_top10_conviction": 0.0,
        }

    active = table[table["activity_qty"] > 0]
    ids = ",".join(
        f"{r['broker_id']}({r['net_amount_lac']:.0f}|{r['conviction_score']:.0f})"
        for _, r in active.head(10).iterrows()
    )
    net_sum = float(active["net_amount_lac"].sum())
    buy_bias = float((active["bias"] == "buy").sum())
    conv_mean = float(active["conviction_score"].mean()) if not active.empty else 0.0

    return {
        "top_broker_ids": ids,
        "top_broker_net_lac": net_sum,
        "broker_top10_buy_pressure": round(buy_bias / max(len(active), 1) * 100, 1),
        "broker_top10_conviction": round(conv_mean, 1),
    }
