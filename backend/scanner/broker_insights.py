from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import load_yaml_config

HORIZON_ORDER = {h["key"]: h["order"] for h in load_yaml_config("horizons.yaml")["horizons"]}
SHORT_HORIZONS = ("1D", "2D", "3D", "4D", "1W")


def _horizon_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "horizon" not in df.columns:
        return df
    out = df.copy()
    out["_ord"] = out["horizon"].map(HORIZON_ORDER).fillna(99)
    return out.sort_values("_ord").drop(columns=["_ord"], errors="ignore")


def broker_pressure_score(sym_panel: pd.DataFrame) -> float:
    """0–100: short-horizon buy vs sell qty + heavy-broker skew (distribution panel)."""
    if sym_panel.empty:
        return 0.0
    sub = sym_panel[sym_panel["horizon"].isin(SHORT_HORIZONS)] if "horizon" in sym_panel.columns else sym_panel
    if sub.empty:
        sub = sym_panel
    buy = pd.to_numeric(sub.get("buy_qty_sum", 0), errors="coerce").fillna(0).sum()
    sell = pd.to_numeric(sub.get("sell_qty_sum", 0), errors="coerce").fillna(0).sum()
    total = buy + sell
    if total <= 0:
        return 0.0
    ofi = (buy - sell) / total
    heavy = float(pd.to_numeric(sub.get("heavy_broker_count", 0), errors="coerce").fillna(0).sum())
    brokers = float(pd.to_numeric(sub.get("broker_count", 0), errors="coerce").fillna(0).sum())
    heavy_skew = min(heavy / max(brokers, 1), 1.0)
    raw = (ofi + 1) / 2 * 70 + heavy_skew * 30
    return float(np.clip(raw, 0, 100))


def horizon_net_flow(sym_panel: pd.DataFrame, side: str) -> pd.DataFrame:
    """Per-horizon net amount (Lac) for acc or dist — comparable across horizons."""
    if sym_panel.empty:
        return pd.DataFrame()
    sub = sym_panel[sym_panel["side"] == side].copy() if "side" in sym_panel.columns else sym_panel
    if sub.empty:
        return sub
    sub = _horizon_sort(sub)
    sub["net_lac"] = pd.to_numeric(sub.get("net_amount_sum", 0), errors="coerce").fillna(0) / 100_000
    sub["power"] = sub.get("dominant_power", "—")
    return sub[["horizon", "net_lac", "power", "buy_qty_sum", "sell_qty_sum"]].copy()


def attach_broker_metrics(df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop(
        columns=[c for c in ("broker_pressure", "short_ofi", "heavy_broker_share", "dist_shakeout_flag") if c in out.columns],
        errors="ignore",
    )
    if out.empty or panel.empty:
        for col in ("broker_pressure", "short_ofi", "heavy_broker_share", "dist_shakeout_flag"):
            if col not in out.columns:
                out[col] = 0.0 if col != "dist_shakeout_flag" else False
        return out

    rows = []
    for sym, grp in panel.groupby("symbol"):
        dist = grp[grp["side"] == "distribution"] if "side" in grp.columns else grp
        d1 = dist[dist["horizon"] == "1D"] if "horizon" in dist.columns and not dist.empty else dist
        d1w = dist[dist["horizon"] == "1W"] if "horizon" in dist.columns and not dist.empty else pd.DataFrame()
        p1 = float(d1["dominant_power_score"].iloc[0]) if not d1.empty and "dominant_power_score" in d1.columns else 3
        p1w = float(d1w["dominant_power_score"].iloc[0]) if not d1w.empty and "dominant_power_score" in d1w.columns else 3
        shakeout = p1 <= 1 and p1w >= 2
        buy = pd.to_numeric(d1["buy_qty_sum"], errors="coerce").fillna(0).sum() if not d1.empty else 0
        sell = pd.to_numeric(d1["sell_qty_sum"], errors="coerce").fillna(0).sum() if not d1.empty else 0
        total = buy + sell
        ofi = (buy - sell) / total if total > 0 else 0.0
        heavy = float(pd.to_numeric(dist.get("heavy_broker_count", 0), errors="coerce").fillna(0).sum())
        brokers = float(pd.to_numeric(dist.get("broker_count", 0), errors="coerce").fillna(0).sum())
        rows.append(
            {
                "symbol": sym,
                "broker_pressure": broker_pressure_score(dist),
                "short_ofi": ofi,
                "heavy_broker_share": min(heavy / max(brokers, 1), 1.0),
                "dist_shakeout_flag": shakeout,
            }
        )
    metrics = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["symbol", "broker_pressure", "short_ofi", "heavy_broker_share", "dist_shakeout_flag"])
    out = out.merge(metrics, on="symbol", how="left")
    for col, default in (
        ("broker_pressure", 0.0),
        ("short_ofi", 0.0),
        ("heavy_broker_share", 0.0),
        ("dist_shakeout_flag", False),
    ):
        if col not in out.columns:
            out[col] = default
        if col == "dist_shakeout_flag":
            out[col] = out[col].fillna(False)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out
