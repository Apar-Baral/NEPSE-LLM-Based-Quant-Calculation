from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import load_yaml_config
from backend.scanner.broker_insights import attach_broker_metrics
from backend.ingest.panel_utils import snapshot_panel_all_horizons
from backend.signals.momentum_rules import assign_signal_tier

VOLUME_COLS = ("daily_volume", "daily_turnover_lac", "float_turnover_1d_abs")


def _cfg() -> dict:
    return load_yaml_config("settings.yaml").get("scanner", {})


def _ensure_volume_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in VOLUME_COLS:
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def _safe_sum(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0).sum())


def _safe_max(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0).abs().max())


def attach_volume_from_panel(features: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Add volume / turnover per symbol from floorsheet panel."""
    out = features.copy()
    out = out.drop(columns=[c for c in VOLUME_COLS if c in out.columns], errors="ignore")

    if panel.empty or "buy_qty_sum" not in panel.columns:
        return _ensure_volume_cols(out)

    panel = panel.copy()
    rows = []
    for sym, grp in panel.groupby("symbol"):
        buy = pd.to_numeric(grp["buy_qty_sum"], errors="coerce").fillna(0)
        sell = (
            pd.to_numeric(grp["sell_qty_sum"], errors="coerce").fillna(0)
            if "sell_qty_sum" in grp.columns
            else pd.Series(0.0, index=grp.index)
        )
        turnover = (
            pd.to_numeric(grp["net_amount_sum"], errors="coerce").fillna(0).abs()
            if "net_amount_sum" in grp.columns
            else pd.Series(0.0, index=grp.index)
        )
        ft = (
            pd.to_numeric(grp["net_float_turnover_mean"], errors="coerce").fillna(0).abs()
            if "net_float_turnover_mean" in grp.columns
            else pd.Series(0.0, index=grp.index)
        )

        d1 = grp[grp["horizon"] == "1D"] if "horizon" in grp.columns else grp
        if d1.empty:
            d1 = grp

        d1_buy = pd.to_numeric(d1["buy_qty_sum"], errors="coerce").fillna(0) if "buy_qty_sum" in d1.columns else pd.Series(0.0)
        d1_sell = (
            pd.to_numeric(d1["sell_qty_sum"], errors="coerce").fillna(0)
            if "sell_qty_sum" in d1.columns
            else pd.Series(0.0)
        )

        daily_vol = _safe_sum(d1_buy) + _safe_sum(d1_sell)
        peak_vol = float(buy.max() + sell.max()) if len(buy) else daily_vol

        rows.append(
            {
                "symbol": sym,
                "daily_volume": max(daily_vol, peak_vol),
                "daily_turnover_lac": _safe_max(turnover),
                "float_turnover_1d_abs": _safe_max(
                    pd.to_numeric(d1["net_float_turnover_mean"], errors="coerce").fillna(0).abs()
                    if "net_float_turnover_mean" in d1.columns
                    else ft
                ),
            }
        )

    if not rows:
        return _ensure_volume_cols(out)

    vol_df = pd.DataFrame(rows)
    out = out.merge(vol_df, on="symbol", how="left")
    return _ensure_volume_cols(out)


def compute_early_rank_score(df: pd.DataFrame) -> pd.Series:
    df = _ensure_volume_cols(df)
    p = df.get("p_long_momentum", pd.Series(0, index=df.index)).fillna(0)
    ems = df.get("early_momentum_score", pd.Series(0, index=df.index)).fillna(0) / 100
    sms = df.get("smart_money_score", pd.Series(0, index=df.index)).fillna(0) / 100
    drs = df.get("distribution_risk_score", pd.Series(0, index=df.index)).fillna(0) / 100
    z = df.get("float_turnover_zscore", pd.Series(0, index=df.index)).fillna(0).clip(0, 3) / 3
    analog = df.get("analog_hit_rate", pd.Series(0, index=df.index)).fillna(0)

    dist_only_boost = pd.Series(0.0, index=df.index)
    if sms.sum() == 0:
        short_act = df["float_turnover_1d_abs"].fillna(0)
        if short_act.max() > 0:
            short_act = short_act / short_act.max()
        long_dist = df.get("dist_3M_power_score", pd.Series(0, index=df.index)).fillna(0)
        dist_only_boost = short_act * (1 - long_dist / 3) * 0.2

    vol_boost = pd.Series(0.0, index=df.index)
    if df["daily_volume"].max() > 0:
        vol_boost = df["daily_volume"] / (df["daily_volume"].max() + 1e-9) * 0.1

    return (p * 0.3 + ems * 0.2 + sms * 0.15 + z * 0.1 + analog * 0.1 + dist_only_boost + vol_boost - drs * 0.15).clip(0, 1)


def _pick_volume_column(df: pd.DataFrame) -> str:
    df = _ensure_volume_cols(df)
    for col in ("daily_volume", "daily_turnover_lac", "float_turnover_1d_abs"):
        if df[col].sum() > 0:
            return col
    return "daily_volume"


def select_high_volume_universe(
    df: pd.DataFrame,
    top_n: int | None = None,
    min_volume: float = 0,
) -> pd.DataFrame:
    cfg = _cfg()
    top_n = top_n or cfg.get("high_volume_top_n", 120)
    min_volume = min_volume or cfg.get("min_daily_volume", 0)
    out = _ensure_volume_cols(df.copy())

    vol_col = _pick_volume_column(out)
    out = out[out[vol_col] >= min_volume]
    if out.empty:
        return out

    out = out.sort_values(vol_col, ascending=False).head(top_n).copy()
    out["volume_rank"] = range(1, len(out) + 1)
    out["early_rank_score"] = compute_early_rank_score(out)

    if "acc_1D_float_turnover" in out.columns:
        ft = pd.to_numeric(out["acc_1D_float_turnover"], errors="coerce").fillna(out["float_turnover_1d_abs"])
        std = ft.std()
        out["float_turnover_zscore_hv"] = (ft - ft.mean()) / (std + 1e-9) if std > 0 else 0.0

    return out.sort_values("early_rank_score", ascending=False)


def get_latest_scanner_universe(
    predictions: pd.DataFrame,
    panel: pd.DataFrame | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions

    latest = predictions["report_date"].max()
    day = predictions[predictions["report_date"] == latest].copy()
    p = snapshot_panel_all_horizons(panel if panel is not None else pd.DataFrame())
    day = attach_volume_from_panel(day, p)
    day = attach_broker_metrics(day, p)
    out = select_high_volume_universe(day, top_n=top_n)
    if not out.empty:
        out["signal_tier"] = out.apply(assign_signal_tier, axis=1)
    return out


def symbol_horizon_snapshot(sym_panel: pd.DataFrame, side: str) -> pd.DataFrame:
    """One row per horizon for a symbol (best for multi-horizon charts)."""
    if sym_panel.empty:
        return sym_panel
    sub = sym_panel[sym_panel["side"] == side].copy()
    if sub.empty:
        return sub
    if "horizon" not in sub.columns:
        return sub
    sub["abs_net"] = pd.to_numeric(sub.get("net_amount_sum", 0), errors="coerce").fillna(0).abs()
    sub = sub.sort_values("abs_net", ascending=False).drop_duplicates(subset=["horizon"], keep="first")
    order = load_yaml_config("horizons.yaml")["horizons"]
    order_map = {h["key"]: h["order"] for h in order}
    sub["horizon_order"] = sub["horizon"].map(order_map).fillna(99)
    return sub.sort_values("horizon_order")
