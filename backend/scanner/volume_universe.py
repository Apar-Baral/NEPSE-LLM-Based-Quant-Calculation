from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import load_yaml_config
from backend.scanner.broker_insights import attach_broker_metrics
from backend.ingest.panel_utils import snapshot_panel_all_horizons

VOLUME_COLS = ("daily_volume", "daily_turnover_lac", "float_turnover_1d_abs")
SHORT_HORIZONS = ("1D", "2D", "3D", "4D", "1W")


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


def _short_horizon_rows(grp: pd.DataFrame, side: str | None = None) -> pd.DataFrame:
    sub = grp.copy()
    if side and "side" in sub.columns:
        sub = sub[sub["side"] == side]
    if "horizon" in sub.columns:
        sub = sub[sub["horizon"].isin(SHORT_HORIZONS)]
    return sub


def attach_volume_from_panel(features: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """1D-only volume and turnover from distribution panel (trader-relevant)."""
    out = features.copy()
    out = out.drop(columns=[c for c in VOLUME_COLS if c in out.columns], errors="ignore")

    if panel.empty or "buy_qty_sum" not in panel.columns:
        return _ensure_volume_cols(out)

    rows = []
    for sym, grp in panel.groupby("symbol"):
        dist = _short_horizon_rows(grp, "distribution")
        d1 = dist[dist["horizon"] == "1D"] if "horizon" in dist.columns else dist.head(1)
        if d1.empty and not dist.empty:
            d1 = dist.head(1)
        if d1.empty:
            d1 = grp[grp["horizon"] == "1D"] if "horizon" in grp.columns else grp.head(1)

        buy = pd.to_numeric(d1["buy_qty_sum"], errors="coerce").fillna(0).sum() if not d1.empty else 0.0
        sell = pd.to_numeric(d1["sell_qty_sum"], errors="coerce").fillna(0).sum() if not d1.empty else 0.0
        net_amt = (
            pd.to_numeric(d1["net_amount_sum"], errors="coerce").fillna(0).abs().sum()
            if not d1.empty and "net_amount_sum" in d1.columns
            else 0.0
        )
        ft = (
            pd.to_numeric(d1["net_float_turnover_mean"], errors="coerce").fillna(0).abs().mean()
            if not d1.empty and "net_float_turnover_mean" in d1.columns
            else 0.0
        )

        rows.append(
            {
                "symbol": sym,
                "daily_volume": float(buy + sell),
                "daily_turnover_lac": float(net_amt),
                "float_turnover_1d_abs": float(ft),
            }
        )

    if not rows:
        return _ensure_volume_cols(out)

    vol_df = pd.DataFrame(rows)
    out = out.merge(vol_df, on="symbol", how="left")
    return _ensure_volume_cols(out)


def compute_early_rank_score(df: pd.DataFrame) -> pd.Series:
    df = _ensure_volume_cols(df)
    llm_p = df.get("llm_p_long", df.get("p_long_momentum", pd.Series(0, index=df.index))).fillna(0)
    p = df.get("p_long_momentum", pd.Series(0, index=df.index)).fillna(0)
    p = np.maximum(p, llm_p)
    ems = df.get("early_momentum_score", pd.Series(0, index=df.index)).fillna(0) / 100
    broker = df.get("broker_pressure", pd.Series(0, index=df.index)).fillna(0) / 100
    drs = df.get("distribution_risk_score", pd.Series(0, index=df.index)).fillna(0) / 100
    shakeout = df.get("dist_shakeout_flag", pd.Series(False, index=df.index)).fillna(False).astype(float)

    turn = df["daily_turnover_lac"].fillna(0)
    turn_n = turn / (turn.max() + 1e-9) if turn.max() > 0 else turn

    return (
        p * 0.25
        + ems * 0.2
        + broker * 0.2
        + turn_n * 0.15
        + shakeout * 0.1
        - drs * 0.1
    ).clip(0, 1)


def select_high_volume_universe(
    df: pd.DataFrame,
    top_n: int | None = None,
    min_volume: float = 0,
) -> pd.DataFrame:
    cfg = _cfg()
    top_n = top_n or cfg.get("high_volume_top_n", 120)
    min_volume = min_volume or cfg.get("min_daily_volume", 0)
    out = _ensure_volume_cols(df.copy())

    out = out[out["daily_turnover_lac"] >= min_volume]
    if out.empty:
        return out

    out = out.sort_values("daily_turnover_lac", ascending=False).head(top_n).copy()
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
        from backend.scanner.llm_scorer import score_universe_with_llm
        from backend.signals import assign_universe_tiers

        out = score_universe_with_llm(out, p, fetch_new=False)
        out["early_rank_score"] = compute_early_rank_score(out)
        out = out.sort_values("early_rank_score", ascending=False)
        out["volume_rank"] = range(1, len(out) + 1)
        out["signal_tier"] = assign_universe_tiers(out)
        if "llm_p_long" in out.columns:
            out["p_long_momentum"] = out["llm_p_long"].fillna(out["p_long_momentum"])
    return out


def symbol_horizon_snapshot(sym_panel: pd.DataFrame, side: str) -> pd.DataFrame:
    """One row per short horizon for a symbol."""
    if sym_panel.empty:
        return sym_panel
    sub = sym_panel[sym_panel["side"] == side].copy() if "side" in sym_panel.columns else sym_panel.copy()
    if sub.empty:
        return sub
    if "horizon" in sub.columns:
        sub = sub[sub["horizon"].isin(SHORT_HORIZONS)]
    if sub.empty or "horizon" not in sub.columns:
        return sub
    sub["abs_net"] = pd.to_numeric(sub.get("net_amount_sum", 0), errors="coerce").fillna(0).abs()
    sub = sub.sort_values("abs_net", ascending=False).drop_duplicates(subset=["horizon"], keep="first")
    order = load_yaml_config("horizons.yaml")["horizons"]
    order_map = {h["key"]: h["order"] for h in order if h["key"] in SHORT_HORIZONS}
    sub["horizon_order"] = sub["horizon"].map(order_map).fillna(99)
    return sub.sort_values("horizon_order")
