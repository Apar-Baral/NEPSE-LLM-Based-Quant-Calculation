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


def day_frame_from_broker_panel(
    broker_panel: pd.DataFrame,
    report_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per symbol with 1D volume/turnover when panel/features are sparse."""
    if broker_panel.empty:
        return pd.DataFrame()

    bp = broker_panel.copy()
    if "report_date" in bp.columns:
        bp["report_date"] = pd.to_datetime(bp["report_date"]).dt.normalize()
        if report_date is not None:
            rd = pd.Timestamp(report_date).normalize()
            on_day = bp[bp["report_date"] == rd]
            if not on_day.empty:
                bp = on_day
    if "horizon" in bp.columns:
        h1 = bp[bp["horizon"] == "1D"]
        if not h1.empty:
            bp = h1

    rows = []
    for sym, grp in bp.groupby("symbol"):
        buy = pd.to_numeric(grp.get("buy_qty", 0), errors="coerce").fillna(0).sum()
        sell = pd.to_numeric(grp.get("sell_qty", 0), errors="coerce").fillna(0).sum()
        net_amt = pd.to_numeric(grp.get("net_amount", 0), errors="coerce").fillna(0).abs().sum()
        ltp_vals = pd.to_numeric(grp.get("ltp"), errors="coerce").dropna()
        rows.append(
            {
                "symbol": str(sym).upper(),
                "report_date": grp["report_date"].max() if "report_date" in grp.columns else report_date,
                "daily_volume": float(buy + sell),
                "daily_turnover_lac": float(net_amt),
                "float_turnover_1d_abs": 0.0,
                "ltp": float(ltp_vals.iloc[-1]) if len(ltp_vals) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _expand_day_from_broker_panel(
    day: pd.DataFrame,
    broker_panel: pd.DataFrame,
    report_date: pd.Timestamp | None,
    min_symbols: int,
) -> pd.DataFrame:
    if broker_panel.empty or len(day) >= min_symbols:
        return day

    base = day_frame_from_broker_panel(broker_panel, report_date)
    if base.empty:
        return day

    if report_date is not None:
        base["report_date"] = pd.Timestamp(report_date).normalize()

    if day.empty:
        return base

    day_u = day.copy()
    day_u["symbol"] = day_u["symbol"].astype(str).str.upper()
    overlap = [c for c in day_u.columns if c in base.columns and c != "symbol"]
    feat = day_u.drop(columns=overlap, errors="ignore")
    return base.merge(feat, on="symbol", how="left")


def attach_volume_from_panel(features: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """1D-only volume and turnover from distribution panel (trader-relevant)."""
    out = features.copy()
    if panel.empty or "buy_qty_sum" not in panel.columns:
        return _ensure_volume_cols(out)

    panel_syms = panel["symbol"].nunique() if "symbol" in panel.columns else 0
    has_vol = any(
        c in out.columns and pd.to_numeric(out[c], errors="coerce").fillna(0).sum() > 0 for c in VOLUME_COLS
    )
    if has_vol and panel_syms < max(10, int(len(out) * 0.15)):
        return _ensure_volume_cols(out)

    out = out.drop(columns=[c for c in VOLUME_COLS if c in out.columns], errors="ignore")

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


def attach_ltp_from_panel(
    df: pd.DataFrame,
    panel: pd.DataFrame,
    broker_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["ltp"] = pd.to_numeric(out.get("ltp"), errors="coerce")
    fills: dict[str, float] = {}

    sources = [panel]
    if broker_panel is not None and not broker_panel.empty:
        sources.append(broker_panel)
    for source in sources:
        if source.empty or "ltp" not in source.columns:
            continue
        for sym, grp in source.groupby("symbol"):
            if sym in fills:
                continue
            vals = pd.to_numeric(grp["ltp"], errors="coerce").dropna()
            if len(vals):
                fills[str(sym)] = float(vals.iloc[-1])

    if fills:
        fill_df = pd.DataFrame({"symbol": list(fills.keys()), "ltp_fill": list(fills.values())})
        out = out.merge(fill_df, on="symbol", how="left")
        out["ltp"] = out["ltp"].fillna(pd.to_numeric(out["ltp_fill"], errors="coerce"))
        out.drop(columns=["ltp_fill"], inplace=True, errors="ignore")
    return out


def _attach_broker_metrics_from_broker_panel(df: pd.DataFrame, broker_panel: pd.DataFrame) -> pd.DataFrame:
    """Broker pressure when symbol_panel is empty but broker_panel has desks."""
    from backend.scanner.broker_flow import broker_flow_metrics

    out = df.copy()
    if broker_panel.empty:
        return attach_broker_metrics(out, pd.DataFrame())

    bp = broker_panel.copy()
    if "horizon" in bp.columns:
        h1 = bp[bp["horizon"] == "1D"]
        if not h1.empty:
            bp = h1

    rows = []
    for sym, grp in bp.groupby("symbol"):
        m = broker_flow_metrics(grp)
        total = m["buy_qty"] + m["sell_qty"]
        ofi = (m["buy_qty"] - m["sell_qty"]) / total if total > 0 else 0.0
        pressure = float(np.clip((ofi + 1) / 2 * 70 + min(m["buy_share_pct"] / 100, 1) * 30, 0, 100))
        rows.append(
            {
                "symbol": str(sym).upper(),
                "broker_pressure": pressure,
                "short_ofi": ofi,
                "heavy_broker_share": 0.0,
                "dist_shakeout_flag": False,
            }
        )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return attach_broker_metrics(out, pd.DataFrame())
    return out.merge(metrics, on="symbol", how="left")


def get_latest_scanner_universe(
    predictions: pd.DataFrame,
    panel: pd.DataFrame | None = None,
    broker_panel: pd.DataFrame | None = None,
    top_n: int | None = None,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build scanner universe from features/predictions; fall back to broker_panel when sparse."""
    cfg = _cfg()
    top_n = top_n or cfg.get("high_volume_top_n", 120)
    latest: pd.Timestamp | None = None

    if features is not None and not features.empty:
        latest = features["report_date"].max()
        day = features[features["report_date"] == latest].copy()
        if not predictions.empty:
            pred = predictions[predictions["report_date"] == latest]
            pred_only = [
                c
                for c in (
                    "p_long_momentum",
                    "expected_return_10d",
                    "confidence",
                    "anomaly_score",
                    "anomaly_flag",
                    "signal_tier",
                    "early_momentum_score",
                    "distribution_risk_score",
                    "smart_money_score",
                )
                if c in pred.columns
            ]
            merge_cols = ["report_date", "symbol"] + pred_only
            day = day.drop(columns=[c for c in pred_only if c in day.columns], errors="ignore")
            day = day.merge(pred[merge_cols], on=["report_date", "symbol"], how="left")
    elif not predictions.empty:
        latest = predictions["report_date"].max()
        day = predictions[predictions["report_date"] == latest].copy()
    else:
        return pd.DataFrame()

    if latest is None and not day.empty:
        latest = day["report_date"].max()

    min_symbols = min(top_n, max(10, top_n // 4))
    if broker_panel is not None and not broker_panel.empty:
        day = _expand_day_from_broker_panel(day, broker_panel, latest, min_symbols)

    p = snapshot_panel_all_horizons(panel if panel is not None else pd.DataFrame())
    day = attach_volume_from_panel(day, p)
    if p.empty and broker_panel is not None and not broker_panel.empty:
        vol_bp = day_frame_from_broker_panel(broker_panel, latest)
        if not vol_bp.empty:
            vol_cols = ["symbol", "daily_volume", "daily_turnover_lac", "float_turnover_1d_abs"]
            day = day.drop(columns=[c for c in vol_cols if c in day.columns and c != "symbol"], errors="ignore")
            day = day.merge(vol_bp[vol_cols], on="symbol", how="left")
            day = _ensure_volume_cols(day)
    day = attach_ltp_from_panel(day, p, broker_panel)
    panel_syms = p["symbol"].nunique() if not p.empty and "symbol" in p.columns else 0
    if (p.empty or panel_syms < min_symbols) and broker_panel is not None and not broker_panel.empty:
        day = _attach_broker_metrics_from_broker_panel(day, broker_panel)
    else:
        day = attach_broker_metrics(day, p)
    out = select_high_volume_universe(day, top_n=top_n)
    if not out.empty:
        from backend.scanner.broker_desk import attach_broker_desk_metrics
        from backend.scanner.llm_cache import apply_cached_llm_scores
        from backend.signals.universe_tiers import assign_universe_tiers
        from backend.utils.numeric import coerce_numeric

        if broker_panel is not None and not broker_panel.empty:
            out = attach_broker_desk_metrics(out, broker_panel)

        try:
            out = apply_cached_llm_scores(out)
        except Exception:
            pass
        out["early_rank_score"] = compute_early_rank_score(out)
        out = coerce_numeric(out.sort_values("early_rank_score", ascending=False))
        out["early_pick_rank"] = range(1, len(out) + 1)
        out["turnover_rank"] = out["volume_rank"]
        out["signal_tier"] = assign_universe_tiers(out)
        if "llm_p_long" in out.columns:
            out["p_long_momentum"] = pd.to_numeric(out["llm_p_long"], errors="coerce").fillna(
                pd.to_numeric(out["p_long_momentum"], errors="coerce")
            )
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
