from __future__ import annotations

import pandas as pd

from backend.config import load_yaml_config

TIER_ORDER = ("Invalidated", "Neutral", "Watch", "Setup", "Trigger", "Confirmed")


def _distribution_mode(row: pd.Series, cfg: dict) -> bool:
    if not cfg.get("distribution_mode", True):
        return False
    acc_power = float(row.get("acc_1D_power_score", 0) or 0)
    acc_amt = float(row.get("acc_1D_net_amount", 0) or 0)
    return acc_power < 1 and abs(acc_amt) < 1


def _effective_scores(row: pd.Series, cfg: dict) -> tuple[float, float, float]:
    p = float(row.get("p_long_momentum", 0) or 0)
    ems = float(row.get("early_momentum_score", 0) or 0)
    rank = float(row.get("early_rank_score", 0) or 0)
    broker_p = float(row.get("broker_pressure", 0) or 0)
    floorsheet = float(row.get("floorsheet_momentum_score", 0) or 0)

    if _distribution_mode(row, cfg):
        p = max(p, rank * 0.55, broker_p / 200)
        ems = max(ems, floorsheet, broker_p * 0.45, rank * 100 * 0.35)
    elif rank > 0:
        p = max(p, rank * 0.25)
        ems = max(ems, rank * 100 * 0.2)

    return p, ems, broker_p


def assign_signal_tier(row: pd.Series) -> str:
    cfg = load_yaml_config("settings.yaml")["signals"]
    p, ems, broker_p = _effective_scores(row, cfg)
    drs = float(row.get("distribution_risk_score", 100) or 100)
    z = float(row.get("float_turnover_zscore", 0) or row.get("float_turnover_zscore_hv", 0) or 0)
    rank = float(row.get("early_rank_score", 0) or 0)
    shakeout = bool(row.get("dist_shakeout_flag", False) or row.get("pattern_dist_shakeout", False))

    dist_mode = _distribution_mode(row, cfg)
    trig_p = cfg["dist_trigger_probability"] if dist_mode else cfg["trigger_probability"]
    conf_p = cfg["dist_confirmed_probability"] if dist_mode else cfg["confirmed_probability"]
    ems_thr = cfg["dist_early_momentum_score"] if dist_mode else cfg["early_momentum_score"]
    inv_drs = cfg.get("dist_invalidate_drs", 72)

    long_heavy = float(row.get("dist_3Y_power_score", 0) or 0) >= 2 and float(row.get("dist_1D_power_score", 3) or 3) >= 2
    if drs >= inv_drs and p < 0.40 and not shakeout and long_heavy:
        return "Invalidated"
    if p >= conf_p and ems >= ems_thr * 1.1:
        return "Confirmed"
    if p >= trig_p and ems >= ems_thr * 0.85:
        return "Trigger"
    if dist_mode and broker_p >= cfg.get("dist_broker_pressure_trigger", 55) and rank >= 0.12:
        return "Trigger"
    if shakeout and rank >= 0.08 and drs < inv_drs:
        return "Setup"
    if row.get("mtf_convergence", 0) >= 0.5 and z >= cfg["watch_zscore"]:
        return "Setup"
    if dist_mode and (rank >= 0.15 or broker_p >= 45 or z >= 1.0):
        return "Setup"
    if z >= cfg["watch_zscore"] or row.get("acc_1D_power_score", 0) >= 2:
        return "Watch"
    if dist_mode and (rank >= 0.06 or broker_p >= 35):
        return "Watch"
    return "Neutral"


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return df[name].fillna(default)
    return pd.Series(default, index=df.index)


PREDICTION_ONLY_COLS = {
    "p_long_momentum",
    "expected_return_10d",
    "confidence",
    "anomaly_score",
    "anomaly_flag",
    "analog_count",
    "analog_hit_rate",
    "analog_mover_count",
}


def apply_momentum_rules(features: pd.DataFrame, predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    df = features.copy()
    if predictions is not None and not predictions.empty:
        merge_cols = ["report_date", "symbol"]
        pred_cols = [c for c in predictions.columns if c in PREDICTION_ONLY_COLS]
        if pred_cols:
            df = df.drop(columns=[c for c in pred_cols if c in df.columns], errors="ignore")
            df = df.merge(predictions[merge_cols + pred_cols], on=merge_cols, how="left")

    df["signal_tier"] = df.apply(assign_signal_tier, axis=1)

    df["pattern_horizon_ladder"] = (
        (_col(df, "acc_1D_power_score") >= 2)
        & (_col(df, "acc_2D_power_score") >= 2)
        & (_col(df, "acc_1M_power_score") <= 2)
    )
    df["pattern_dist_shakeout"] = (
        (_col(df, "dist_1D_power_score") <= 1)
        & (_col(df, "dist_1W_power_score") >= 2)
        & (_col(df, "demand_zone_distance_pct", 999).between(-2, 5))
    ) | _col(df, "dist_shakeout_flag", 0).astype(bool)
    df["pattern_float_spike"] = _col(df, "float_turnover_zscore") >= 2
    df["pattern_zone_power"] = (
        (_col(df, "acc_1W_power_score") >= 3)
        & (_col(df, "demand_zone_distance_pct", 999).between(-3, 3))
    )

    return df
