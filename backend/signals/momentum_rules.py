from __future__ import annotations

import pandas as pd

from backend.config_signals import get_signal_config
from backend.signals.effective_scores import distribution_mode, effective_scores

TIER_ORDER = ("Invalidated", "Neutral", "Watch", "Setup", "Trigger", "Confirmed")

_distribution_mode = distribution_mode
_effective_scores = effective_scores


def assign_signal_tier(row: pd.Series) -> str:
    cfg = get_signal_config()
    p, ems, broker_p = effective_scores(row, cfg)
    drs = float(row.get("distribution_risk_score", 100) or 100)
    z = float(row.get("float_turnover_zscore", 0) or row.get("float_turnover_zscore_hv", 0) or 0)
    rank = float(row.get("early_rank_score", 0) or 0)
    shakeout = bool(row.get("dist_shakeout_flag", False) or row.get("pattern_dist_shakeout", False))

    dist_mode = distribution_mode(row, cfg)
    trig_p = cfg.get("dist_trigger_probability", 0.30) if dist_mode else cfg.get("trigger_probability", 0.60)
    conf_p = cfg.get("dist_confirmed_probability", 0.40) if dist_mode else cfg.get("confirmed_probability", 0.65)
    ems_thr = cfg.get("dist_early_momentum_score", 20) if dist_mode else cfg.get("early_momentum_score", 70)
    inv_drs = cfg.get("dist_invalidate_drs", 85)
    watch_z = cfg.get("watch_zscore", 1.5)

    long_heavy = float(row.get("dist_3Y_power_score", 0) or 0) >= 2 and float(row.get("dist_1D_power_score", 3) or 3) >= 2
    if drs >= inv_drs and p < 0.40 and not shakeout and long_heavy:
        return "Invalidated"
    if p >= conf_p and ems >= ems_thr * 1.1:
        return "Confirmed"
    if p >= trig_p and ems >= ems_thr * 0.85:
        return "Trigger"
    if dist_mode and broker_p >= cfg.get("dist_broker_pressure_trigger", 18) and rank >= 0.12:
        return "Trigger"
    if shakeout and rank >= 0.08 and drs < inv_drs:
        return "Setup"
    if row.get("mtf_convergence", 0) >= 0.5 and z >= watch_z:
        return "Setup"
    if dist_mode and (rank >= 0.15 or broker_p >= 45 or z >= 1.0):
        return "Setup"
    if z >= watch_z or row.get("acc_1D_power_score", 0) >= 2:
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
