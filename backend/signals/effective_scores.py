"""Distribution-aware effective P(long) and EMS — shared by tiers and quant pipeline."""

from __future__ import annotations

import pandas as pd

from backend.config import load_yaml_config


def distribution_mode(row: pd.Series, cfg: dict | None = None) -> bool:
    cfg = cfg or load_yaml_config("settings.yaml")["signals"]
    if not cfg.get("distribution_mode", True):
        return False
    acc_power = float(row.get("acc_1D_power_score", 0) or 0)
    acc_amt = float(row.get("acc_1D_net_amount", 0) or 0)
    return acc_power < 1 and abs(acc_amt) < 1


def effective_scores(row: pd.Series, cfg: dict | None = None) -> tuple[float, float, float]:
    """
    Return (p_long_effective, ems_effective, broker_pressure).
    Adjusts raw model scores when only distribution floorsheet data exists.
    """
    cfg = cfg or load_yaml_config("settings.yaml")["signals"]
    p = float(row.get("p_long_momentum", 0) or 0)
    ems = float(row.get("early_momentum_score", 0) or 0)
    rank = float(row.get("early_rank_score", 0) or 0)
    broker_p = float(row.get("broker_pressure", 0) or 0)
    floorsheet = float(row.get("floorsheet_momentum_score", 0) or 0)

    if distribution_mode(row, cfg):
        turn = float(row.get("daily_turnover_lac") or 0)
        turn_boost = min(0.22, turn / 1200) if turn > 0 else 0
        p = max(p, rank * 0.72, broker_p / 200, 0.28 + turn_boost)
        ems = max(ems, floorsheet, broker_p * 0.45, rank * 100 * 0.4, 12 + turn_boost * 80)
    elif rank > 0:
        p = max(p, rank * 0.25)
        ems = max(ems, rank * 100 * 0.2)

    return p, ems, broker_p


# Backward-compatible alias
_effective_scores = effective_scores
