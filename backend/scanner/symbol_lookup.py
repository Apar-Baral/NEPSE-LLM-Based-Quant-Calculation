from __future__ import annotations

import pandas as pd

from backend.config import load_yaml_config
from backend.scanner.broker_desk import attach_broker_desk_metrics
from backend.scanner.broker_insights import attach_broker_metrics
from backend.scanner.llm_cache import apply_cached_llm_scores
from backend.scanner.volume_universe import (
    attach_ltp_from_panel,
    attach_volume_from_panel,
    compute_early_rank_score,
)
from backend.signals.effective_scores import effective_scores
from backend.signals.universe_tiers import assign_universe_tiers
from backend.utils.numeric import coerce_numeric


def all_tracked_symbols(predictions: pd.DataFrame, features: pd.DataFrame | None = None) -> list[str]:
    syms = set()
    if not predictions.empty:
        syms.update(predictions["symbol"].astype(str).str.upper().unique())
    if features is not None and not features.empty:
        syms.update(features["symbol"].astype(str).str.upper().unique())
    return sorted(syms)


def _merge_feature_row(row: pd.DataFrame, features: pd.DataFrame | None, sym: str) -> pd.DataFrame:
    if features is None or features.empty:
        return row
    feat = features[features["symbol"].astype(str).str.upper() == sym].sort_values("report_date")
    if feat.empty:
        return row
    fr = feat.iloc[-1]
    skip = {"symbol", "report_date", "as_of_date"}
    for col in fr.index:
        if col in skip:
            continue
        val = fr[col]
        if col not in row.columns or pd.isna(row[col].iloc[0]) or row[col].iloc[0] in (0, None, ""):
            row[col] = val
        elif col in (
            "early_momentum_score",
            "floorsheet_momentum_score",
            "distribution_risk_score",
            "smart_money_score",
            "ltp",
            "tech_demand_zone",
            "tech_supply_zone",
            "ofi",
            "mtf_convergence",
            "acc_dist_ratio",
        ):
            row[col] = val
    return row


def enrich_symbol_row(
    sym: str,
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    broker_panel: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full scanner + feature metrics for one symbol (even outside top 120)."""
    sym = str(sym).strip().upper()
    if predictions.empty:
        return pd.DataFrame()

    latest = predictions["report_date"].max()
    day = predictions[predictions["report_date"] == latest].copy()
    row = day[day["symbol"].astype(str).str.upper() == sym]
    if row.empty:
        row = predictions[predictions["symbol"].astype(str).str.upper() == sym].tail(1)
    if row.empty:
        return pd.DataFrame()

    row = row.tail(1).copy()
    row = _merge_feature_row(row, features, sym)
    row = attach_volume_from_panel(row, panel)
    row = attach_ltp_from_panel(row, panel, broker_panel)
    row = attach_broker_metrics(row, panel)
    if broker_panel is not None and not broker_panel.empty:
        row = attach_broker_desk_metrics(row, broker_panel)

    try:
        row = apply_cached_llm_scores(row)
    except Exception:
        pass

    row["early_rank_score"] = compute_early_rank_score(row)
    row = coerce_numeric(row)
    cfg = load_yaml_config("settings.yaml")["signals"]
    p_eff, ems_eff, _ = effective_scores(row.iloc[0], cfg)
    row["p_long_momentum"] = p_eff
    row["early_momentum_score"] = ems_eff
    row["signal_tier"] = assign_universe_tiers(row)
    return row


def filter_universe_by_symbol(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = (query or "").strip().upper()
    if not q or df.empty:
        return df
    mask = df["symbol"].astype(str).str.upper().str.contains(q, na=False)
    return df[mask]
