"""Build one symbol row with horizons + composite scores for deep dive / quant."""

from __future__ import annotations

import pandas as pd

from backend.config import load_yaml_config
from backend.features.engineer import _add_composite_features
from backend.ingest.panel_utils import snapshot_panel_all_horizons

HORIZON_POWER = load_yaml_config("horizons.yaml").get("power_scores", {})


def _latest_panel_slice(panel: pd.DataFrame, sym: str) -> pd.DataFrame:
    if panel.empty:
        return panel
    sp = panel[panel["symbol"].astype(str).str.upper() == sym].copy()
    if sp.empty:
        return sp
    if "report_date" in sp.columns:
        sp["report_date"] = pd.to_datetime(sp["report_date"]).dt.normalize()
        rd = sp["report_date"].max()
        on_day = sp[sp["report_date"] == rd]
        if not on_day.empty and (on_day["horizon"] != "unknown").any() if "horizon" in on_day.columns else True:
            return on_day
    return snapshot_panel_all_horizons(sp) if len(sp) > 50 else sp


def overlay_panel_horizons(row: pd.DataFrame, panel: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Fill acc_/dist_ horizon columns and LTP/zones from floorsheet panel."""
    out = row.copy()
    sp = _latest_panel_slice(panel, sym)
    if sp.empty:
        return out

    for _, r in sp.iterrows():
        side = str(r.get("side", "distribution")).lower()
        if side not in ("accumulation", "distribution"):
            continue
        h = str(r.get("horizon", "1D"))
        if h == "unknown":
            continue
        prefix = "acc" if side == "accumulation" else "dist"
        net = float(pd.to_numeric(r.get("net_amount_sum", 0), errors="coerce") or 0)
        ft = float(pd.to_numeric(r.get("net_float_turnover_mean", 0), errors="coerce") or 0)
        pwr = HORIZON_POWER.get(r.get("dominant_power"), 0) or 0
        out[f"{prefix}_{h}_net_amount"] = net
        out[f"{prefix}_{h}_float_turnover"] = ft
        out[f"{prefix}_{h}_power_score"] = pwr
        if side == "accumulation" and h == "1D":
            out[f"{prefix}_1D_power"] = r.get("dominant_power")
        if h == "1D":
            if pd.notna(r.get("ltp")):
                out["ltp"] = r.get("ltp")
            if pd.notna(r.get("tech_demand_zone")):
                out["tech_demand_zone"] = r.get("tech_demand_zone")
            if pd.notna(r.get("tech_supply_zone")):
                out["tech_supply_zone"] = r.get("tech_supply_zone")
            if pd.notna(r.get("broker_concentration")):
                out["broker_concentration"] = r.get("broker_concentration")
    return out


def recompute_composite_scores(row: pd.DataFrame) -> pd.DataFrame:
    """Derive floorsheet EMS, OFI, dist risk from horizon columns on this row."""
    if row.empty:
        return row
    base = row.copy()
    enriched = _add_composite_features(base)
    if enriched.empty:
        return row
    for col in enriched.columns:
        if col in ("symbol", "report_date"):
            continue
        row[col] = enriched[col].iloc[0]
    return row
