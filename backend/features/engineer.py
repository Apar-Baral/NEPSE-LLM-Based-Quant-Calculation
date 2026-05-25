from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import load_yaml_config

HORIZON_ORDER = {h["key"]: h["order"] for h in load_yaml_config("horizons.yaml")["horizons"]}
POWER_SCORES = load_yaml_config("horizons.yaml")["power_scores"]


def pivot_side(panel: pd.DataFrame, side: str) -> pd.DataFrame:
    sub = panel[panel["side"] == side].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values("report_date")
    return sub


def build_daily_feature_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Build symbol-day feature matrix from long-format panel."""
    if panel.empty:
        return pd.DataFrame()

    panel = panel.copy()
    panel["report_date"] = pd.to_datetime(panel["report_date"]).dt.normalize()

    dates = sorted(panel["report_date"].unique())
    symbols = panel["symbol"].unique()
    rows = []

    for rd in dates:
        day = panel[panel["report_date"] == rd]
        for sym in symbols:
            sym_day = day[day["symbol"] == sym]
            if sym_day.empty:
                continue
            row: dict = {"report_date": rd, "symbol": sym}
            acc = sym_day[sym_day["side"] == "accumulation"]
            dist = sym_day[sym_day["side"] == "distribution"]

            for side_name, sdf in [("acc", acc), ("dist", dist)]:
                for _, r in sdf.iterrows():
                    h = r["horizon"]
                    prefix = f"{side_name}_{h}"
                    row[f"{prefix}_net_amount"] = r.get("net_amount_sum", 0) or 0
                    row[f"{prefix}_float_turnover"] = r.get("net_float_turnover_mean", 0) or 0
                    row[f"{prefix}_power_score"] = POWER_SCORES.get(r.get("dominant_power"), 0) or 0
                    if side_name == "acc":
                        row.setdefault("ltp", r.get("ltp"))
                        row.setdefault("tech_demand_zone", r.get("tech_demand_zone"))
                        row.setdefault("tech_supply_zone", r.get("tech_supply_zone"))
                        row.setdefault("broker_concentration", r.get("broker_concentration", 0))

            for side_name, sdf in [("acc", acc), ("dist", dist)]:
                if sdf.empty:
                    continue
                for _, r in sdf.iterrows():
                    h = r["horizon"]
                    if row.get("ltp") is None or (isinstance(row.get("ltp"), float) and np.isnan(row.get("ltp"))):
                        row["ltp"] = r.get("ltp")
                    if h == "1D":
                        row[f"{side_name}_1D_power"] = r.get("dominant_power")

            rows.append(row)

    feat = pd.DataFrame(rows)
    if feat.empty:
        return feat

    feat = _add_composite_features(feat)
    return feat


def _horizon_cols(feat: pd.DataFrame, side: str, metric: str) -> list[str]:
    return [c for c in feat.columns if c.startswith(f"{side}_") and c.endswith(f"_{metric}")]


def _add_composite_features(feat: pd.DataFrame) -> pd.DataFrame:
    feat = feat.copy()

    # Multi-horizon accumulation convergence
    acc_power_cols = [c for c in feat.columns if c.startswith("acc_") and c.endswith("_power_score")]
    if acc_power_cols:
        weights = []
        scores = []
        for col in acc_power_cols:
            h = col.replace("acc_", "").replace("_power_score", "")
            w = 1.0 / max(HORIZON_ORDER.get(h, 99), 1)
            weights.append(w)
            scores.append(feat[col].fillna(0))
        w_arr = np.array(weights)
        mat = np.column_stack([s.values for s in scores])
        feat["acc_horizon_score"] = (mat * w_arr).sum(axis=1) / w_arr.sum()

    # MTF convergence: short horizons acc positive, aligned
    short_hs = ["1D", "2D", "3D", "1W"]
    short_acc = [f"acc_{h}_net_amount" for h in short_hs if f"acc_{h}_net_amount" in feat.columns]
    if short_acc:
        feat["mtf_convergence"] = (feat[short_acc].fillna(0) > 0).sum(axis=1) / len(short_acc)

    # Acc/Dist ratio
    acc_amt_cols = _horizon_cols(feat, "acc", "net_amount")
    dist_amt_cols = _horizon_cols(feat, "dist", "net_amount")
    if acc_amt_cols and dist_amt_cols:
        acc_total = feat[acc_amt_cols].fillna(0).sum(axis=1)
        dist_total = feat[dist_amt_cols].fillna(0).abs().sum(axis=1)
        feat["acc_dist_ratio"] = np.where(
            (acc_total + dist_total) > 0, acc_total / (acc_total + dist_total), 0.5
        )

    # Zone proximity
    if "ltp" in feat.columns and "tech_demand_zone" in feat.columns:
        feat["demand_zone_distance_pct"] = np.where(
            feat["ltp"].fillna(0) > 0,
            (feat["ltp"] - feat["tech_demand_zone"]) / feat["ltp"] * 100,
            np.nan,
        )
    if "ltp" in feat.columns and "tech_supply_zone" in feat.columns:
        feat["supply_zone_distance_pct"] = np.where(
            feat["ltp"].fillna(0) > 0,
            (feat["tech_supply_zone"] - feat["ltp"]) / feat["ltp"] * 100,
            np.nan,
        )

    # OFI from 1D
    if "acc_1D_net_amount" in feat.columns and "dist_1D_net_amount" in feat.columns:
        buy = feat["acc_1D_net_amount"].fillna(0)
        sell = feat["dist_1D_net_amount"].fillna(0).abs()
        denom = buy + sell
        feat["ofi"] = np.where(denom > 0, (buy - sell) / denom, 0)

    # Float turnover z-score (cross-sectional per day)
    if "acc_1D_float_turnover" in feat.columns:
        feat["float_turnover_zscore"] = feat.groupby("report_date")["acc_1D_float_turnover"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

    # Net amount momentum: 1D vs 1W
    if "acc_1D_net_amount" in feat.columns and "acc_1W_net_amount" in feat.columns:
        feat["net_amount_momentum"] = feat["acc_1D_net_amount"].fillna(0) - feat["acc_1W_net_amount"].fillna(0)

    # Power escalation
    if "acc_1D_power_score" in feat.columns and "acc_1M_power_score" in feat.columns:
        feat["power_escalation"] = feat["acc_1D_power_score"].fillna(0) - feat["acc_1M_power_score"].fillna(0)

    # Smart money index (0-100)
    feat["smart_money_score"] = _smart_money_score(feat)
    feat["floorsheet_momentum_score"] = _floorsheet_momentum_score(feat)
    feat["early_momentum_score"] = _early_momentum_score(feat)
    feat["distribution_risk_score"] = _distribution_risk_score(feat)

    # When accumulation sheets are missing, use distribution/broker proxy for EMS
    acc_cols = [c for c in feat.columns if c.startswith("acc_") and c.endswith("_net_amount")]
    has_acc = bool(acc_cols) and feat[acc_cols].fillna(0).abs().sum().sum() > 0
    if not has_acc:
        feat["early_momentum_score"] = feat[["early_momentum_score", "floorsheet_momentum_score"]].max(axis=1)

    return feat


def _smart_money_score(feat: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=feat.index)
    if "acc_horizon_score" in feat.columns:
        score += feat["acc_horizon_score"].fillna(0) / 3 * 35
    if "mtf_convergence" in feat.columns:
        score += feat["mtf_convergence"].fillna(0) * 25
    if "acc_dist_ratio" in feat.columns:
        score += feat["acc_dist_ratio"].fillna(0.5) * 20
    if "broker_concentration" in feat.columns:
        score += feat["broker_concentration"].fillna(0).clip(0, 1) * 10
    if "demand_zone_distance_pct" in feat.columns:
        near_demand = (feat["demand_zone_distance_pct"].fillna(999).between(-3, 8)).astype(float)
        score += near_demand * 10
    return score.clip(0, 100)


def _floorsheet_momentum_score(feat: pd.DataFrame) -> pd.Series:
    """Early-movement proxy from distribution horizons when accumulation is absent."""
    score = pd.Series(0.0, index=feat.index)
    if "dist_1D_power_score" in feat.columns:
        light_short = (feat["dist_1D_power_score"].fillna(3) <= 1).astype(float)
        score += light_short * 28
    if "dist_2D_power_score" in feat.columns:
        score += (feat["dist_2D_power_score"].fillna(3) <= 1).astype(float) * 12
    if "dist_1W_power_score" in feat.columns:
        score += (feat["dist_1W_power_score"].fillna(0) >= 2).astype(float) * 15
    if "dist_1D_net_amount" in feat.columns and "dist_1M_net_amount" in feat.columns:
        short = feat["dist_1D_net_amount"].fillna(0).abs()
        long_ = feat["dist_1M_net_amount"].fillna(0).abs()
        score += (short < long_ * 0.5).astype(float) * 20
    if "float_turnover_zscore" in feat.columns:
        score += feat["float_turnover_zscore"].fillna(0).clip(0, 3) / 3 * 15
    if "broker_concentration" in feat.columns:
        score += feat["broker_concentration"].fillna(0).clip(0, 1) * 10
    return score.clip(0, 100)


def _early_momentum_score(feat: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=feat.index)
    if "acc_1D_power_score" in feat.columns:
        score += (feat["acc_1D_power_score"].fillna(0) / 3) * 30
    if "mtf_convergence" in feat.columns:
        score += feat["mtf_convergence"].fillna(0) * 25
    if "ofi" in feat.columns:
        score += feat["ofi"].fillna(0).clip(0, 1) * 20
    if "float_turnover_zscore" in feat.columns:
        score += feat["float_turnover_zscore"].fillna(0).clip(0, 3) / 3 * 15
    if "demand_zone_distance_pct" in feat.columns:
        near = (feat["demand_zone_distance_pct"].fillna(999).between(0, 5)).astype(float)
        score += near * 10
    return score.clip(0, 100)


def _distribution_risk_score(feat: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=feat.index)
    dist_cols = [c for c in feat.columns if c.startswith("dist_") and c.endswith("_power_score")]
    if dist_cols:
        score += feat[dist_cols].fillna(0).max(axis=1) / 3 * 50
    if "dist_1D_net_amount" in feat.columns:
        score += (feat["dist_1D_net_amount"].fillna(0).abs() > 0).astype(float) * 25
    long_dist = [c for c in feat.columns if c.startswith("dist_") and ("1Y" in c or "2Y" in c or "3Y" in c) and c.endswith("_power_score")]
    if long_dist:
        score += (feat[long_dist].fillna(0).max(axis=1) >= 2).astype(float) * 25
    return score.clip(0, 100)
