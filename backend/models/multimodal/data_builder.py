from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import PROCESSED_DIR, load_yaml_config

SHORT_HORIZONS = ("1D", "2D", "3D", "4D", "1W")
LLM_CACHE = PROCESSED_DIR / "llm_scanner_scores.json"
TIER_MAP = {"Confirmed": 4, "Trigger": 3, "Setup": 2, "Watch": 1, "Neutral": 0, "Invalidated": -1}


def _horizon_net(feat: pd.Series, side: str, h: str) -> float:
    col = f"{side}_{h}_net_amount"
    v = feat.get(col, 0)
    try:
        return float(v) if pd.notna(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _horizon_power(feat: pd.Series, side: str, h: str) -> float:
    col = f"{side}_{h}_power_score"
    v = feat.get(col, 0)
    try:
        return float(v) if pd.notna(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def build_temporal_tensor(row: pd.Series) -> np.ndarray:
    """Shape (T, C): T horizons, C=4 channels [acc_net, dist_net, acc_power, dist_power]."""
    seq = []
    for h in SHORT_HORIZONS:
        acc = _horizon_net(row, "acc", h)
        dist = _horizon_net(row, "dist", h)
        scale = max(abs(acc), abs(dist), 1.0)
        seq.append([acc / scale, dist / scale, _horizon_power(row, "acc", h) / 3.0, _horizon_power(row, "dist", h) / 3.0])
    return np.array(seq, dtype=np.float32)


def accumulation_phase_weight(row: pd.Series) -> float:
    """Phase-aware sample weight: reward early accumulation, penalize distribution."""
    drs = float(row.get("distribution_risk_score", 50) or 50) / 100.0
    acc = float(row.get("acc_horizon_score", 0) or 0) / 100.0
    ems = float(row.get("early_momentum_score", 0) or 0) / 100.0
    shakeout = 1.0 if row.get("pattern_dist_shakeout") in (True, 1, "1", "True") else 0.0
    w = 0.35 + acc * 0.35 + ems * 0.2 + shakeout * 0.25 - drs * 0.45
    return float(np.clip(w, 0.15, 2.0))


def _llm_cache_for_symbol(symbol: str, report_date) -> dict:
    if not LLM_CACHE.exists():
        return {}
    try:
        cache = json.loads(LLM_CACHE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    rd = str(pd.Timestamp(report_date).date()) if report_date is not None else "latest"
    return cache.get(f"scores_{rd}", {}).get(symbol, cache.get("scores_latest", {}).get(symbol, {}))


def build_semantic_vector(row: pd.Series) -> np.ndarray:
    """Numeric semantic proxy from LLM cache + patterns (no API at train time)."""
    sym = row.get("symbol", "")
    cached = _llm_cache_for_symbol(str(sym), row.get("report_date"))
    p_long = float(cached.get("llm_p_long", row.get("llm_p_long", row.get("p_long_momentum", 0.3)) or 0.3))
    tier = str(cached.get("llm_tier", row.get("signal_tier", "Neutral")))
    note = str(cached.get("llm_note", row.get("llm_note", "")) or "").lower()
    tier_score = TIER_MAP.get(tier, 0) / 4.0

    kw_acc = 1.0 if any(k in note for k in ("accum", "shakeout", "early", "long")) else 0.0
    kw_dist = 1.0 if any(k in note for k in ("distrib", "avoid", "heavy sell")) else 0.0
    kw_broker = 1.0 if "broker" in note else 0.0

    return np.array(
        [
            p_long,
            tier_score,
            len(note) / 200.0,
            kw_acc,
            kw_dist,
            kw_broker,
            float(row.get("mtf_convergence", 0) or 0),
            float(row.get("broker_pressure", 0) or 0) / 100.0,
            float(row.get("floorsheet_momentum_score", 0) or 0) / 100.0,
            float(row.get("pattern_dist_shakeout", 0) or 0),
            float(row.get("pattern_horizon_ladder", 0) or 0),
            float(row.get("smart_money_score", 0) or 0) / 100.0,
            float(row.get("acc_dist_ratio", 1) or 1) / 10.0,
            float(row.get("float_turnover_zscore", 0) or 0) / 3.0,
            float(row.get("ofi", 0) or 0),
            1.0,
        ],
        dtype=np.float32,
    )


def broker_sets(broker_panel: pd.DataFrame, horizon: str = "1D") -> dict[str, set[str]]:
    if broker_panel.empty:
        return {}
    sub = broker_panel[broker_panel["horizon"] == horizon] if "horizon" in broker_panel.columns else broker_panel
    out: dict[str, set[str]] = {}
    for sym, grp in sub.groupby("symbol"):
        ids = grp["broker_id"].astype(str).tolist()
        out[str(sym)] = set(ids)
    return out


def build_graph_adjacency(symbols: list[str], broker_map: dict[str, set[str]], top_k: int = 8) -> np.ndarray:
    n = len(symbols)
    adj = np.eye(n, dtype=np.float32)
    for i, si in enumerate(symbols):
        bi = broker_map.get(si, set())
        for j, sj in enumerate(symbols):
            if i == j:
                continue
            bj = broker_map.get(sj, set())
            if not bi or not bj:
                continue
            inter = len(bi & bj)
            union = len(bi | bj) or 1
            adj[i, j] = inter / union
    # Sparsify: keep top_k neighbors per row
    for i in range(n):
        row = adj[i].copy()
        row[i] = 1.0
        idx = np.argsort(row)[::-1][: top_k + 1]
        mask = np.zeros(n)
        mask[idx] = 1.0
        adj[i] *= mask
    # Symmetrize
    adj = (adj + adj.T) / 2.0
    d = adj.sum(axis=1, keepdims=True) + 1e-6
    return (adj / d).astype(np.float32)


def build_training_batch(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    broker_panel: pd.DataFrame | None = None,
    max_samples: int | None = None,
) -> dict:
    df = features.merge(
        labels[["symbol", "as_of_date", "long_momentum_10d"]],
        left_on=["symbol", "report_date"],
        right_on=["symbol", "as_of_date"],
        how="inner",
    )
    df = df.dropna(subset=["long_momentum_10d"])
    if df.empty:
        return {}
    df = df.sort_values("report_date")
    if max_samples and len(df) > max_samples:
        df = df.tail(max_samples)

    bp = broker_panel if broker_panel is not None and not broker_panel.empty else pd.DataFrame()
    broker_map = broker_sets(bp)
    temporal, semantic, y, weights, symbols, dates = [], [], [], [], [], []

    for _, row in df.iterrows():
        temporal.append(build_temporal_tensor(row))
        semantic.append(build_semantic_vector(row))
        y.append(int(row["long_momentum_10d"]))
        weights.append(accumulation_phase_weight(row))
        symbols.append(str(row["symbol"]))
        dates.append(row["report_date"])

    # Batch graph over unique symbols in this chunk (last date slice for inference uses per-batch)
    uniq = list(dict.fromkeys(symbols))
    adj = build_graph_adjacency(uniq, broker_map)
    sym_to_idx = {s: i for i, s in enumerate(uniq)}
    node_idx = [sym_to_idx[s] for s in symbols]

    return {
        "temporal": np.stack(temporal),
        "semantic": np.stack(semantic),
        "y": np.array(y, dtype=np.float32),
        "weights": np.array(weights, dtype=np.float32),
        "adjacency": adj,
        "node_idx": np.array(node_idx, dtype=np.int64),
        "symbols": symbols,
        "report_dates": dates,
        "n_nodes": len(uniq),
    }
