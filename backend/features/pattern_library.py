from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from backend.config import PROCESSED_DIR

PATTERN_STORE = PROCESSED_DIR / "pattern_vectors.parquet"
META_PATH = PROCESSED_DIR / "pattern_meta.json"

FEATURE_KEYS = [
    "early_momentum_score",
    "smart_money_score",
    "acc_horizon_score",
    "mtf_convergence",
    "acc_dist_ratio",
    "ofi",
    "float_turnover_zscore",
    "distribution_risk_score",
]


def _vectorize(row: pd.Series) -> np.ndarray:
    return np.array([float(row.get(k, 0) or 0) for k in FEATURE_KEYS], dtype=float)


def build_pattern_store(features: pd.DataFrame, labels: pd.DataFrame) -> int:
    """Store feature vectors with forward outcomes for k-NN matching."""
    if features.empty:
        return 0

    merged = features.merge(
        labels,
        left_on=["symbol", "report_date"],
        right_on=["symbol", "as_of_date"],
        how="left",
    )
    rows = []
    for _, row in merged.iterrows():
        vec = _vectorize(row)
        if np.all(vec == 0):
            continue
        rows.append(
            {
                "symbol": row["symbol"],
                "report_date": row["report_date"],
                "vector": vec.tolist(),
                "early_momentum_score": float(row.get("early_momentum_score", 0) or 0),
                "forward_return_10d": float(row.get("forward_return_10d", 0) or 0),
                "long_momentum_10d": int(row.get("long_momentum_10d", 0) or 0),
            }
        )

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PATTERN_STORE, index=False)
    winners = df[df["long_momentum_10d"] == 1]
    meta = {
        "total": len(df),
        "winners": len(winners),
        "hit_rate": float(winners["long_momentum_10d"].mean()) if len(df) else 0,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return len(df)


def find_historical_analogs(
    row: pd.Series,
    k: int = 5,
    min_return_pct: float = 5.0,
) -> dict:
    """k-NN similarity to past early movers."""
    if not PATTERN_STORE.exists():
        return {"analog_count": 0, "analog_hit_rate": 0.0, "analogs": []}

    store = pd.read_parquet(PATTERN_STORE)
    if len(store) < k + 1:
        return {"analog_count": 0, "analog_hit_rate": 0.0, "analogs": []}

    vectors = np.array(store["vector"].tolist())
    query = _vectorize(row).reshape(1, -1)

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(store)), metric="euclidean")
    nn.fit(vectors)
    dists, idxs = nn.kneighbors(query)

    analogs = []
    for dist, idx in zip(dists[0], idxs[0]):
        rec = store.iloc[int(idx)]
        if rec["symbol"] == row.get("symbol") and str(rec["report_date"]) == str(row.get("report_date")):
            continue
        analogs.append(
            {
                "symbol": rec["symbol"],
                "date": str(rec["report_date"]),
                "distance": float(dist),
                "forward_return_10d": float(rec["forward_return_10d"]),
                "was_winner": bool(rec["long_momentum_10d"]),
            }
        )
        if len(analogs) >= k:
            break

    movers = [a for a in analogs if a["forward_return_10d"] >= min_return_pct or a["was_winner"]]
    hit_rate = sum(1 for a in analogs if a["was_winner"]) / len(analogs) if analogs else 0.0

    return {
        "analog_count": len(analogs),
        "analog_mover_count": len(movers),
        "analog_hit_rate": hit_rate,
        "analogs": analogs,
    }


def enrich_with_analogs(df: pd.DataFrame) -> pd.DataFrame:
    """Add k-NN analog columns to feature/prediction dataframe."""
    out = df.copy()
    analog_counts = []
    hit_rates = []
    for _, row in out.iterrows():
        res = find_historical_analogs(row)
        analog_counts.append(res["analog_count"])
        hit_rates.append(res["analog_hit_rate"])
    out["analog_count"] = analog_counts
    out["analog_hit_rate"] = hit_rates
    return out
