from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import load_yaml_config


def compute_forward_returns(ohlcv: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    cfg = load_yaml_config("settings.yaml")["labels"]
    windows = windows or cfg["forward_windows_days"]
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["symbol", "date"])

    out_rows = []
    for sym, grp in df.groupby("symbol"):
        grp = grp.set_index("date").sort_index()
        close = grp["close"]
        for w in windows:
            fwd = close.shift(-w) / close - 1
            for dt, ret in fwd.items():
                if pd.isna(ret):
                    continue
                out_rows.append({"symbol": sym, "as_of_date": dt, f"forward_return_{w}d": ret * 100})

    if not out_rows:
        return pd.DataFrame()

    merged = pd.DataFrame(out_rows)
    pivot_cols = [c for c in merged.columns if c.startswith("forward_return_")]
    result = merged.groupby(["symbol", "as_of_date"], as_index=False)[pivot_cols].first()
    return result


def compute_forward_max_drawdown(ohlcv: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    rows = []
    for sym, grp in df.groupby("symbol"):
        grp = grp.sort_values("date")
        closes = grp["close"].values
        dates = grp["date"].values
        for i in range(len(closes)):
            end = min(i + window + 1, len(closes))
            path = closes[i:end]
            if len(path) < 2:
                continue
            peak = np.maximum.accumulate(path)
            dd = (path - peak) / peak * 100
            rows.append({"symbol": sym, "as_of_date": dates[i], f"max_drawdown_{window}d": dd.min()})
    return pd.DataFrame(rows)


def build_labels(features: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    cfg = load_yaml_config("settings.yaml")["labels"]
    threshold = cfg["forward_return_threshold_pct"]
    dd_limit = cfg["max_drawdown_threshold_pct"]

    fwd = compute_forward_returns(ohlcv)
    dd = compute_forward_max_drawdown(ohlcv, window=10)

    labels = features[["report_date", "symbol"]].copy()
    labels = labels.rename(columns={"report_date": "as_of_date"})
    labels["as_of_date"] = pd.to_datetime(labels["as_of_date"]).dt.normalize()

    if not fwd.empty:
        labels = labels.merge(fwd, on=["symbol", "as_of_date"], how="left")
    if not dd.empty:
        labels = labels.merge(dd, on=["symbol", "as_of_date"], how="left")

    ret_col = "forward_return_10d"
    dd_col = "max_drawdown_10d"
    if ret_col in labels.columns:
        labels["long_momentum_10d"] = (
            (labels[ret_col] >= threshold)
            & (labels.get(dd_col, -999).fillna(-999) >= -dd_limit)
        ).astype(int)
    else:
        labels["long_momentum_10d"] = 0

    if "forward_return_5d" in labels.columns:
        labels["long_momentum_5d"] = (labels["forward_return_5d"] >= threshold).astype(int)

    if "forward_return_3d" in labels.columns:
        labels["early_onset"] = (labels["forward_return_3d"] >= threshold * 0.6).astype(int)
    else:
        labels["early_onset"] = 0

    return labels
