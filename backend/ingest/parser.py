from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd

Side = Literal["accumulation", "distribution"]

COLUMN_ALIASES = {
    "symbol": ["symbol"],
    "ltp": ["ltp"],
    "broker": ["seller broker", "buyer broker", "broker"],
    "buy_qty": ["buy qty", "buy quantity"],
    "sell_qty": ["sell qty", "sell quantity"],
    "broker_holding": ["broker holding"],
    "net_float_turnover": ["net float turnover"],
    "net_amount": ["net sell amt", "net buy amt", "net sell amount", "net buy amount"],
    "avg_rate": ["avg sell rate", "avg buy rate", "avg rate"],
    "power": [
        "distribution power",
        "accumulation power",
        "distribution             power",
        "accumulation             power",
    ],
    "tech_supply_zone": ["tech supply zone"],
    "tech_demand_zone": ["tech demand zone"],
}


def normalize_column(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def detect_side(columns: list[str]) -> Side:
    joined = " ".join(normalize_column(c) for c in columns)
    if "accumulation" in joined or "net buy" in joined or "buyer broker" in joined:
        return "accumulation"
    return "distribution"


def parse_numeric(value: object) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if s in ("", "-", "nan", "None"):
        return None
    if "infinity" in s.lower():
        return None
    s = s.replace(",", "")
    s = re.sub(r"\s*Lac\.?\s*", "", s, flags=re.IGNORECASE)
    s = s.replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_power(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if s in ("", "-"):
        return None
    for p in ("Heavy", "Medium", "Light"):
        if p.lower() in s.lower():
            return p
    return s


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map: dict[str, str] = {}
    normalized = {normalize_column(c): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                col_map[normalized[alias]] = canonical
                break
    out = df.rename(columns=col_map)
    return out


def clean_raw_df(df: pd.DataFrame, side: Side | None = None) -> pd.DataFrame:
    if side is None:
        side = detect_side(list(df.columns))
    df = map_columns(df)
    if "symbol" not in df.columns:
        raise ValueError("Missing Symbol column after normalization")

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df = df[df["symbol"].notna() & (df["symbol"] != "") & (df["symbol"] != "-")]

    for col in ("ltp", "buy_qty", "sell_qty", "broker_holding", "net_float_turnover", "net_amount", "avg_rate", "tech_supply_zone", "tech_demand_zone"):
        if col in df.columns:
            df[col] = df[col].apply(parse_numeric)

    if "power" in df.columns:
        df["power"] = df["power"].apply(parse_power)

    df["side"] = side
    if side == "distribution" and "net_amount" in df.columns:
        df["net_amount"] = df["net_amount"].abs() * -1

    if side == "accumulation" and "net_amount" in df.columns:
        df["net_amount"] = df["net_amount"].abs()

    return df


def aggregate_symbol_level(df: pd.DataFrame, power_scores: dict[str, int]) -> pd.DataFrame:
    """Aggregate broker-level rows to symbol-level snapshot."""

    def power_score(p: object) -> int:
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return 0
        return power_scores.get(str(p), 0)

    df = df.copy()
    df["_power_score"] = df.get("power", pd.Series(dtype=object)).apply(power_score)

    agg: dict[str, tuple] = {
        "ltp": ("ltp", "first"),
        "buy_qty_sum": ("buy_qty", "sum"),
        "sell_qty_sum": ("sell_qty", "sum"),
        "broker_holding_sum": ("broker_holding", "sum"),
        "net_amount_sum": ("net_amount", "sum"),
        "net_float_turnover_mean": ("net_float_turnover", "mean"),
        "avg_rate_mean": ("avg_rate", "mean"),
        "tech_supply_zone": ("tech_supply_zone", "first"),
        "tech_demand_zone": ("tech_demand_zone", "first"),
        "broker_count": ("broker", "nunique"),
        "heavy_broker_count": ("power", lambda s: (s == "Heavy").sum()),
        "dominant_power_score": ("_power_score", "max"),
    }

    available = {k: v for k, v in agg.items() if v[0] in df.columns or k == "broker_count"}
    if "broker" not in df.columns:
        df["broker"] = df.index

    grouped = df.groupby("symbol", as_index=False).agg(**{k: v for k, v in available.items() if k in agg})

    score_to_power = {v: k for k, v in power_scores.items()}
    grouped["dominant_power"] = grouped["dominant_power_score"].map(score_to_power)

    if "broker_holding_sum" in grouped.columns and grouped["broker_holding_sum"].notna().any():
        holdings = df.groupby("symbol")["broker_holding"].apply(lambda x: x.fillna(0).abs().sum())
        grouped = grouped.merge(holdings.rename("_abs_hold"), on="symbol", how="left")
        top3 = df.groupby("symbol")["broker_holding"].apply(
            lambda x: x.fillna(0).abs().nlargest(min(3, len(x))).sum()
        )
        grouped = grouped.merge(top3.rename("_top3"), on="symbol", how="left")
        grouped["broker_concentration"] = np.where(
            grouped["_abs_hold"] > 0, grouped["_top3"] / grouped["_abs_hold"], 0
        )
        grouped.drop(columns=["_abs_hold", "_top3"], inplace=True)
    else:
        grouped["broker_concentration"] = 0.0

    return grouped
