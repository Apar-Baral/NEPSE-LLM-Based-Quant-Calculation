from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from backend.ingest.parser import parse_numeric


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    rename = {
        "ticker": "symbol",
        "scrip": "symbol",
        "close_price": "close",
        "ltp": "close",
        "vol": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    required = {"symbol", "date", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"OHLCV file must contain {required}, got {set(df.columns)}")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = df[col].apply(parse_numeric)
    return df.sort_values(["symbol", "date"])


def build_ltp_proxy_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Use daily-horizon LTP as price proxy when OHLCV unavailable."""
    daily = panel[panel["horizon"] == "1D"].copy()
    if daily.empty:
        daily = panel.sort_values("horizon").groupby(["symbol", "report_date"], as_index=False).first()
    rows = []
    for _, r in daily.iterrows():
        ltp = r.get("ltp")
        if ltp is None or (isinstance(ltp, float) and pd.isna(ltp)):
            continue
        rows.append(
            {
                "symbol": r["symbol"],
                "date": pd.Timestamp(r["report_date"]).normalize(),
                "open": ltp,
                "high": ltp,
                "low": ltp,
                "close": ltp,
                "volume": r.get("buy_qty_sum", 0) or 0,
            }
        )
    return pd.DataFrame(rows)


def fetch_nepse_prices_stub(symbols: list[str] | None = None) -> pd.DataFrame:
    """Placeholder for NEPSE API/scraper integration."""
    return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
