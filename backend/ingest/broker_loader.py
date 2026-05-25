from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from backend.config import load_yaml_config
from backend.ingest.excel_loader import load_csv_file, load_excel_workbook, resolve_horizon_from_csv_suffix
from backend.ingest.parser import clean_raw_df, detect_side


def _broker_id(val: object) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s == "-":
        return None
    # NEPSE floorsheet uses numeric broker codes (e.g. 58, 49)
    import re

    m = re.search(r"\d+", s)
    return m.group(0) if m else s


def broker_rows_from_cleaned(cleaned: pd.DataFrame, horizon: str, side: str, report_date, source: str) -> pd.DataFrame:
    if cleaned.empty or "broker" not in cleaned.columns:
        return pd.DataFrame()
    df = cleaned.copy()
    df["broker_id"] = df["broker"].apply(_broker_id)
    df = df[df["broker_id"].notna()]
    buy = pd.to_numeric(df.get("buy_qty", 0), errors="coerce").fillna(0)
    sell = pd.to_numeric(df.get("sell_qty", 0), errors="coerce").fillna(0)
    rows = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "broker_id": df["broker_id"],
            "ltp": pd.to_numeric(df.get("ltp"), errors="coerce"),
            "buy_qty": buy,
            "sell_qty": sell,
            "net_qty": buy - sell,
            "net_amount": pd.to_numeric(df.get("net_amount", 0), errors="coerce").fillna(0),
            "broker_holding": pd.to_numeric(df.get("broker_holding", 0), errors="coerce").fillna(0),
            "power": df.get("power"),
            "horizon": horizon,
            "side": side,
            "report_date": pd.Timestamp(report_date),
            "source_file": source,
        }
    )
    rows["activity_qty"] = rows["buy_qty"] + rows["sell_qty"]
    return rows


def load_csv_broker_detail(path: Path, horizon: str | None = None, report_date: date | None = None) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    side = detect_side(list(raw.columns))
    cleaned = clean_raw_df(raw, side=side)
    h = horizon or resolve_horizon_from_csv_suffix(path) or "unknown"
    return broker_rows_from_cleaned(cleaned, h, side, report_date or date.today(), path.name)


def load_excel_broker_detail(path: Path, report_date: date | None = None) -> pd.DataFrame:
    from backend.ingest.excel_loader import resolve_horizon_from_sheet

    xl = pd.ExcelFile(path)
    frames = []
    rd = report_date or date.today()
    for sheet in xl.sheet_names:
        horizon = resolve_horizon_from_sheet(sheet)
        if horizon is None:
            continue
        raw = pd.read_excel(path, sheet_name=sheet)
        side = detect_side(list(raw.columns))
        cleaned = clean_raw_df(raw, side=side)
        frames.append(broker_rows_from_cleaned(cleaned, horizon, side, rd, path.name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def backfill_broker_panel_from_data(root: Path | None = None) -> pd.DataFrame:
    """Build broker-level panel from Data/ CSV folders (All in one Data preferred)."""
    from backend.config import ROOT
    from backend.ingest.backfill import resolve_all_in_one_dir

    root = root or ROOT
    frames = []

    aio = resolve_all_in_one_dir()
    if aio and list(aio.glob("*.csv")):
        for f in sorted(aio.glob("*.csv")):
            try:
                frames.append(load_csv_broker_detail(f))
            except Exception:
                continue
    else:
        for folder in ("Distribution Data", "Accumulation Data"):
            path = root / "Data" / folder
            if not path.exists():
                continue
            for f in sorted(path.glob("*.csv")):
                try:
                    frames.append(load_csv_broker_detail(f))
                except Exception:
                    continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
