from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from backend.config import load_yaml_config
from backend.ingest.parser import aggregate_symbol_level, clean_raw_df, detect_side


def resolve_horizon_from_sheet(sheet_name: str) -> str | None:
    cfg = load_yaml_config("horizons.yaml")
    name = sheet_name.strip().lower()
    for h in cfg["horizons"]:
        if h["key"].lower() == name:
            return h["key"]
        for alias in h.get("aliases", []):
            if alias.lower() == name:
                return h["key"]
    return None


def resolve_horizon_from_csv_suffix(path: Path) -> str | None:
    cfg = load_yaml_config("horizons.yaml")
    match = re.search(r"\((\d+)\)", path.stem)
    if not match:
        return None
    suffix = match.group(1)
    for h in cfg["horizons"]:
        if str(h.get("csv_suffix")) == suffix:
            return h["key"]
    return None


def load_csv_file(path: Path, horizon: str | None = None, report_date: date | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    side = detect_side(list(df.columns))
    cleaned = clean_raw_df(df, side=side)
    power_scores = load_yaml_config("horizons.yaml")["power_scores"]
    agg = aggregate_symbol_level(cleaned, power_scores)
    agg["horizon"] = horizon or resolve_horizon_from_csv_suffix(path) or "unknown"
    agg["side"] = side
    agg["report_date"] = pd.Timestamp(report_date or date.today())
    agg["source_file"] = path.name
    return agg


def load_excel_workbook(path: Path, report_date: date | None = None) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    power_scores = load_yaml_config("horizons.yaml")["power_scores"]
    rd = report_date or date.today()

    for sheet in xl.sheet_names:
        horizon = resolve_horizon_from_sheet(sheet)
        if horizon is None:
            continue
        raw = pd.read_excel(path, sheet_name=sheet)
        side = detect_side(list(raw.columns))
        cleaned = clean_raw_df(raw, side=side)
        agg = aggregate_symbol_level(cleaned, power_scores)
        agg["horizon"] = horizon
        agg["side"] = side
        agg["report_date"] = pd.Timestamp(rd)
        agg["source_file"] = path.name
        frames.append(agg)

    if not frames:
        raise ValueError(f"No recognized horizon sheets in {path.name}")
    return pd.concat(frames, ignore_index=True)


def load_side_folder(folder: Path, side: str, report_date: date | None = None) -> pd.DataFrame:
    """Load all CSV files from a folder (legacy backfill)."""
    files = sorted(folder.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    frames = [load_csv_file(f, report_date=report_date) for f in files]
    return pd.concat(frames, ignore_index=True)
