from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from backend.config import ROOT
from backend.ingest.excel_loader import load_csv_file, load_side_folder


def resolve_all_in_one_dir() -> Path | None:
    """Find Data/All in one Data (case-insensitive folder name)."""
    data_root = ROOT / "Data"
    if not data_root.exists():
        return None
    for child in data_root.iterdir():
        if not child.is_dir():
            continue
        key = child.name.lower().replace(" ", "").replace("-", "")
        if key in ("allinonedata", "allinone"):
            return child
    fixed = data_root / "All in one Data"
    return fixed if fixed.is_dir() else None


def _load_csv_folder(folder: Path, report_date: date | None = None) -> pd.DataFrame:
    """Load every CSV; side per file from column headers (acc or dist)."""
    if not folder.exists():
        return pd.DataFrame()
    files = sorted(folder.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    rd = report_date or date.today()
    frames = []
    for i, f in enumerate(files):
        snap_date = pd.Timestamp(rd) - pd.Timedelta(days=len(files) - 1 - i)
        frames.append(load_csv_file(f, report_date=snap_date.date()))
    return pd.concat(frames, ignore_index=True)


def backfill_all_in_one_data(report_date: date | None = None) -> pd.DataFrame:
    """Load combined acc + dist floorsheet exports from Data/All in one Data."""
    folder = resolve_all_in_one_dir()
    if folder is None:
        return pd.DataFrame()
    return _load_csv_folder(folder, report_date)


def backfill_distribution_data(report_date: date | None = None) -> pd.DataFrame:
    dist_dir = ROOT / "Data" / "Distribution Data"
    if not dist_dir.exists():
        return pd.DataFrame()
    return _load_csv_folder(dist_dir, report_date)


def backfill_accumulation_data(report_date: date | None = None) -> pd.DataFrame:
    acc_dir = ROOT / "Data" / "Accumulation Data"
    if not acc_dir.exists() or not list(acc_dir.glob("*.csv")):
        return pd.DataFrame()
    return load_side_folder(acc_dir, "accumulation", report_date=report_date)


def backfill_combined_floorsheet(report_date: date | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Prefer Data/All in one Data when it has CSVs (acc + dist auto-detected per file).
    Otherwise fall back to separate Accumulation + Distribution folders.
    """
    rd = report_date or date.today()
    aio = resolve_all_in_one_dir()
    aio_files = sorted(aio.glob("*.csv")) if aio else []

    if aio_files:
        panel = _load_csv_folder(aio, rd)
        sides = panel["side"].astype(str).str.lower().value_counts().to_dict() if "side" in panel.columns else {}
        return panel, {
            "source": "all_in_one",
            "folder": str(aio),
            "file_count": len(aio_files),
            "accumulation_rows": int(sides.get("accumulation", 0)),
            "distribution_rows": int(sides.get("distribution", 0)),
        }

    dist_legacy = backfill_distribution_data(report_date=rd)
    acc_legacy = backfill_accumulation_data(report_date=rd)
    frames = [f for f in (dist_legacy, acc_legacy) if not f.empty]
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    sides = panel["side"].astype(str).str.lower().value_counts().to_dict() if "side" in panel.columns else {}
    return panel, {
        "source": "legacy_folders",
        "folder": "Distribution Data + Accumulation Data",
        "file_count": 0,
        "accumulation_rows": int(sides.get("accumulation", 0)),
        "distribution_rows": int(sides.get("distribution", 0)),
    }
