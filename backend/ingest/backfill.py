from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from backend.config import ROOT
from backend.ingest.excel_loader import load_csv_file, load_side_folder


def backfill_distribution_data(report_date: date | None = None) -> pd.DataFrame:
    dist_dir = ROOT / "Data" / "Distribution Data"
    if not dist_dir.exists():
        return pd.DataFrame()
    rd = report_date or date.today()
    files = sorted(dist_dir.glob("*.csv"))
    frames = []
    for i, f in enumerate(files):
        # Spread legacy snapshots across days for demo panel history
        snap_date = pd.Timestamp(rd) - pd.Timedelta(days=len(files) - 1 - i)
        frames.append(load_csv_file(f, report_date=snap_date.date()))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def backfill_accumulation_data(report_date: date | None = None) -> pd.DataFrame:
    acc_dir = ROOT / "Data" / "Accumulation Data"
    if not acc_dir.exists() or not list(acc_dir.glob("*.csv")):
        return pd.DataFrame()
    return load_side_folder(acc_dir, "accumulation", report_date=report_date)
