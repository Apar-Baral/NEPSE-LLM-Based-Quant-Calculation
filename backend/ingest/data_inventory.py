"""Report what is loaded from Data/Accumulation vs Data/Distribution folders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.config import ROOT
from backend.ingest.parser import detect_side


def _scan_folder(folder: Path) -> list[dict]:
    rows = []
    if not folder.exists():
        return rows
    for f in sorted(folder.glob("*.csv")) + sorted(folder.glob("*.xlsx")) + sorted(folder.glob("*.xls")):
        try:
            if f.suffix.lower() == ".csv":
                cols = list(pd.read_csv(f, nrows=0, encoding="utf-8-sig").columns)
            else:
                xl = pd.ExcelFile(f)
                cols = list(pd.read_excel(f, sheet_name=xl.sheet_names[0], nrows=0).columns)
            side = detect_side(cols)
            hint = "OK for accumulation EMS" if side == "accumulation" else "Distribution export only"
            rows.append(
                {
                    "file": f.name,
                    "folder": folder.name,
                    "detected_side": side,
                    "hint": hint,
                }
            )
        except Exception as exc:
            rows.append({"file": f.name, "folder": folder.name, "detected_side": "error", "hint": str(exc)[:80]})
    return rows


def data_folder_inventory() -> dict:
    acc_dir = ROOT / "Data" / "Accumulation Data"
    dist_dir = ROOT / "Data" / "Distribution Data"
    acc_rows = _scan_folder(acc_dir)
    dist_rows = _scan_folder(dist_dir)
    acc_files = len(acc_rows)
    dist_files = len(dist_rows)
    acc_side_files = sum(1 for r in acc_rows if r.get("detected_side") == "accumulation")
    dist_side_files = sum(1 for r in dist_rows if r.get("detected_side") == "distribution")

    return {
        "accumulation_dir": str(acc_dir),
        "distribution_dir": str(dist_dir),
        "accumulation_file_count": acc_files,
        "distribution_file_count": dist_files,
        "accumulation_side_files": acc_side_files,
        "distribution_side_files": dist_side_files,
        "files": acc_rows + dist_rows,
        "has_true_accumulation": acc_side_files > 0,
        "message": _build_message(acc_files, dist_files, acc_side_files, dist_side_files),
    }


def _build_message(acc_files: int, dist_files: int, acc_side: int, dist_side: int) -> str:
    if acc_side > 0:
        return f"Accumulation data detected ({acc_side} file(s)). Floorsheet + EMS use acc + dist."
    if acc_files == 0 and dist_files > 0:
        return (
            f"Only **distribution** CSVs found ({dist_files} in Distribution Data). "
            "**Accumulation Data** folder is empty — floorsheet uses **distribution proxy** (not zero by design after pipeline). "
            "Export separate accumulation floorsheet (columns: Net **Buy** Amt, **Accumulation** Power) into `Data/Accumulation Data/`."
        )
    if acc_files == 0 and dist_files == 0:
        return "No CSV/Excel files in Data folders. Add floorsheet exports."
    return "No files with accumulation column headers found."


def panel_side_summary(panel: pd.DataFrame) -> dict:
    if panel.empty or "side" not in panel.columns:
        return {"distribution_rows": 0, "accumulation_rows": 0}
    vc = panel["side"].astype(str).str.lower().value_counts()
    return {
        "distribution_rows": int(vc.get("distribution", 0)),
        "accumulation_rows": int(vc.get("accumulation", 0)),
    }
