"""Report what is loaded from Data/Accumulation vs Data/Distribution folders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.config import ROOT
from backend.ingest.backfill import resolve_all_in_one_dir
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
    aio_dir = resolve_all_in_one_dir()
    acc_dir = ROOT / "Data" / "Accumulation Data"
    dist_dir = ROOT / "Data" / "Distribution Data"

    aio_rows = _scan_folder(aio_dir) if aio_dir else []
    acc_rows = _scan_folder(acc_dir)
    dist_rows = _scan_folder(dist_dir)

    aio_acc = sum(1 for r in aio_rows if r.get("detected_side") == "accumulation")
    aio_dist = sum(1 for r in aio_rows if r.get("detected_side") == "distribution")
    acc_side_files = sum(1 for r in acc_rows if r.get("detected_side") == "accumulation")
    dist_side_files = sum(1 for r in dist_rows if r.get("detected_side") == "distribution")

    use_aio = len(aio_rows) > 0
    has_acc = aio_acc > 0 if use_aio else acc_side_files > 0

    return {
        "all_in_one_dir": str(aio_dir) if aio_dir else "",
        "accumulation_dir": str(acc_dir),
        "distribution_dir": str(dist_dir),
        "all_in_one_file_count": len(aio_rows),
        "all_in_one_acc_files": aio_acc,
        "all_in_one_dist_files": aio_dist,
        "accumulation_file_count": len(acc_rows),
        "distribution_file_count": len(dist_rows),
        "accumulation_side_files": acc_side_files,
        "distribution_side_files": dist_side_files,
        "files": aio_rows + acc_rows + dist_rows,
        "has_true_accumulation": has_acc,
        "uses_all_in_one": use_aio,
        "message": _build_message(
            len(aio_rows), aio_acc, aio_dist, len(acc_rows), len(dist_rows), acc_side_files, dist_side_files, use_aio
        ),
    }


def _build_message(
    aio_files: int,
    aio_acc: int,
    aio_dist: int,
    acc_files: int,
    dist_files: int,
    acc_side: int,
    dist_side: int,
    use_aio: bool,
) -> str:
    if use_aio and aio_acc > 0 and aio_dist > 0:
        return (
            f"**All in one Data** has {aio_files} files ({aio_acc} accumulation + {aio_dist} distribution). "
            "Pipeline uses this folder first — run **Run Pipeline** to refresh features & predictions."
        )
    if use_aio and aio_acc > 0:
        return f"All in one Data: {aio_acc} accumulation file(s), {aio_dist} distribution. Pipeline loads from All in one Data."
    if use_aio:
        return (
            f"All in one Data: {aio_dist} distribution file(s) only. Add accumulation exports to the same folder for full EMS."
        )
    if acc_side > 0:
        return f"Accumulation data in legacy folders ({acc_side} file(s)). Floorsheet + EMS use acc + dist."
    if acc_files == 0 and dist_files > 0:
        return (
            f"Only distribution CSVs in **Distribution Data** ({dist_files} files). "
            "Put acc+dist exports in **Data/All in one Data/** or add files to **Accumulation Data/**."
        )
    if acc_files == 0 and dist_files == 0 and aio_files == 0:
        return "No CSV/Excel in Data folders. Add floorsheet exports to **Data/All in one Data/**."
    return "No accumulation column headers found in scanned folders."


def panel_side_summary(panel: pd.DataFrame) -> dict:
    if panel.empty or "side" not in panel.columns:
        return {"distribution_rows": 0, "accumulation_rows": 0}
    vc = panel["side"].astype(str).str.lower().value_counts()
    return {
        "distribution_rows": int(vc.get("distribution", 0)),
        "accumulation_rows": int(vc.get("accumulation", 0)),
    }
