"""Detect and repair corrupted symbol_panel parquet."""

from __future__ import annotations

import importlib
from datetime import date

import pandas as pd

from backend.db.store import DataStore

MIN_PANEL_ROWS = 500
MIN_PANEL_SYMBOLS = 50


def _load_combined_floorsheet(report_date: date | None = None) -> tuple[pd.DataFrame, dict]:
    """Lazy import avoids circular-import partial init of backend.ingest.backfill."""
    mod = importlib.import_module("backend.ingest.backfill")
    fn = getattr(mod, "backfill_combined_floorsheet", None)
    if callable(fn):
        return fn(report_date=report_date)

    # Fallback if an old .pyc / truncated module is missing the helper
    aio = getattr(mod, "backfill_all_in_one_data", None)
    if callable(aio):
        panel = aio(report_date=report_date)
        return panel, {"source": "all_in_one", "folder": "", "file_count": 0}

    raise ImportError(
        "backend.ingest.backfill is missing backfill_combined_floorsheet. "
        "Restart Python and delete backend/ingest/__pycache__ if this persists."
    )


def panel_needs_repair(panel: pd.DataFrame) -> bool:
    if panel.empty:
        return True
    if panel["symbol"].nunique() < MIN_PANEL_SYMBOLS:
        return True
    if len(panel) < MIN_PANEL_ROWS:
        return True
    if "horizon" in panel.columns and (panel["horizon"] == "unknown").all():
        return True
    return False


def ensure_symbol_panel(store: DataStore | None = None) -> tuple[pd.DataFrame, bool]:
    """
    Reload floorsheet panel from Data/ when parquet was truncated or corrupted.
    Returns (panel, repaired_flag).
    """
    store = store or DataStore()
    panel = store.load_panel()
    if not panel_needs_repair(panel):
        return panel, False

    fresh, _meta = _load_combined_floorsheet()
    if fresh.empty:
        return panel, False

    store.save_panel(fresh)
    return fresh, True
