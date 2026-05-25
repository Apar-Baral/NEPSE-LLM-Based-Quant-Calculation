"""Streamlit-safe data loading (handles stale cached DataStore)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _fresh_datastore():
    for mod in list(sys.modules):
        if mod == "backend.db.store" or mod.startswith("backend.db."):
            del sys.modules[mod]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("backend.db.store").DataStore()


def load_broker_panel_safe(store=None) -> pd.DataFrame:
    store = store or _fresh_datastore()
    if hasattr(store, "load_broker_panel"):
        return store.load_broker_panel()
    if hasattr(store, "load_panel"):
        return store.load_panel("broker_panel")
    return pd.DataFrame()


def save_broker_panel_safe(store, df: pd.DataFrame) -> int:
    if hasattr(store, "save_broker_panel"):
        return store.save_broker_panel(df)
    if hasattr(store, "save_panel"):
        return store.save_panel(df, "broker_panel")
    return 0


def load_panel_safe(store=None, *, repair: bool = True) -> pd.DataFrame:
    """Load symbol_panel; auto-repair if parquet was overwritten (e.g. write test)."""
    store = store or _fresh_datastore()
    panel = store.load_panel()
    if not repair:
        return panel
    from backend.ingest.panel_health import ensure_symbol_panel

    panel, _repaired = ensure_symbol_panel(store)
    return panel
