from __future__ import annotations

import pandas as pd


def snapshot_panel_all_horizons(panel: pd.DataFrame) -> pd.DataFrame:
    """Merge multi-day panel rows into one snapshot (latest row per symbol/side/horizon)."""
    if panel.empty:
        return panel
    p = panel.copy()
    p["report_date"] = pd.to_datetime(p["report_date"]).dt.normalize()
    snap = p.sort_values("report_date").groupby(["symbol", "side", "horizon"], as_index=False).last()
    snap["report_date"] = p["report_date"].max()
    return snap
