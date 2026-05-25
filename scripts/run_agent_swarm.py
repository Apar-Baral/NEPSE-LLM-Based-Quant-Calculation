"""Run parallel analysis agents for one symbol."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.agents.orchestrator import run_analysis_swarm
from backend.db.store import DataStore
from backend.ingest.panel_utils import snapshot_panel_all_horizons
from backend.scanner.symbol_lookup import enrich_symbol_row


def main() -> int:
    sym = (sys.argv[1] if len(sys.argv) > 1 else "NGPL").upper()
    store = DataStore()
    preds = store.load_predictions()
    features = store.load_features()
    panel = snapshot_panel_all_horizons(store.load_panel())
    bp = store.load_broker_panel()
    row = enrich_symbol_row(sym, preds, panel, bp, features=features)
    if row.empty:
        print("No data for", sym)
        return 1
    r = row.iloc[0]
    sp = panel[panel["symbol"].astype(str).str.upper() == sym]
    out = run_analysis_swarm(sym, r, sp, bp)
    print("Symbol:", sym)
    print("Quant composite:", out.get("quant", {}).get("composite_score"))
    print("Brokers analyzed:", len(out.get("broker_table", [])))
    print("Agents:", list(out.get("agents", {}).keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
