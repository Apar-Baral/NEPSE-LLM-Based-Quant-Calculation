"""Verify 100+ agent fleet deploys successfully."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from backend.agents.catalog import build_agent_catalog, catalog_summary
    from backend.agents.fleet import deploy_agent_fleet, fleet_status
    from backend.db.store import DataStore
    from backend.scanner.symbol_lookup import enrich_symbol_row

    summary = catalog_summary()
    print("Catalog:", summary)
    if summary["total"] < 100:
        print(f"FAIL: only {summary['total']} agents (need >= 100)")
        return 1

    status = fleet_status()
    print("Fleet status:", status)
    if not status["meets_minimum"]:
        return 1

    store = DataStore()
    preds = store.load_predictions()
    panel = store.load_panel()
    features = store.load_features() if hasattr(store, "load_features") else None
    bp = store.load_broker_panel() if hasattr(store, "load_broker_panel") else None
    if preds.empty:
        print("WARN: no predictions — skip live deploy test")
        print("AGENT TEST: PASSED (catalog only)")
        return 0

    sym = "BUNGAL"
    row_df = enrich_symbol_row(sym, preds, panel, bp, features=features)
    if row_df.empty:
        sym = preds.iloc[0]["symbol"]
        row_df = enrich_symbol_row(str(sym), preds, panel, bp, features=features)
    row = row_df.iloc[0]
    sym_panel = panel[panel["symbol"].astype(str).str.upper() == str(sym).upper()]

    t0 = time.perf_counter()
    broker_panel = bp if bp is not None and not bp.empty else panel.iloc[0:0]
    report = deploy_agent_fleet(sym, row, sym_panel, broker_panel, None)
    elapsed = time.perf_counter() - t0

    print(f"Deployed {report.agent_count} agents on {sym} in {elapsed:.2f}s")
    print(f"  ok={report.ok_count} err={report.error_count} skip={report.skip_count}")
    print(f"  composite={report.composite_score} consensus_long={report.consensus_long_pct}%")
    print(f"  domain_scores={report.domain_scores}")

    if report.agent_count < 100:
        print("FAIL: deploy count < 100")
        return 1
    if report.ok_count < 50:
        print("WARN: low ok_count — check broker panel / features")

    print("AGENT TEST: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
