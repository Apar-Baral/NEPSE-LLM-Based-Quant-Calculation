"""Run before Streamlit — exits 0 only if scanner imports and runs."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main() -> int:
    print("Python:", sys.executable)
    print("ROOT:", ROOT)

    try:
        from backend.signals.universe_tiers import assign_universe_tiers
        print("universe_tiers.assign_universe_tiers: OK")
    except ImportError as e:
        print("FAIL:", e)
        return 1

    try:
        from backend.scanner.volume_universe import get_latest_scanner_universe
        from backend.db.store import DataStore

        store = DataStore()
        preds = store.load_predictions()
        panel = store.load_panel()
        if preds.empty:
            print("WARN: no predictions — run pipeline first")
            return 0
        df = get_latest_scanner_universe(preds, panel=panel, top_n=10)
        print("scanner rows:", len(df))
        if not df.empty:
            print(df[["symbol", "signal_tier", "daily_turnover_lac", "broker_pressure"]].head(5))
            print("tiers:", df["signal_tier"].value_counts().to_dict())
        print("SCANNER TEST: PASSED")
        return 0
    except Exception as e:
        import traceback
        print("FAIL:", e)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
