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
        from backend.scanner.llm_scorer import apply_cached_llm_scores, score_universe_with_llm
        import inspect

        print("universe_tiers.assign_universe_tiers: OK")
        sig = inspect.signature(score_universe_with_llm)
        if "fetch_new" in sig.parameters:
            print("FAIL: score_universe_with_llm still has fetch_new — update llm_scorer.py")
            return 1
        print("score_universe_with_llm params:", list(sig.parameters.keys()))
        print("apply_cached_llm_scores: OK")
    except ImportError as e:
        print("FAIL import:", e)
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
        if df.empty:
            print("FAIL: empty scanner universe")
            return 1
        print(df[["symbol", "signal_tier", "daily_turnover_lac", "broker_pressure"]].head(5))
        print("tiers:", df["signal_tier"].value_counts().to_dict())

        # Streamlit code path (import app scanner block)
        import importlib.util

        spec = importlib.util.spec_from_file_location("streamlit_app", ROOT / "frontend" / "streamlit_app.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print("streamlit_app import: OK")

        print("SCANNER TEST: PASSED")
        return 0
    except Exception as e:
        import traceback
        print("FAIL:", e)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
