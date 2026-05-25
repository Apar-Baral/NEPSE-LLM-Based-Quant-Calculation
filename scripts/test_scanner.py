"""Run before Streamlit — exits 0 only if scanner works."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    print("Python:", sys.executable)
    print("ROOT:", ROOT)

    errors = []

    try:
        from backend.scanner.llm_cache import apply_cached_llm_scores
        print("llm_cache.apply_cached_llm_scores: OK")
    except ImportError as e:
        errors.append(f"llm_cache: {e}")

    try:
        from backend.signals.universe_tiers import assign_universe_tiers
        print("universe_tiers: OK")
    except ImportError as e:
        errors.append(f"universe_tiers: {e}")

    try:
        from backend.scanner.volume_universe import get_latest_scanner_universe
        from backend.db.store import DataStore

        store = DataStore()
        preds = store.load_predictions()
        panel = store.load_panel()
        features = store.load_features()
        broker_panel = store.load_broker_panel() if hasattr(store, "load_broker_panel") else store.load_panel("broker_panel")
        if preds.empty and features.empty and broker_panel.empty:
            print("WARN: no predictions — run: python scripts/run_pipeline.py")
            return 0

        df = get_latest_scanner_universe(
            preds, panel=panel, broker_panel=broker_panel, top_n=10, features=features
        )
        if df.empty:
            errors.append("scanner returned empty dataframe")
        else:
            print("scanner rows:", len(df))
            cols = [c for c in ("symbol", "turnover_rank", "early_pick_rank", "signal_tier", "daily_turnover_lac", "ltp", "broker_pressure") if c in df.columns]
            print(df[cols].head(5))
            print("tiers:", df["signal_tier"].value_counts().to_dict())
            from backend.llm.analyst import _prepare_scanner_df
            ranked = _prepare_scanner_df(df)
            ranked.nlargest(3, "early_rank_score")
            print("nlargest early_rank: OK")
    except Exception as e:
        import traceback
        errors.append(str(e))
        traceback.print_exc()

    if errors:
        print("SCANNER TEST: FAILED")
        for err in errors:
            print(" -", err)
        return 1

    print("SCANNER TEST: PASSED")
    print("Now run: streamlit run frontend/streamlit_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
