"""Train multimodal model on stored features + labels."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.db.store import DataStore
from backend.ingest.panel_utils import snapshot_panel_all_horizons
from backend.models.labels import build_labels
from backend.models.multimodal.train import train_multimodal


def main() -> int:
    store = DataStore()
    features = store.load_features()
    ohlcv = store.load_ohlcv()
    broker = store.load_broker_panel()
    if features.empty:
        print("No features. Run: python scripts/run_pipeline.py")
        return 1
    labels = build_labels(features, ohlcv)
    result = train_multimodal(features, labels, broker)
    print(result)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
