#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.db.store import DataStore
from backend.features.engineer import build_daily_feature_matrix
from backend.models.labels import build_labels
from backend.models.trainer import train_models

if __name__ == "__main__":
    store = DataStore()
    features = store.load_features()
    if features.empty:
        panel = store.load_panel()
        features = build_daily_feature_matrix(panel)
    labels = build_labels(features, store.load_ohlcv())
    meta = train_models(features, labels)
    print(meta)
