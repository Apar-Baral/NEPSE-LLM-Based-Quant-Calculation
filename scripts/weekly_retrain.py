#!/usr/bin/env python3
"""Weekly model retrain + optional alert check."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.alerts.webhooks import check_trigger_alerts
from backend.db.store import DataStore
from backend.models.labels import build_labels
from backend.models.trainer import train_models
from backend.pipeline import run_pipeline


def main():
    run_pipeline(retrain=True)
    store = DataStore()
    features = store.load_features()
    labels = build_labels(features, store.load_ohlcv())
    if labels["long_momentum_10d"].sum() > 0:
        meta = train_models(features, labels)
        print("Retrained:", meta)
    preds = store.load_predictions()
    if not preds.empty:
        latest = preds[preds["report_date"] == preds["report_date"].max()]
        sent = check_trigger_alerts(latest)
        print("Alerts sent:", sent)


if __name__ == "__main__":
    main()
