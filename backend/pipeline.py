from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from backend.db.store import DataStore
from backend.features.engineer import build_daily_feature_matrix
from backend.features.pattern_library import build_pattern_store, enrich_with_analogs
from backend.scanner.broker_insights import attach_broker_metrics
from backend.scanner.volume_universe import attach_volume_from_panel, compute_early_rank_score, get_latest_scanner_universe
from backend.ingest.backfill import backfill_accumulation_data, backfill_distribution_data
from backend.ingest.data_inventory import data_folder_inventory, panel_side_summary
from backend.ingest.broker_loader import backfill_broker_panel_from_data, load_excel_broker_detail
from backend.ingest.panel_utils import snapshot_panel_all_horizons
from backend.ingest.excel_loader import load_excel_workbook
from backend.ingest.ohlcv_loader import build_ltp_proxy_from_panel, load_ohlcv_csv

from backend.llm.rag import SimpleRAG
from backend.models.labels import build_labels
from backend.models.trainer import predict, train_models
from backend.models.multimodal.train import train_multimodal
from backend.signals.momentum_rules import apply_momentum_rules
from backend.signals.universe_tiers import assign_universe_tiers


def run_pipeline(
    report_date: date | None = None,
    acc_path: Path | None = None,
    dist_path: Path | None = None,
    ohlcv_path: Path | None = None,
    retrain: bool = True,
) -> dict:
    store = DataStore()
    rd = report_date or date.today()
    frames = []

    if acc_path and acc_path.exists():
        acc = load_excel_workbook(acc_path, report_date=rd)
        frames.append(acc)
    if dist_path and dist_path.exists():
        dist = load_excel_workbook(dist_path, report_date=rd)
        frames.append(dist)

    if not frames:
        dist_legacy = backfill_distribution_data(report_date=rd)
        acc_legacy = backfill_accumulation_data(report_date=rd)
        if not dist_legacy.empty:
            frames.append(dist_legacy)
        if not acc_legacy.empty:
            frames.append(acc_legacy)

    if not frames:
        return {"status": "error", "message": "No data to ingest"}

    panel = pd.concat(frames, ignore_index=True)
    existing_panel = store.load_panel()
    if acc_path or dist_path:
        store.append_panel(panel)
    else:
        # Re-import from Data/ folders on every pipeline run (keeps acc+dist in sync)
        dist_legacy = backfill_distribution_data(report_date=rd)
        acc_legacy = backfill_accumulation_data(report_date=rd)
        legacy_frames = [f for f in (dist_legacy, acc_legacy) if not f.empty]
        if legacy_frames:
            store.append_panel(pd.concat(legacy_frames, ignore_index=True))
        elif existing_panel.empty and not panel.empty:
            store.save_panel(panel)
        elif existing_panel.empty:
            return {"status": "error", "message": "No CSV/Excel in Data/Accumulation Data or Data/Distribution Data"}

    full_panel = snapshot_panel_all_horizons(store.load_panel())
    inv = data_folder_inventory()

    broker_frames = []
    if acc_path and acc_path.exists():
        broker_frames.append(load_excel_broker_detail(acc_path, report_date=rd))
    if dist_path and dist_path.exists():
        broker_frames.append(load_excel_broker_detail(dist_path, report_date=rd))
    if broker_frames:
        store.append_broker_panel(pd.concat(broker_frames, ignore_index=True))
    elif store.load_broker_panel().empty:
        bp = backfill_broker_panel_from_data()
        if not bp.empty:
            store.save_broker_panel(bp)

    features = build_daily_feature_matrix(full_panel)
    store.save_features(features)

    ohlcv = store.load_ohlcv()
    if ohlcv_path and ohlcv_path.exists():
        ohlcv = load_ohlcv_csv(ohlcv_path)
        store.save_ohlcv(ohlcv)
    elif ohlcv.empty:
        ohlcv = build_ltp_proxy_from_panel(full_panel)
        if not ohlcv.empty:
            store.save_ohlcv(ohlcv)

    labels = build_labels(features, ohlcv)
    build_pattern_store(features, labels)

    meta = {}
    mm_meta = {}
    broker_panel_stored = store.load_broker_panel()
    if retrain and labels["long_momentum_10d"].sum() > 0 and len(features) > 20:
        try:
            meta = train_models(features, labels)
        except Exception as exc:
            meta = {"error": str(exc)}
        try:
            mm_meta = train_multimodal(features, labels, broker_panel_stored)
        except Exception as exc:
            mm_meta = {"error": str(exc)}

    latest_date = features["report_date"].max()
    latest_features = features[features["report_date"] == latest_date].copy()
    latest_features = attach_volume_from_panel(latest_features, full_panel)
    predictions = predict(latest_features, broker_panel=broker_panel_stored)
    predictions = enrich_with_analogs(predictions)
    signals = apply_momentum_rules(latest_features, predictions)
    signals = attach_volume_from_panel(signals, full_panel)
    signals = attach_broker_metrics(signals, full_panel)
    signals["early_rank_score"] = compute_early_rank_score(signals)
    signals = signals.sort_values("daily_turnover_lac", ascending=False)
    signals["signal_tier"] = assign_universe_tiers(signals)
    # Keep composite scores visible in UI (predictions table / deep dive)
    for col in (
        "floorsheet_momentum_score",
        "early_momentum_score",
        "distribution_risk_score",
        "smart_money_score",
        "acc_dist_ratio",
        "ofi",
        "mtf_convergence",
    ):
        if col in latest_features.columns and col not in signals.columns:
            signals[col] = latest_features[col].values
    store.save_predictions(signals)

    rag = SimpleRAG()
    rag.index_outcomes(features, labels)

    psum = panel_side_summary(full_panel)
    fs_mean = float(features["floorsheet_momentum_score"].mean()) if "floorsheet_momentum_score" in features.columns else 0.0
    return {
        "status": "ok",
        "report_date": str(latest_date.date()) if pd.notna(latest_date) else str(rd),
        "symbols": int(signals["symbol"].nunique()),
        "trigger_count": int(signals[signals["signal_tier"].isin(["Trigger", "Confirmed"])].shape[0]),
        "model_meta": meta,
        "multimodal_meta": mm_meta,
        "data_inventory": inv,
        "panel_sides": psum,
        "floorsheet_score_avg": round(fs_mean, 1),
    }
