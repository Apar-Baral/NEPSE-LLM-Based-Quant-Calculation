from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from backend.db.store import DataStore
from backend.features.engineer import build_daily_feature_matrix, expand_features_with_broker
from backend.features.pattern_library import build_pattern_store, enrich_with_analogs
from backend.scanner.broker_insights import attach_broker_metrics
from backend.scanner.volume_universe import attach_volume_from_panel, compute_early_rank_score, get_latest_scanner_universe
from backend.ingest.backfill import backfill_combined_floorsheet
from backend.ingest.panel_health import ensure_symbol_panel
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

    ingest_meta: dict = {"source": "upload"}
    if not frames:
        panel, ingest_meta = backfill_combined_floorsheet(report_date=rd)
        if panel.empty:
            return {"status": "error", "message": "No data to ingest — add CSVs to Data/All in one Data/"}
        frames.append(panel)

    panel = pd.concat(frames, ignore_index=True)
    if acc_path or dist_path:
        store.append_panel(panel)
    else:
        # Full refresh from disk folders (All in one Data replaces panel when present)
        store.save_panel(panel)

    raw_panel, panel_repaired = ensure_symbol_panel(store)
    full_panel = snapshot_panel_all_horizons(raw_panel)
    inv = data_folder_inventory()

    broker_frames = []
    if acc_path and acc_path.exists():
        broker_frames.append(load_excel_broker_detail(acc_path, report_date=rd))
    if dist_path and dist_path.exists():
        broker_frames.append(load_excel_broker_detail(dist_path, report_date=rd))
    if broker_frames:
        store.append_broker_panel(pd.concat(broker_frames, ignore_index=True))
    else:
        bp = backfill_broker_panel_from_data()
        if not bp.empty:
            store.save_broker_panel(bp)

    broker_panel_stored = store.load_broker_panel()
    # Multi-day history from raw panel; latest day uses full acc+dist snapshot (richer horizons).
    hist_features = build_daily_feature_matrix(raw_panel)
    snap_features = build_daily_feature_matrix(full_panel)
    if not snap_features.empty and not hist_features.empty:
        snap_rd = snap_features["report_date"].max()
        hist_part = hist_features[hist_features["report_date"] != snap_rd]
        features = pd.concat([hist_part, snap_features], ignore_index=True)
    elif not snap_features.empty:
        features = snap_features
    else:
        features = hist_features
    latest_rd = features["report_date"].max() if not features.empty else pd.Timestamp(rd)
    features = expand_features_with_broker(features, broker_panel_stored, report_date=latest_rd)
    from backend.features.engineer import _add_composite_features

    features = _add_composite_features(features)
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
    vol_panel = full_panel if not full_panel.empty else raw_panel
    signals = attach_volume_from_panel(signals, vol_panel)
    if broker_panel_stored is not None and not broker_panel_stored.empty:
        from backend.scanner.volume_universe import day_frame_from_broker_panel

        vol_bp = day_frame_from_broker_panel(broker_panel_stored, latest_date)
        if not vol_bp.empty:
            for col in ("daily_volume", "daily_turnover_lac", "float_turnover_1d_abs"):
                if col in signals.columns:
                    signals = signals.drop(columns=[col], errors="ignore")
            signals = signals.merge(
                vol_bp[["symbol", "daily_volume", "daily_turnover_lac", "float_turnover_1d_abs", "ltp"]],
                on="symbol",
                how="left",
                suffixes=("", "_bp"),
            )
            if "ltp_bp" in signals.columns:
                signals["ltp"] = signals["ltp"].fillna(signals["ltp_bp"])
                signals.drop(columns=["ltp_bp"], inplace=True, errors="ignore")
    signals = attach_broker_metrics(signals, vol_panel)
    score_cols = [
        "floorsheet_momentum_score",
        "early_momentum_score",
        "distribution_risk_score",
        "smart_money_score",
        "acc_dist_ratio",
        "ofi",
        "mtf_convergence",
        "float_turnover_zscore",
    ]
    feat_scores = latest_features[["symbol"] + [c for c in score_cols if c in latest_features.columns]].drop_duplicates(
        "symbol"
    )
    signals = signals.drop(columns=[c for c in score_cols if c in signals.columns], errors="ignore")
    signals = signals.merge(feat_scores, on="symbol", how="left")
    from backend.config_signals import get_signal_config
    from backend.signals.effective_scores import effective_scores

    cfg = get_signal_config()

    def _eff_row(row: pd.Series) -> pd.Series:
        p, ems, _ = effective_scores(row, cfg)
        return pd.Series({"p_long_momentum": p, "early_momentum_score": ems})

    eff = signals.apply(_eff_row, axis=1)
    signals["p_long_momentum"] = eff["p_long_momentum"].values
    signals["early_momentum_score"] = eff["early_momentum_score"].values
    signals["early_rank_score"] = compute_early_rank_score(signals)
    signals = signals.sort_values("daily_turnover_lac", ascending=False)
    turn = pd.to_numeric(signals["daily_turnover_lac"], errors="coerce").fillna(0)
    signals["volume_rank"] = turn.rank(ascending=False, method="min").astype("Int64")
    signals["signal_tier"] = assign_universe_tiers(signals)
    store.save_predictions(signals)

    rag = SimpleRAG()
    rag.index_outcomes(features, labels)

    psum = panel_side_summary(store.load_panel())
    psum_snap = panel_side_summary(full_panel)
    fs_mean = float(features["floorsheet_momentum_score"].mean()) if "floorsheet_momentum_score" in features.columns else 0.0
    panel_syms = int(full_panel["symbol"].nunique()) if not full_panel.empty else 0
    broker_syms = int(broker_panel_stored["symbol"].nunique()) if not broker_panel_stored.empty else 0
    return {
        "status": "ok",
        "report_date": str(latest_date.date()) if pd.notna(latest_date) else str(rd),
        "symbols": int(signals["symbol"].nunique()),
        "panel_symbols": panel_syms,
        "broker_symbols": broker_syms,
        "feature_symbols": int(features["symbol"].nunique()) if not features.empty else 0,
        "trigger_count": int(signals[signals["signal_tier"].isin(["Trigger", "Confirmed"])].shape[0]),
        "model_meta": meta,
        "multimodal_meta": mm_meta,
        "data_inventory": inv,
        "panel_sides": psum,
        "panel_sides_snapshot": psum_snap,
        "floorsheet_score_avg": round(fs_mean, 1),
        "ingest_source": ingest_meta.get("source", "unknown"),
        "ingest_folder": ingest_meta.get("folder", ""),
        "panel_repaired": panel_repaired,
    }
