"""Trace early-momentum onset — what fired, when, and in what order."""

from __future__ import annotations

import pandas as pd

TRACE_METRICS = (
    "early_momentum_score",
    "floorsheet_momentum_score",
    "distribution_risk_score",
    "daily_turnover_lac",
    "broker_pressure",
    "p_long_momentum",
    "early_rank_score",
)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _history_from_panel(sym: str, panel: pd.DataFrame) -> pd.DataFrame:
    """Rebuild per-upload-day metrics when features were collapsed to one snapshot."""
    if panel.empty or "symbol" not in panel.columns:
        return pd.DataFrame()
    from backend.features.engineer import build_daily_feature_matrix

    sp = panel[panel["symbol"].astype(str).str.upper() == sym].copy()
    if sp.empty or sp["report_date"].nunique() < 2:
        return pd.DataFrame()
    feat = build_daily_feature_matrix(sp)
    if feat.empty:
        return pd.DataFrame()
    cols = [c for c in TRACE_METRICS if c in feat.columns]
    extra = [c for c in ("signal_tier", "float_turnover_zscore", "pattern_dist_shakeout") if c in feat.columns]
    keep = ["report_date", "symbol"] + cols + extra
    return feat[keep].sort_values("report_date")


def trace_early_momentum(
    sym: str,
    features: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    panel: pd.DataFrame | None = None,
) -> dict:
    """
    Build a day-by-day trace and discrete events for one symbol.
    Used in Symbol Deep Dive to show *how* early momentum built up.
    """
    sym = str(sym).strip().upper()
    out: dict = {
        "symbol": sym,
        "events": [],
        "timeline": pd.DataFrame(),
        "lead_score": 0,
        "stage": "No history",
        "summary": "",
        "history_source": "features",
    }

    hist = pd.DataFrame()
    if not features.empty and "symbol" in features.columns:
        hist = features[features["symbol"].astype(str).str.upper() == sym].copy()
    if hist.empty and predictions is not None and not predictions.empty:
        hist = predictions[predictions["symbol"].astype(str).str.upper() == sym].copy()
        out["history_source"] = "predictions"

    if hist.empty or "report_date" not in hist.columns:
        out["summary"] = "No dated history — run pipeline after more daily uploads."
        return out

    hist["report_date"] = pd.to_datetime(hist["report_date"]).dt.normalize()
    hist = hist.sort_values("report_date").drop_duplicates("report_date", keep="last")

    if len(hist) < 2 and panel is not None:
        panel_hist = _history_from_panel(sym, panel)
        if len(panel_hist) >= 2:
            hist = panel_hist
            out["history_source"] = "panel_uploads"

    if hist.empty or "report_date" not in hist.columns:
        out["summary"] = "No dated history — run pipeline after more daily uploads."
        return out

    hist["report_date"] = pd.to_datetime(hist["report_date"]).dt.normalize()
    hist = hist.sort_values("report_date").drop_duplicates("report_date", keep="last")

    # Merge latest prediction columns on the last day
    if predictions is not None and not predictions.empty:
        pred = predictions[predictions["symbol"].astype(str).str.upper() == sym].copy()
        if not pred.empty:
            pred["report_date"] = pd.to_datetime(pred["report_date"]).dt.normalize()
            pred = pred.sort_values("report_date").iloc[-1]
            for c in TRACE_METRICS:
                if c in pred.index and c not in hist.columns:
                    hist[c] = 0.0
                if c in pred.index:
                    hist.loc[hist["report_date"] == hist["report_date"].max(), c] = pred[c]

    cols = [c for c in TRACE_METRICS if c in hist.columns]
    timeline = hist[["report_date", "symbol"] + cols].copy()
    out["timeline"] = timeline

    events: list[dict] = []

    def _delta(col: str, thresh: float, label: str, unit: str = "") -> None:
        if col not in hist.columns or len(hist) < 2:
            return
        s = _num(hist[col])
        d = s.diff()
        for i in range(1, len(hist)):
            if d.iloc[i] >= thresh:
                events.append(
                    {
                        "date": str(hist["report_date"].iloc[i].date()),
                        "event": label,
                        "detail": f"+{d.iloc[i]:.1f}{unit} (now {s.iloc[i]:.1f}{unit})",
                        "metric": col,
                    }
                )

    _delta("early_momentum_score", 12, "EMS jump", " pts")
    _delta("floorsheet_momentum_score", 10, "Floorsheet momentum ramp", " pts")
    _delta("daily_turnover_lac", 40, "Turnover surge", " Lac")
    _delta("broker_pressure", 8, "Broker pressure rise", " pts")
    _delta("p_long_momentum", 0.08, "P(long) step up", "")

    if "float_turnover_zscore" in hist.columns:
        _delta("float_turnover_zscore", 0.6, "Float activity spike", " z")

    if "signal_tier" in hist.columns:
        tier_rank = {"Invalidated": 0, "Neutral": 1, "Watch": 2, "Setup": 3, "Trigger": 4, "Confirmed": 5}
        ranks = hist["signal_tier"].map(tier_rank).fillna(1)
        for i in range(1, len(hist)):
            if ranks.iloc[i] > ranks.iloc[i - 1]:
                events.append(
                    {
                        "date": str(hist["report_date"].iloc[i].date()),
                        "event": "Signal tier upgrade",
                        "detail": f"{hist['signal_tier'].iloc[i-1]} → {hist['signal_tier'].iloc[i]}",
                        "metric": "signal_tier",
                    }
                )

    if "pattern_dist_shakeout" in hist.columns:
        sh = hist["pattern_dist_shakeout"].fillna(False).astype(bool)
        for i in range(len(hist)):
            if sh.iloc[i]:
                events.append(
                    {
                        "date": str(hist["report_date"].iloc[i].date()),
                        "event": "Distribution shakeout",
                        "detail": "Short-term dist exhaustion — early-long watch",
                        "metric": "pattern_dist_shakeout",
                    }
                )
                break

    events.sort(key=lambda x: x["date"])
    out["events"] = events[-20:]

    latest = hist.iloc[-1]
    ems = float(_num(pd.Series([latest.get("early_momentum_score", 0)])).iloc[0])
    fs = float(_num(pd.Series([latest.get("floorsheet_momentum_score", 0)])).iloc[0])
    turn = float(_num(pd.Series([latest.get("daily_turnover_lac", 0)])).iloc[0])
    bp = float(_num(pd.Series([latest.get("broker_pressure", 0)])).iloc[0])
    p = float(_num(pd.Series([latest.get("p_long_momentum", 0)])).iloc[0])

    lead = min(100, int(ems * 0.35 + fs * 0.25 + min(turn / 5, 25) + bp * 0.2 + p * 100 * 0.15))
    out["lead_score"] = lead

    n_days = len(hist)
    if lead >= 65 and len(events) >= 3:
        out["stage"] = "Active early momentum"
    elif lead >= 45 or len(events) >= 2:
        out["stage"] = "Building"
    elif n_days >= 2:
        out["stage"] = "Early / thin"
    else:
        out["stage"] = "Single snapshot"

    bullets = [e["event"] for e in events[-4:]]
    src_note = f" ({n_days} upload days from {out['history_source']})"
    out["summary"] = (
        f"**{out['stage']}** (lead score {lead}/100){src_note}. "
        + (f"Recent: {', '.join(bullets)}." if bullets else "No sharp jumps yet in stored history.")
    )
    return out
