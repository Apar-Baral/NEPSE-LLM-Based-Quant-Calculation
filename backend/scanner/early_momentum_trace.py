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


def trace_early_momentum(
    sym: str,
    features: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
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
    }

    hist = pd.DataFrame()
    if not features.empty and "symbol" in features.columns:
        hist = features[features["symbol"].astype(str).str.upper() == sym].copy()
    if hist.empty and predictions is not None and not predictions.empty:
        hist = predictions[predictions["symbol"].astype(str).str.upper() == sym].copy()

    if hist.empty or "report_date" not in hist.columns:
        out["summary"] = "No dated history — run pipeline after more daily uploads."
        return out

    hist["report_date"] = pd.to_datetime(hist["report_date"]).dt.normalize()
    hist = hist.sort_values("report_date").drop_duplicates("report_date", keep="last")

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

    if lead >= 65 and len(events) >= 3:
        out["stage"] = "Active early momentum"
    elif lead >= 45 or len(events) >= 2:
        out["stage"] = "Building"
    elif len(hist) >= 2:
        out["stage"] = "Early / thin"
    else:
        out["stage"] = "Single snapshot"

    bullets = [e["event"] for e in events[-4:]]
    out["summary"] = (
        f"**{out['stage']}** (lead score {lead}/100). "
        + (f"Recent: {', '.join(bullets)}." if bullets else "No sharp jumps yet in stored history.")
    )
    return out
