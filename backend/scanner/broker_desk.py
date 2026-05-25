from __future__ import annotations

import pandas as pd

from backend.config import load_yaml_config

SHORT_HORIZONS = ("1D", "2D", "3D", "4D", "1W")


def _watch_brokers() -> list[str]:
    raw = load_yaml_config("settings.yaml").get("brokers", {}).get("watch_list", [58, 49, 45, 55, 34, 6])
    return [str(int(b)) for b in raw]


def _circ_cfg() -> dict:
    return load_yaml_config("settings.yaml").get("brokers", {})


def _symbol_broker_slice(sym: str, broker_panel: pd.DataFrame, horizon: str = "1D") -> pd.DataFrame:
    sub = broker_panel[(broker_panel["symbol"] == sym) & (broker_panel["horizon"] == horizon)].copy()
    if sub.empty and horizon == "1D":
        sub = broker_panel[broker_panel["symbol"] == sym].copy()
        if "horizon" in sub.columns:
            sub = sub[sub["horizon"].isin(SHORT_HORIZONS)]
    return sub


def analyze_symbol_brokers(sym: str, broker_panel: pd.DataFrame, horizon: str = "1D") -> dict:
    """
    Circular / wash detection at SYMBOL level (not per-broker row).

    Flags only when ALL hold:
    - High wash score (low net vs total two-sided activity)
    - Enough volume to matter
    - Multiple brokers trading both sides (reciprocal churn)
    """
    cfg = _circ_cfg()
    out = {
        "top_broker_net_lac": 0.0,
        "top_broker_buy_lac": 0.0,
        "top_broker_ids": "",
        "circular_risk": 0.0,
        "circular_flag": False,
        "circular_confirmed": False,
        "wash_score": 0.0,
        "directional_pct": 100.0,
        "reciprocal_brokers": 0,
        "symbol_activity_qty": 0.0,
    }
    if broker_panel.empty:
        return out

    sub = _symbol_broker_slice(sym, broker_panel, horizon)
    if sub.empty:
        return out

    watch = set(_watch_brokers())
    sub["broker_id"] = sub["broker_id"].astype(str)
    sub["buy_qty"] = pd.to_numeric(sub.get("buy_qty", 0), errors="coerce").fillna(0)
    sub["sell_qty"] = pd.to_numeric(sub.get("sell_qty", 0), errors="coerce").fillna(0)
    sub["net_qty"] = pd.to_numeric(sub.get("net_qty", 0), errors="coerce").fillna(0)
    sub["net_amount"] = pd.to_numeric(sub.get("net_amount", 0), errors="coerce").fillna(0)
    sub["activity_qty"] = sub["buy_qty"] + sub["sell_qty"]

    total_buy = float(sub["buy_qty"].sum())
    total_sell = float(sub["sell_qty"].sum())
    total_act = total_buy + total_sell
    net_qty = abs(float(sub["net_qty"].sum()))
    net_amt = abs(float(sub["net_amount"].sum()))

    out["symbol_activity_qty"] = total_act
    if total_act <= 0:
        return out

    # Directional flow: high = genuine accumulation/distribution; low = wash
    directional_pct = min(100.0, (net_qty / total_act) * 100.0)
    out["directional_pct"] = round(directional_pct, 1)
    out["wash_score"] = round(100.0 - directional_pct, 1)

    # Reciprocal: brokers with meaningful two-sided flow (same broker buying AND selling)
    min_side_ratio = float(cfg.get("reciprocal_min_two_side_ratio", 0.30))
    rec_count = 0
    for _, grp in sub.groupby("broker_id"):
        b, s = float(grp["buy_qty"].sum()), float(grp["sell_qty"].sum())
        if b <= 0 or s <= 0:
            continue
        two_side = min(b, s) / (b + s)
        if two_side >= min_side_ratio and (b + s) >= total_act * 0.03:
            rec_count += 1
    out["reciprocal_brokers"] = rec_count

    # Composite risk (not used alone for flag — calibrated on universe below)
    rec_penalty = min(35.0, rec_count * 8.0)
    amt_penalty = 0.0
    if net_amt > 0 and total_act > 0:
        amt_dir = min(100.0, (net_amt / (total_act + 1e-9)) * 100.0)
        amt_penalty = max(0.0, 100.0 - amt_dir) * 0.25
    out["circular_risk"] = round(min(100.0, out["wash_score"] * 0.55 + rec_penalty + amt_penalty), 1)

    from backend.scanner.broker_top10 import aggregate_top_broker_scores, discover_top_brokers

    top_ids = discover_top_brokers(broker_panel, horizon)
    watch = set(top_ids) | watch
    top = sub[sub["broker_id"].isin(watch)]
    if not top.empty:
        out["top_broker_net_lac"] = float(top["net_amount"].sum())
        out["top_broker_buy_lac"] = float(top["buy_qty"].sum() - top["sell_qty"].sum())
        active = top.groupby("broker_id")["net_amount"].sum().sort_values(ascending=False)
        out["top_broker_ids"] = ",".join(f"{k}({v:.0f})" for k, v in active.head(10).items())

    agg10 = aggregate_top_broker_scores(sym, broker_panel)
    out.update(agg10)
    return out


def _apply_circular_flags(metrics: pd.DataFrame) -> pd.DataFrame:
    """Universe-calibrated flags — only top wash names with real activity."""
    cfg = _circ_cfg()
    out = metrics.copy()
    out["circular_flag"] = False
    out["circular_confirmed"] = False

    if out.empty or "wash_score" not in out.columns:
        return out

    min_act = float(cfg.get("min_activity_qty", 8000))
    min_wash = float(cfg.get("min_wash_score", 62))
    min_rec = int(cfg.get("reciprocal_broker_min", 3))
    pct = float(cfg.get("circular_flag_percentile", 0.93))

    active = out[out["symbol_activity_qty"] >= min_act].copy()
    if active.empty:
        return out

    wash_thresh = max(min_wash, float(active["wash_score"].quantile(pct)))
    # Adaptive: many two-sided brokers lowers wash bar (real circular pattern)
    adaptive_wash = out["wash_score"] >= (min_wash - out["reciprocal_brokers"] * 3).clip(lower=45)

    suspect = (
        adaptive_wash
        & (out["symbol_activity_qty"] >= min_act)
        & (out["reciprocal_brokers"] >= 2)
    )
    confirmed = (
        (out["wash_score"] >= wash_thresh)
        & (out["reciprocal_brokers"] >= min_rec)
        & (out["symbol_activity_qty"] >= min_act)
    )

    out.loc[suspect, "circular_flag"] = True
    out.loc[confirmed, "circular_confirmed"] = True
    return out


def attach_broker_desk_metrics(df: pd.DataFrame, broker_panel: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for sym in df["symbol"].unique():
        rows.append({"symbol": sym, **analyze_symbol_brokers(sym, broker_panel)})
    metrics = _apply_circular_flags(pd.DataFrame(rows))
    out = df.merge(metrics, on="symbol", how="left")
    for col in ("top_broker_net_lac", "top_broker_buy_lac", "circular_risk", "wash_score", "symbol_activity_qty"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    for col in ("circular_flag", "circular_confirmed"):
        if col in out.columns:
            out[col] = out[col].fillna(False)
    if "top_broker_ids" not in out.columns:
        out["top_broker_ids"] = ""
    else:
        out["top_broker_ids"] = out["top_broker_ids"].fillna("")
    return out


def circular_detail(sym: str, broker_panel: pd.DataFrame) -> dict:
    """Human-readable circular analysis for deep dive."""
    m = analyze_symbol_brokers(sym, broker_panel)
    if m["symbol_activity_qty"] <= 0:
        return {**m, "verdict": "No broker-level 1D data", "explanation": []}

    lines = [
        f"1D activity (buy+sell qty): **{m['symbol_activity_qty']:,.0f}**",
        f"Directional flow: **{m['directional_pct']:.1f}%** of activity (net vs churn)",
        f"Wash score: **{m['wash_score']:.1f}** (high = low net / high two-sided churn)",
        f"Reciprocal brokers (two-sided): **{m['reciprocal_brokers']}**",
    ]
    if m["circular_confirmed"]:
        verdict = "Confirmed circular / wash pattern"
    elif m["circular_flag"]:
        verdict = "Suspect — review before trusting momentum"
    else:
        verdict = "No strong circular signal"
    return {**m, "verdict": verdict, "explanation": lines}


def top_broker_market_view(broker_panel: pd.DataFrame, horizon: str = "1D", top_n: int = 10) -> pd.DataFrame:
    if broker_panel.empty:
        return pd.DataFrame()
    from backend.scanner.broker_top10 import discover_top_brokers

    sub = broker_panel[broker_panel["horizon"] == horizon].copy()
    if sub.empty:
        sub = broker_panel[broker_panel["horizon"].isin(SHORT_HORIZONS)].copy()
    top_ids = discover_top_brokers(broker_panel, horizon, top_n)
    sub = sub[sub["broker_id"].astype(str).isin(top_ids)]
    if sub.empty:
        return pd.DataFrame()

    sub["net_amount"] = pd.to_numeric(sub["net_amount"], errors="coerce").fillna(0)
    sub["activity_qty"] = pd.to_numeric(sub["activity_qty"], errors="coerce").fillna(0)
    agg = (
        sub.groupby("broker_id", as_index=False)
        .agg(
            symbols=("symbol", "nunique"),
            net_lac=("net_amount", "sum"),
            activity=("activity_qty", "sum"),
        )
        .sort_values("net_lac", ascending=False)
        .head(top_n)
    )
    return agg
