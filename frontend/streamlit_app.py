"""NEPSE Quant Platform — Streamlit Dashboard"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_scanner_helpers():
    """Load scanner functions without relying on package __init__ (Streamlit-safe)."""
    import importlib.util

    vu_path = ROOT / "backend" / "scanner" / "volume_universe.py"
    try:
        from backend.scanner.volume_universe import (
            get_latest_scanner_universe as _g,
            symbol_horizon_snapshot as _s,
        )

        return _g, _s
    except ImportError:
        spec = importlib.util.spec_from_file_location("nepse_volume_universe", vu_path)
        mod = importlib.util.module_from_spec(spec)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load scanner module from {vu_path}") from None
        spec.loader.exec_module(mod)
        return mod.get_latest_scanner_universe, mod.symbol_horizon_snapshot


get_latest_scanner_universe, symbol_horizon_snapshot = _load_scanner_helpers()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from backend.backtest.engine import run_backtest
from backend.config import ensure_dirs
from backend.db.store import DataStore
from backend.llm.analyst import chat_query, generate_daily_brief, generate_symbol_report, llm_status, test_llm_connection
from backend.models.trainer import compute_shap_values
from backend.alerts.webhooks import check_trigger_alerts
from backend.ingest.panel_utils import snapshot_panel_all_horizons
from backend.pipeline import run_pipeline
from backend.scanner.broker_insights import horizon_net_flow
from frontend.display_config import TIER_COLORS, TIER_HELP, TIER_ORDER, format_scanner_table

st.set_page_config(page_title="NEPSE Quant", page_icon="📈", layout="wide")
ensure_dirs()

PAGES = ["Momentum Scanner", "Symbol Deep Dive", "Daily Upload", "Backtest", "LLM Briefing", "Chat"]

with st.sidebar:
    st.title("NEPSE Quant")
    page = st.radio("Navigate", PAGES)
    if st.button("Run Pipeline", use_container_width=True):
        with st.spinner("Running pipeline..."):
            result = run_pipeline()
            st.session_state["last_pipeline"] = result
            st.success(f"Done: {result.get('symbols', 0)} symbols")

def _safe_float(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _zone_from_panel(sym_panel: pd.DataFrame, col: str) -> float | None:
    if sym_panel.empty or col not in sym_panel.columns:
        return None
    vals = sym_panel[col].dropna()
    vals = vals[vals.apply(lambda x: _safe_float(x) is not None)]
    if vals.empty:
        return None
    return _safe_float(vals.iloc[0])


def _col(df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
    """Resolve column, including merge suffixes (_x/_y)."""
    if name in df.columns:
        return df[name]
    for suffix in ("_x", "_y"):
        if f"{name}{suffix}" in df.columns:
            return df[f"{name}{suffix}"]
    return pd.Series(default, index=df.index)


store = DataStore()
preds = store.load_predictions()
features = store.load_features()
panel = snapshot_panel_all_horizons(store.load_panel())
ohlcv = store.load_ohlcv()


def _tier_color_scale():
    return {t: TIER_COLORS.get(t, "#888") for t in TIER_ORDER}

if page == "Momentum Scanner":
    st.header("Momentum Scanner — Top 120 High Volume")
    if preds.empty:
        st.info("No predictions yet. Click **Run Pipeline** in the sidebar.")
    else:
        try:
            df = get_latest_scanner_universe(preds, panel=panel, top_n=120)
        except Exception as exc:
            st.error(f"Scanner error: {exc}")
            st.stop()
        if df.empty:
            st.warning("No symbols in high-volume universe. Run **Run Pipeline** from the sidebar.")
            st.stop()
        latest = df["report_date"].max() if not df.empty else preds["report_date"].max()
        st.caption(
            f"Report date: **{latest.date()}** | Universe: **{len(df)}** symbols | "
            f"Ranked by **1D turnover (Lac)** then LLM + quant early-momentum score"
        )
        if st.button("Refresh LLM scores (top 60)", help="Requires DeepSeek/OpenAI in .env"):
            with st.spinner("LLM scoring..."):
                from backend.scanner.llm_scorer import score_universe_with_llm

                df = score_universe_with_llm(df, panel, fetch_new=True)
                from backend.signals.momentum_rules import assign_universe_tiers

                df["signal_tier"] = assign_universe_tiers(df)
                if "llm_p_long" in df.columns:
                    df["p_long_momentum"] = df["llm_p_long"].fillna(df["p_long_momentum"])
            st.success("LLM scores updated (cached).")

        tiers_present = [t for t in TIER_ORDER if t in df["signal_tier"].unique()]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trigger+", int(df[df["signal_tier"].isin(["Trigger", "Confirmed"])].shape[0]))
        c2.metric("Setup", int(df[df["signal_tier"] == "Setup"].shape[0]))
        c3.metric("Watch", int(df[df["signal_tier"] == "Watch"].shape[0]))
        top_turn = df.sort_values("daily_turnover_lac", ascending=False).iloc[0] if "daily_turnover_lac" in df.columns and len(df) else df.iloc[0] if len(df) else None
        c4.metric("Top 1D Turnover", top_turn["symbol"] if top_turn is not None else "—")
        c5.metric("Top Early Pick", df.iloc[0]["symbol"] if len(df) else "—")

        with st.expander("Signal tier guide", expanded=False):
            for t in tiers_present:
                st.markdown(f"**{t}** — {TIER_HELP.get(t, '')}")

        default_tiers = [t for t in ("Trigger", "Confirmed", "Setup", "Watch") if t in tiers_present]
        if not default_tiers:
            default_tiers = tiers_present
        tier_filter = st.multiselect(
            "Filter by signal",
            tiers_present,
            default=default_tiers,
            help="Neutral = no edge; Invalidated = heavy distribution risk",
        )
        df = df[df["signal_tier"].isin(tier_filter)]

        display_cols = [
            "volume_rank", "symbol", "ltp", "daily_volume", "daily_turnover_lac",
            "early_rank_score", "signal_tier", "p_long_momentum", "expected_return_10d",
            "early_momentum_score", "broker_pressure", "distribution_risk_score",
            "float_turnover_1d_abs", "llm_p_long", "llm_note",
        ]
        for canonical in ("early_momentum_score", "distribution_risk_score", "ltp", "early_rank_score", "broker_pressure"):
            if canonical not in df.columns:
                df[canonical] = _col(df, canonical)
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(
            format_scanner_table(df[display_cols].sort_values("early_rank_score", ascending=False)),
            use_container_width=True,
            hide_index=True,
        )

        tab1, tab2, tab3 = st.tabs(["Setup map", "Broker pressure (1D–1W)", "Tier breakdown"])
        plot_df = df.copy()
        plot_df["early_rank_pct"] = (_col(plot_df, "early_rank_score") * 100).round(1)
        plot_df["broker_pressure"] = _col(plot_df, "broker_pressure").round(1)
        p_long = _col(plot_df, "p_long_momentum")
        if "llm_p_long" in plot_df.columns:
            p_long = plot_df["llm_p_long"].fillna(p_long)
        plot_df["p_long_pct"] = (pd.to_numeric(p_long, errors="coerce").fillna(0) * 100).round(1)

        with tab1:
            st.markdown("**X** = LLM/quant P(long) · **Y** = composite early rank · size = 1D turnover")
            fig_map = px.scatter(
                plot_df,
                x="p_long_pct",
                y="early_rank_pct",
                color="signal_tier",
                color_discrete_map=_tier_color_scale(),
                hover_name="symbol",
                size="daily_turnover_lac" if "daily_turnover_lac" in plot_df.columns else None,
                labels={
                    "p_long_pct": "P(long 10D) %",
                    "early_rank_pct": "Early rank %",
                    "signal_tier": "Signal",
                    "daily_turnover_lac": "1D turnover (Lac)",
                },
                title="Actionable setups — probability vs early rank",
            )
            fig_map.update_layout(legend_title_text="Signal")
            st.plotly_chart(fig_map, use_container_width=True)

        with tab2:
            top_b = plot_df[plot_df["broker_pressure"] > 0].nlargest(25, "broker_pressure")
            if top_b.empty:
                st.warning("Broker pressure is zero — click **Run Pipeline** then **Refresh LLM scores**.")
            else:
                fig_b = px.bar(
                    top_b.sort_values("broker_pressure"),
                    x="broker_pressure",
                    y="symbol",
                    orientation="h",
                    color="signal_tier",
                    color_discrete_map=_tier_color_scale(),
                    labels={"broker_pressure": "Broker pressure (0–100)", "symbol": "Symbol"},
                    title="Short-horizon broker buy/sell skew (1D–1W)",
                )
                fig_b.update_xaxes(range=[0, 100])
                st.plotly_chart(fig_b, use_container_width=True)

        with tab3:
            tier_counts = plot_df["signal_tier"].value_counts().reindex(tiers_present).dropna()
            fig_t = px.bar(
                x=tier_counts.index,
                y=tier_counts.values,
                color=tier_counts.index,
                color_discrete_map=_tier_color_scale(),
                labels={"x": "Signal", "y": "Count"},
                title="Signals in filtered universe",
            )
            fig_t.update_layout(showlegend=False)
            st.plotly_chart(fig_t, use_container_width=True)

        if st.button("Send Trigger Alerts"):
            sent = check_trigger_alerts(df)
            st.write(f"Alerts sent for: {sent or 'none (configure webhook)'}")

elif page == "Symbol Deep Dive":
    st.header("Symbol Deep Dive")
    if preds.empty:
        st.info("Run pipeline first.")
    else:
        universe = get_latest_scanner_universe(preds, panel=panel, top_n=120)
        sym_options = universe["symbol"].tolist() if not universe.empty else sorted(preds["symbol"].unique())
        sym = st.selectbox("Symbol (top 120 high volume)", sym_options, index=0)

        sym_pred = preds[preds["symbol"] == sym].tail(1)
        if sym_pred.empty:
            sym_pred = universe[universe["symbol"] == sym].tail(1) if not universe.empty else sym_pred
        sym_feat = features[features["symbol"] == sym].sort_values("report_date")
        sym_panel = panel[panel["symbol"] == sym]

        row = sym_pred.iloc[0] if not sym_pred.empty else None

        if row is not None:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Tier", row.get("signal_tier", "N/A"))
            m2.metric("P(Long)", f"{float(row.get('p_long_momentum') or 0):.0%}")
            ems = row.get("early_momentum_score") or row.get("early_momentum_score_x") or 0
            m3.metric("EMS", f"{float(ems):.0f}")
            m4.metric("Early Rank", f"{float(row.get('early_rank_score') or 0):.0%}")
            ltp = row.get("ltp")
            m5.metric("LTP", f"{ltp:.2f}" if _safe_float(ltp) else "N/A")
            if "daily_volume" in row.index or (hasattr(row, "get") and row.get("daily_volume") is not None):
                vol = float(row.get("daily_volume") or 0)
                if vol > 0:
                    st.caption(f"Daily volume: **{vol:,.0f}** | Turnover: **{float(row.get('daily_turnover_lac') or 0):,.2f} Lac.**")

        if not sym_panel.empty:
            acc_flow = horizon_net_flow(sym_panel, "accumulation")
            dist_flow = horizon_net_flow(sym_panel, "distribution")

            st.subheader("Floorsheet horizon ladder")
            col1, col2 = st.columns(2)
            with col1:
                if not acc_flow.empty:
                    fig_acc = px.bar(
                        acc_flow,
                        x="horizon",
                        y="net_lac",
                        color="power",
                        labels={"net_lac": "Net amount (Lac)", "horizon": "Horizon", "power": "Power"},
                        title=f"{sym} — Accumulation net flow",
                    )
                    fig_acc.add_hline(y=0, line_width=1, line_color="gray")
                    st.plotly_chart(fig_acc, use_container_width=True)
                else:
                    st.info("Upload **Accumulation** Excel for acc ladder + higher EMS scores.")
            with col2:
                if not dist_flow.empty:
                    fig_dist = px.bar(
                        dist_flow,
                        x="horizon",
                        y="net_lac",
                        color="power",
                        labels={"net_lac": "Net amount (Lac)", "horizon": "Horizon", "power": "Power"},
                        title=f"{sym} — Distribution net flow",
                    )
                    fig_dist.add_hline(y=0, line_width=1, line_color="gray")
                    st.plotly_chart(fig_dist, use_container_width=True)

            if not dist_flow.empty and "buy_qty_sum" in dist_flow.columns:
                flow = dist_flow.copy()
                buy = pd.to_numeric(flow["buy_qty_sum"], errors="coerce").fillna(0)
                sell = pd.to_numeric(flow["sell_qty_sum"], errors="coerce").fillna(0)
                flow_melt = pd.DataFrame(
                    {
                        "horizon": list(flow["horizon"]) * 2,
                        "side": ["Buy qty", "Sell qty"] * len(flow),
                        "qty": list(buy) + list(sell),
                    }
                )
                fig_broker = px.bar(
                    flow_melt,
                    x="horizon",
                    y="qty",
                    color="side",
                    barmode="group",
                    category_orders={"horizon": ["1D", "2D", "3D", "4D", "1W"]},
                    labels={"qty": "Quantity (1D–1W)", "horizon": "Horizon"},
                    title=f"{sym} — Short-horizon broker buy vs sell qty",
                )
                st.plotly_chart(fig_broker, use_container_width=True)

        if not sym_feat.empty:
            r = sym_feat.iloc[-1]
            ltp_val = _safe_float(r.get("ltp")) or _safe_float(row.get("ltp") if row is not None else None)
            demand = _safe_float(r.get("tech_demand_zone")) or _zone_from_panel(sym_panel, "tech_demand_zone")
            supply = _safe_float(r.get("tech_supply_zone")) or _zone_from_panel(sym_panel, "tech_supply_zone")

            if ltp_val is not None:
                zones = go.Figure()
                if demand is not None:
                    zones.add_hline(y=demand, line_dash="dash", line_color="green", annotation_text="Demand")
                if supply is not None:
                    zones.add_hline(y=supply, line_dash="dash", line_color="red", annotation_text="Supply")
                zones.add_trace(go.Scatter(x=[sym_feat["report_date"].iloc[-1]], y=[ltp_val], mode="markers", name="LTP"))
                zones.update_layout(title=f"{sym} LTP vs Tech Zones", yaxis_title="Price")
                st.plotly_chart(zones, use_container_width=True)
            elif demand is None and supply is None:
                st.caption("Tech supply/demand zones not available for this symbol.")

        shap = compute_shap_values(sym_feat, sym) if not sym_feat.empty else {}
        if shap:
            shap_df = pd.DataFrame({"feature": list(shap.keys()), "shap": list(shap.values())})
            shap_df = shap_df.reindex(shap_df["shap"].abs().sort_values(ascending=False).index).head(15)
            fig_shap = px.bar(shap_df, x="shap", y="feature", orientation="h", title="SHAP Feature Impact")
            st.plotly_chart(fig_shap, use_container_width=True)

        prob = float(row.get("p_long_momentum", 0) or 0) if row is not None else 0.0
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "P(Long Momentum 10D)"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2ecc71"}},
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if row is not None and st.button(f"Generate LLM Report for {sym}"):
            with st.spinner("Analyzing..."):
                report = generate_symbol_report(row)
            st.markdown(report)

elif page == "Daily Upload":
    st.header("Daily Upload")
    st.markdown("Upload **Accumulation** and **Distribution** Excel workbooks (multi-sheet). Optional OHLCV CSV.")
    rd = st.date_input("Report date", value=date.today())
    acc_file = st.file_uploader("Accumulation Excel", type=["xlsx", "xls"])
    dist_file = st.file_uploader("Distribution Excel", type=["xlsx", "xls"])
    ohlcv_file = st.file_uploader("OHLCV CSV", type=["csv"])

    if st.button("Ingest & Analyze"):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            acc_path = dist_path = ohlcv_path = None
            if acc_file:
                acc_path = td / acc_file.name
                acc_path.write_bytes(acc_file.read())
            if dist_file:
                dist_path = td / dist_file.name
                dist_path.write_bytes(dist_file.read())
            if ohlcv_file:
                ohlcv_path = td / ohlcv_file.name
                ohlcv_path.write_bytes(ohlcv_file.read())
            with st.spinner("Processing..."):
                result = run_pipeline(report_date=rd, acc_path=acc_path, dist_path=dist_path, ohlcv_path=ohlcv_path)
            st.json(result)

elif page == "Backtest":
    st.header("Backtest — Trigger Tier")
    st.markdown(
        """
        **How it works:** Takes symbols from your saved predictions with tier **Trigger** or **Confirmed**,
        enters at the **report date close** (LTP proxy from floorsheet if no OHLCV CSV),
        exits after **N hold days**, and reports win rate / average return.
        Requires historical signal dates + OHLCV — with a single-day upload, trade count may be **0**.
        Upload daily Excel over time or add OHLCV CSV for meaningful backtests.
        """
    )
    tier = st.selectbox("Entry tier", ["Trigger", "Confirmed", "Setup"])
    hold = st.slider("Hold days", 3, 30, 10)
    if st.button("Run Backtest") or not ohlcv.empty:
        result = run_backtest(preds, ohlcv, entry_tier=tier, hold_days=hold)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", result["trades"])
        c2.metric("Win Rate", f"{result['win_rate']:.0%}")
        c3.metric("Avg Return", f"{result['avg_return']:.2f}%")
        c4.metric("CAGR Proxy", f"{result['cagr_proxy']:.1%}")
        if result["details"]:
            st.dataframe(pd.DataFrame(result["details"]), use_container_width=True)

elif page == "LLM Briefing":
    st.header("LLM Daily Brief")
    status = llm_status()
    st.caption(f"Provider: **{status['provider']}** | Model: **{status.get('model')}** | Ready: **{status['ready']}**")
    if status.get("hint"):
        st.info(status["hint"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Test LLM Connection"):
            with st.spinner("Testing..."):
                test = test_llm_connection()
            st.write("OK:", test.get("ok"))
            st.markdown(test.get("response", ""))
    with c2:
        if st.button("Generate Brief (Top 120 Vol)"):
            universe = get_latest_scanner_universe(preds, panel=panel, top_n=120)
            with st.spinner("Generating..."):
                brief = generate_daily_brief(universe)
            st.session_state["llm_brief"] = brief
    if st.session_state.get("llm_brief"):
        st.markdown(st.session_state["llm_brief"])

elif page == "Chat":
    st.header("Quant Chat")
    q = st.text_input("Ask about momentum setups")
    if q and st.button("Send"):
        universe = get_latest_scanner_universe(preds, panel=panel, top_n=120) if not preds.empty else preds
        with st.spinner("Thinking..."):
            ans = chat_query(q, universe)
        st.markdown(ans)
