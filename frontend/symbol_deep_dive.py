"""Symbol deep dive — multi-step quant + LLM verification."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backend.llm.analyst import generate_symbol_report, llm_status
from backend.quant.engine import run_quant_analysis
from backend.agents.orchestrator import run_analysis_swarm
from backend.knowledge.graph_store import LogicGraphStore
from backend.scanner.broker_desk import circular_detail
from backend.scanner.broker_top10 import discover_top_brokers, symbol_top_brokers_table
from backend.scanner.broker_insights import horizon_net_flow
from backend.scanner.symbol_lookup import enrich_symbol_row
from backend.models.trainer import compute_shap_values
from frontend.display_config import TIER_COLORS, TIER_HELP


def _col(df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
    if name in df.columns:
        return df[name]
    for suffix in ("_x", "_y"):
        if f"{name}{suffix}" in df.columns:
            return df[f"{name}{suffix}"]
    return pd.Series(default, index=df.index)


def _momentum_radar(row: pd.Series, quant: dict) -> go.Figure:
    p = float(quant.get("p_long_display", row.get("p_long_momentum") or 0))
    ems = float(quant.get("ems_display", row.get("early_momentum_score") or 0))
    dims = {
        "P(Long)": p * 100,
        "LLM": (float(row.get("llm_p_long") or p) * 100),
        "EMS": ems,
        "Broker": float(row.get("broker_pressure") or 0),
        "Volume": min(100, float(row.get("daily_turnover_lac") or 0) / 3),
        "Floorsheet": float(row.get("floorsheet_momentum_score") or 0),
    }
    fig = go.Figure(
        data=go.Scatterpolar(
            r=list(dims.values()),
            theta=list(dims.keys()),
            fill="toself",
            line_color="#58a6ff",
        )
    )
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), title="Momentum radar", height=360)
    return fig


def _horizon_heatmap(sym_panel: pd.DataFrame, sym: str) -> go.Figure | None:
    if sym_panel.empty:
        return None
    rows = []
    for side in ("accumulation", "distribution"):
        sub = sym_panel[sym_panel["side"] == side] if "side" in sym_panel.columns else sym_panel
        for _, r in sub.iterrows():
            rows.append(
                {
                    "horizon": r.get("horizon", "?"),
                    "side": side[:3],
                    "net_lac": float(pd.to_numeric(r.get("net_amount_sum", 0), errors="coerce") or 0),
                }
            )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="side", columns="horizon", values="net_lac", aggfunc="sum").fillna(0)
    order = ["1D", "2D", "3D", "4D", "1W", "1M", "3M"]
    cols = [c for c in order if c in pivot.columns] + [c for c in pivot.columns if c not in order]
    return px.imshow(
        pivot[cols],
        title=f"{sym} — Acc/Dist heatmap",
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )


def _render_quant_steps(quant: dict) -> None:
    st.markdown(f"### {quant['verdict']} — **{quant['composite_score']}/100** ({quant['steps_passed']}/{quant['steps_total']} steps pass)")
    for step in quant["steps"]:
        icon = "✅" if step.get("pass") else "❌"
        with st.expander(f"{icon} {step['step']} — score {step['score']}/100", expanded=step["step"] in ("Quant momentum", "LLM verification")):
            for n in step.get("notes", []):
                st.markdown(f"- {n}")
            if step.get("order_block_bias"):
                st.caption(f"Order block: {step['order_block_bias']} | FVG bull={step.get('bullish_fvg')} bear={step.get('bearish_fvg')}")


def render_symbol_deep_dive(
    sym: str,
    preds: pd.DataFrame,
    features: pd.DataFrame,
    panel: pd.DataFrame,
    broker_panel: pd.DataFrame,
    universe_df: pd.DataFrame | None = None,
    run_llm_verify: bool = False,
) -> None:
    sym = str(sym).strip().upper()
    enriched = enrich_symbol_row(sym, preds, panel, broker_panel, features=features)
    if enriched.empty:
        st.error(f"No data for **{sym}**. Run pipeline or check symbol.")
        return

    row = enriched.iloc[0]
    sym_panel = panel[panel["symbol"].astype(str).str.upper() == sym]
    sym_feat = features[features["symbol"].astype(str).str.upper() == sym].sort_values("report_date") if not features.empty else pd.DataFrame()

    quant = run_quant_analysis(sym, row, sym_panel, broker_panel, universe_df, run_llm=run_llm_verify)
    conf_score = quant["composite_score"]

    tier = row.get("signal_tier", "N/A")
    tier_color = TIER_COLORS.get(str(tier), "#888")
    p_show = float(quant.get("p_long_display", row.get("p_long_momentum") or 0))
    ems_show = float(quant.get("ems_display", row.get("early_momentum_score") or 0))

    st.markdown(
        f'<div style="border-left:4px solid {tier_color};padding-left:12px">'
        f"<h3 style='margin:0'>{sym}</h3>"
        f"<span style='color:{tier_color};font-weight:600'>{tier}</span> — {TIER_HELP.get(str(tier), '')}</div>",
        unsafe_allow_html=True,
    )

    c0, c1, c2, c3, c4, c5, c6, c7 = st.columns(8)
    c0.metric("Quant confirmation", f"{conf_score}/100")
    c1.metric("P(Long) eff.", f"{p_show:.0%}")
    c2.metric("LLM P(Long)", f"{float(row.get('llm_p_long')):.0%}" if pd.notna(row.get("llm_p_long")) else "—")
    c3.metric("1D Turnover", f"{float(row.get('daily_turnover_lac') or 0):,.0f} Lac")
    c4.metric("EMS eff.", f"{ems_show:.0f}")
    c5.metric("Broker Δ", f"{float(row.get('broker_pressure') or 0):.0f}")
    c6.metric("Floorsheet", f"{float(row.get('floorsheet_momentum_score') or 0):.0f}")
    c7.metric("LTP", f"{float(row.get('ltp')):.2f}" if pd.notna(row.get("ltp")) else "—")

    tab_q, tab_v, tab_c, tab_b, tab_l = st.tabs(
        ["Quant pipeline", "Volume", "Floorsheet & PA", "Brokers", "LLM advisor"]
    )

    with tab_q:
        _render_quant_steps(quant)
        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(_momentum_radar(row, quant), use_container_width=True)
        with col_r:
            st.plotly_chart(
                go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=conf_score,
                        title={"text": "Composite confirmation"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#2ecc71" if conf_score >= 60 else "#e67e22"},
                        },
                    )
                ),
                use_container_width=True,
            )
    with tab_v:
        vol_step = quant["steps"][0]
        for n in vol_step.get("notes", []):
            st.markdown(f"- {n}")
        if universe_df is not None and not universe_df.empty and "daily_turnover_lac" in universe_df.columns:
            top15 = universe_df.nlargest(15, "daily_turnover_lac").copy()
            top15["symbol"] = top15["symbol"].astype(str).str.upper()
            top15["highlight"] = top15["symbol"] == sym
            st.plotly_chart(
                px.bar(
                    top15,
                    x="symbol",
                    y="daily_turnover_lac",
                    color="highlight",
                    title="1D turnover vs top 15 (highlight = selected)",
                    color_discrete_map={True: "#58a6ff", False: "#444"},
                ),
                use_container_width=True,
            )

    with tab_c:
        pa_step = quant["steps"][2]
        st.markdown("#### Price action — order blocks & FVG")
        for n in pa_step.get("notes", []):
            st.markdown(f"- {n}")
        if pa_step.get("demand_zone"):
            st.caption(f"Demand OB: {pa_step['demand_zone']:.2f} | Supply: {pa_step.get('supply_zone')}")
        hm = _horizon_heatmap(sym_panel, sym)
        if hm:
            st.plotly_chart(hm, use_container_width=True)
        if not sym_panel.empty:
            dist_flow = horizon_net_flow(sym_panel, "distribution")
            if not dist_flow.empty:
                st.plotly_chart(px.bar(dist_flow, x="horizon", y="net_lac", color="power", title="Distribution ladder"), use_container_width=True)
        shap = compute_shap_values(sym_feat, sym) if not sym_feat.empty else {}
        if shap:
            shap_df = pd.DataFrame({"feature": list(shap.keys()), "shap": list(shap.values())})
            shap_df = shap_df.reindex(shap_df["shap"].abs().sort_values(ascending=False).index).head(12)
            st.plotly_chart(px.bar(shap_df, x="shap", y="feature", orientation="h"), use_container_width=True)

    with tab_b:
        if broker_panel.empty:
            st.info("Run pipeline for broker-level data.")
        else:
            top10_ids = discover_top_brokers(broker_panel, top_n=10)
            st.caption(f"**Top 10 brokers** (market 1D activity): {', '.join(top10_ids)}")
            btable = symbol_top_brokers_table(sym, broker_panel, top_n=10)
            if not btable.empty:
                st.dataframe(btable, use_container_width=True, hide_index=True)
                active = btable[btable["activity_qty"] > 0]
                if not active.empty:
                    st.plotly_chart(
                        px.bar(active, x="broker_id", y="conviction_score", color="bias",
                               title=f"{sym} — top 10 broker conviction"),
                        use_container_width=True,
                    )
            detail = circular_detail(sym, broker_panel)
            st.markdown(f"**Circular:** {detail.get('verdict')}")
            for line in detail.get("explanation", []):
                st.markdown(line)
            if st.button("Run parallel analysis agents", key=f"swarm_{sym}"):
                with st.spinner("Agents: volumetric · broker · momentum · knowledge…"):
                    st.session_state[f"swarm_{sym}"] = run_analysis_swarm(
                        sym, row, sym_panel, broker_panel, universe_df
                    )
            swarm = st.session_state.get(f"swarm_{sym}")
            if swarm:
                g = LogicGraphStore().subgraph_symbol(sym)
                st.caption(f"Logic graph: {len(g['nodes'])} nodes · {len(g['edges'])} edges")
                if g["edges"]:
                    st.dataframe(pd.DataFrame(g["edges"]), hide_index=True, use_container_width=True)

    with tab_l:
        if not llm_status().get("ready"):
            st.warning("Set DEEPSEEK_API_KEY in .env for live LLM steps.")
        q = st.text_area("Question", value=f"I hold {sym}. Should I add, hold, or exit for early momentum?", key=f"q_{sym}")
        if st.button("Get LLM advice", type="primary", key=f"adv_{sym}"):
            import importlib
            uni = universe_df if universe_df is not None and not universe_df.empty else enriched
            with st.spinner("LLM…"):
                ans = importlib.import_module("backend.llm.analyst").chat_query(q, uni, extra_rows=enriched)
            st.session_state[f"adv_{sym}"] = ans
        if st.session_state.get(f"adv_{sym}"):
            st.markdown(st.session_state[f"adv_{sym}"])
        if st.button("Run LLM verification on all quant steps", key=f"vfy_{sym}"):
            with st.spinner("Verifying…"):
                q2 = run_quant_analysis(sym, row, sym_panel, broker_panel, universe_df, run_llm=True)
            st.session_state[f"quant_llm_{sym}"] = q2
        if st.session_state.get(f"quant_llm_{sym}"):
            _render_quant_steps(st.session_state[f"quant_llm_{sym}"])
        if st.button(f"Full report for {sym}", key=f"rep_{sym}"):
            st.session_state[f"llm_report_{sym}"] = generate_symbol_report(row)
        if st.session_state.get(f"llm_report_{sym}"):
            st.markdown(st.session_state[f"llm_report_{sym}"])
