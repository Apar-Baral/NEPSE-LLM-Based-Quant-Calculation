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
from backend.quant.algorithm_specs import spec_for_step
from frontend.display_config import TIER_COLORS, TIER_HELP


def _turnover_peer_figure(sym: str, universe_df: pd.DataFrame) -> go.Figure | None:
    """Bar chart of top-15 turnover; avoids px.bar color length mismatch (120 vs 15)."""
    if universe_df.empty or "daily_turnover_lac" not in universe_df.columns:
        return None

    u = universe_df[["symbol", "daily_turnover_lac"]].copy()
    u["symbol"] = u["symbol"].astype(str).str.upper()
    u["daily_turnover_lac"] = pd.to_numeric(u["daily_turnover_lac"], errors="coerce").fillna(0)
    u = u.drop_duplicates("symbol", keep="first")

    top = u.nlargest(15, "daily_turnover_lac", keep="first").head(15).reset_index(drop=True)
    sym_u = str(sym).strip().upper()
    if sym_u not in set(top["symbol"]):
        sel = u[u["symbol"] == sym_u].head(1)
        if not sel.empty:
            top = pd.concat([sel, top], ignore_index=True).drop_duplicates("symbol").head(15).reset_index(drop=True)

    colors = ["#58a6ff" if s == sym_u else "#444444" for s in top["symbol"]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=top["symbol"].tolist(),
                y=top["daily_turnover_lac"].tolist(),
                marker_color=colors,
            )
        ]
    )
    fig.update_layout(
        title="1D turnover vs top 15 (blue = selected)",
        xaxis_title="Symbol",
        yaxis_title="Turnover (Lac)",
        height=360,
    )
    return fig


def _col(df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
    if name in df.columns:
        return df[name]
    for suffix in ("_x", "_y"):
        if f"{name}{suffix}" in df.columns:
            return df[f"{name}{suffix}"]
    return pd.Series(default, index=df.index)


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


def _render_algorithm_result(step: dict) -> None:
    """One algorithm: what it used → score → conclusions (no raw variable dump)."""
    name = step.get("step", "Step")
    spec = spec_for_step(name)
    passed = bool(step.get("pass"))
    score = int(step.get("score", 0))

    st.markdown(f"#### {spec['title']}")
    st.markdown("**Uses**")
    for item in spec["uses"]:
        st.markdown(f"- {item}")
    st.markdown(f"**Produces:** {spec['gives']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{score}/100")
    c2.metric("Pass", "Yes" if passed else "No")
    c3.metric("Signal", "Bullish" if passed and score >= 60 else ("Mixed" if score >= 45 else "Weak"))

    notes = step.get("notes") or []
    if notes:
        st.markdown("**Result**")
        for n in notes:
            st.markdown(f"- {n}")

    if step.get("order_block_bias"):
        st.caption(f"Order block: **{step['order_block_bias']}** · FVG bull={step.get('bullish_fvg')} bear={step.get('bearish_fvg')}")
    if step.get("p_long_effective") is not None:
        st.caption(
            f"Momentum output: P(long) **{float(step['p_long_effective']):.0%}** · EMS **{float(step.get('ems_effective', 0)):.0f}**"
        )
    if step.get("circular_confirmed") or step.get("circular_flag"):
        st.caption(f"Circular: **{step.get('verdict', 'checked')}**")
    if step.get("verified") is not None:
        st.caption(f"LLM verified: **{step.get('verified')}** (API: {step.get('api_used', False)})")

    st.divider()


def _render_quant_pipeline(quant: dict) -> None:
    st.markdown(f"### {quant['verdict']}")
    st.caption(
        f"Composite **{quant['composite_score']}/100** · "
        f"**{quant['steps_passed']}/{quant['steps_total']}** algorithms passed (need 3+ for high conviction)"
    )
    for step in quant["steps"]:
        _render_algorithm_result(step)


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
    enriched = enrich_symbol_row(
        sym, preds, panel, broker_panel, features=features, universe_df=universe_df
    )
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

    c0, c1, c2 = st.columns(3)
    c0.metric("Composite", f"{conf_score}/100")
    c1.metric("Verdict", quant["verdict"][:28] + ("…" if len(quant["verdict"]) > 28 else ""))
    c2.metric("Steps passed", f"{quant['steps_passed']}/{quant['steps_total']}")

    fs = float(row.get("floorsheet_momentum_score") or 0)
    ems_raw = float(row.get("early_momentum_score") or 0)
    from backend.ingest.data_inventory import data_folder_inventory

    inv = data_folder_inventory()
    if fs <= 0 or ems_raw <= 0:
        st.warning(
            f"Limited floorsheet depth — {inv.get('message', '')} "
            "Add accumulation CSVs under `Data/Accumulation Data/` and run pipeline."
        )
    elif fs > 0 and not inv.get("has_true_accumulation"):
        st.caption("Floorsheet scores use distribution-proxy until accumulation files are loaded.")

    tab_q, tab_t, tab_v, tab_c, tab_b, tab_l = st.tabs(
        ["Algorithms", "Early momentum trace", "Volume chart", "Floorsheet ladder", "Brokers", "LLM advisor"]
    )

    with tab_q:
        st.caption("Each block: **what the algorithm uses** → **score** → **what it concluded**.")
        _render_quant_pipeline(quant)

    with tab_t:
        from backend.scanner.early_momentum_trace import trace_early_momentum

        trace = trace_early_momentum(sym, features, preds)
        st.markdown(f"### {trace['stage']}")
        c1, c2 = st.columns(2)
        c1.metric("Lead score", f"{trace['lead_score']}/100")
        c2.metric("History days", len(trace["timeline"]))
        st.markdown(trace["summary"])
        if trace["events"]:
            st.markdown("**What fired (chronological)**")
            st.dataframe(pd.DataFrame(trace["events"]), hide_index=True, use_container_width=True)
        else:
            st.info("No momentum events yet — need 2+ report dates in features/predictions.")
        tl = trace["timeline"]
        if not tl.empty and len(tl) >= 2:
            plot_cols = [c for c in ("early_momentum_score", "floorsheet_momentum_score", "daily_turnover_lac", "broker_pressure") if c in tl.columns]
            if plot_cols:
                melt = tl.melt(id_vars=["report_date"], value_vars=plot_cols, var_name="metric", value_name="value")
                st.plotly_chart(
                    px.line(melt, x="report_date", y="value", color="metric", markers=True, title=f"{sym} — momentum trace"),
                    use_container_width=True,
                )
        st.caption(
            "Tracks EMS jumps, turnover surges, tier upgrades, shakeout, and broker pressure — "
            "upload more daily files to lengthen the trace."
        )

    with tab_v:
        vol_step = quant["steps"][0]
        for n in vol_step.get("notes", []):
            st.markdown(f"- {n}")
        if universe_df is not None and not universe_df.empty and "daily_turnover_lac" in universe_df.columns:
            turn = float(pd.to_numeric(row.get("daily_turnover_lac"), errors="coerce") or 0)
            u_turn = pd.to_numeric(universe_df["daily_turnover_lac"], errors="coerce").fillna(0)
            pct = float((u_turn < turn).mean() * 100) if u_turn.max() > 0 else 0.0
            st.caption(f"Turnover vs top-120 universe: **top {pct:.0f}%** by 1D Lac")
            fig_peer = _turnover_peer_figure(sym, universe_df)
            if fig_peer is not None:
                st.plotly_chart(fig_peer, use_container_width=True)

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
            st.info(
                "**Distribution floorsheet:** high `sell_qty` is normal (brokers distributing stock). "
                "**Bias** uses buy-share & accumulation — not raw net_qty sign. Upload **Accumulation** for true acc_buy desks."
            )
            top10_ids = discover_top_brokers(broker_panel, top_n=10)
            st.caption(f"**Top 10 brokers** (market 1D activity): {', '.join(top10_ids)}")
            btable = symbol_top_brokers_table(sym, broker_panel, top_n=10)
            if not btable.empty:
                show_cols = [
                    c
                    for c in (
                        "broker_id", "buy_qty", "sell_qty", "buy_share_pct", "long_pressure",
                        "net_amount_lac", "share_pct", "conviction_score", "bias", "flow_label",
                    )
                    if c in btable.columns
                ]
                st.dataframe(
                    btable[show_cols] if show_cols else btable,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "bias": st.column_config.TextColumn("Bias", help="acc_buy / absorption / dist_heavy / two_sided"),
                        "buy_share_pct": st.column_config.NumberColumn("Buy %", format="%.1f"),
                        "flow_label": st.column_config.TextColumn("Meaning"),
                    },
                )
                active = btable[btable["activity_qty"] > 0]
                if not active.empty:
                    st.plotly_chart(
                        px.bar(active, x="broker_id", y="conviction_score", color="bias",
                               title=f"{sym} — top 10 broker conviction"),
                        use_container_width=True,
                    )
                else:
                    st.caption("Top-10 market brokers listed; this symbol had no 1D qty on those desks.")
            else:
                st.info(f"No broker rows for **{sym}** on the latest report date — re-run pipeline / broker backfill.")
            detail = circular_detail(sym, broker_panel)
            st.markdown(f"**Circular:** {detail.get('verdict')}")
            for line in detail.get("explanation", []):
                st.markdown(line)
            from backend.agents.fleet import fleet_status

            fs = fleet_status()
            with st.expander("What are the agents? (plain English)", expanded=False):
                st.markdown(
                    """
**Purpose:** Each agent is one automated check (like a specialist on a trading desk).  
They run together in ~1 second and vote **bullish / bearish / neutral** with a **0–100 score**.

| Domain | What it checks |
|--------|----------------|
| **Quant** | Volume, ML P(long), RSI, Bollinger, order blocks, FVG, dist/acc power |
| **Financial** | Risk, drawdown, Sharpe proxy, tier, liquidity, forward return |
| **Broker** | Each top broker’s buy/sell on this symbol, wash/circular churn |
| **LLM** | Short narratives from your metrics (no API per row — fast rules) |

**Skip** = no data that day (e.g. broker 101 had no trade in NGPL). **OK** = scored.  
**Logic graph** below = audit trail: symbol → pipeline steps → brokers → final caution label.
                    """
                )
            st.caption(
                f"Agent fleet: **{fs['total']}** agents "
                f"(quant {fs['quant']}, financial {fs['financial']}, broker {fs['broker']}, LLM {fs['llm']})"
            )
            use_llm_graph = st.checkbox(
                "LLM association edges in knowledge graph",
                value=False,
                key=f"llm_graph_{sym}",
                help="Adds DeepSeek-derived cross-links when API key is set",
            )
            if st.button(f"Deploy {fs['total']} agents (parallel)", key=f"btn_swarm_{sym}"):
                with st.spinner(f"Running {fs['total']} agents — comprehensive knowledge graph…"):
                    st.session_state[f"swarm_result_{sym}"] = run_analysis_swarm(
                        sym, row, sym_panel, broker_panel, universe_df,
                        features=features, use_llm_graph=use_llm_graph,
                    )
            swarm = st.session_state.get(f"swarm_result_{sym}")
            if swarm:
                fc1, fc2, fc3, fc4, fc5 = st.columns(5)
                fc1.metric("Agents run", swarm.get("agent_count", "—"))
                fc2.metric("Fleet composite", f"{swarm.get('composite_score', 0):.0f}/100")
                fc3.metric("Long consensus", f"{swarm.get('consensus_long_pct', 0):.0f}%")
                fc4.metric("OK / err / skip", f"{swarm.get('ok_count', 0)}/{swarm.get('error_count', 0)}/{swarm.get('skip_count', 0)}")
                dom = swarm.get("domain_scores") or {}
                fc5.metric("Domains", f"Q{dom.get('quant', 0):.0f} F{dom.get('financial', 0):.0f} B{dom.get('broker', 0):.0f} L{dom.get('llm', 0):.0f}")

                dcols = st.columns(4)
                for i, d in enumerate(("quant", "financial", "broker", "llm")):
                    score = float(dom.get(d, 0) or 0)
                    sig = (swarm.get("domain_signals") or {}).get(d, "neutral")
                    dcols[i].metric(
                        d.title(),
                        f"{score:.0f}/100",
                        delta=str(sig),
                        delta_color="normal" if sig == "neutral" else ("off" if sig == "bullish" else "inverse"),
                    )

                flat = swarm.get("agents_flat") or []
                if flat:
                    adf = pd.DataFrame(flat)
                    domain_filter = st.selectbox("Filter agents by domain", ["all", "quant", "financial", "broker", "llm"], key=f"agent_dom_{sym}")
                    if domain_filter != "all":
                        adf = adf[adf["domain"] == domain_filter]
                    st.dataframe(
                        adf[["agent_id", "domain", "score", "signal", "status", "summary"]].head(120),
                        hide_index=True,
                        use_container_width=True,
                        height=320,
                    )

                from frontend.knowledge_viz import build_comprehensive_figure
                from backend.knowledge.comprehensive_graph import subgraph_for_symbol as _subgraph

                g = _subgraph(sym, depth=2)
                st.caption(
                    f"Comprehensive graph: **{len(g['nodes'])}** nodes · **{len(g['edges'])}** edges "
                    f"(agents, metrics, domains, brokers, LLM links)"
                )
                fig_kg = build_comprehensive_figure(g, f"{sym} — comprehensive knowledge")
                if fig_kg:
                    st.plotly_chart(fig_kg, use_container_width=True)
                elif g["edges"]:
                    st.dataframe(pd.DataFrame(g["edges"]), hide_index=True, use_container_width=True)
                st.caption("Open sidebar → **Knowledge Graph** for full interactive graph + vector search.")

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
            _render_quant_pipeline(st.session_state[f"quant_llm_{sym}"])
        if st.button(f"Full report for {sym}", key=f"rep_{sym}"):
            st.session_state[f"llm_report_{sym}"] = generate_symbol_report(row)
        if st.session_state.get(f"llm_report_{sym}"):
            st.markdown(st.session_state[f"llm_report_{sym}"])
