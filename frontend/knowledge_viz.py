"""Comprehensive dynamic knowledge graph — agents, quant, financial, broker, LLM associations."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.config import DB_PATH, PROCESSED_DIR
from backend.knowledge.graph_store import GRAPH_PATH, LogicGraphStore
from backend.knowledge.vector_rag import VECTOR_PATH, VectorLogicRAG
from backend.llm.analyst import llm_status
from frontend.ui_theme import PLOTLY_ZOOM_CONFIG

NODE_COLORS = {
    "symbol": "#58a6ff",
    "domain": "#bc8cff",
    "agent": "#3fb950",
    "broker": "#f0883e",
    "metric": "#79c0ff",
    "pipeline_step": "#56d364",
    "quant_verdict": "#d2a8ff",
    "signal_tier": "#ff7b72",
    "pattern": "#ffa657",
    "default": "#8b949e",
}

NODE_SIZES = {
    "symbol": 28,
    "domain": 22,
    "agent": 12,
    "broker": 14,
    "metric": 11,
    "pipeline_step": 14,
    "default": 10,
}


def storage_status() -> dict:
    chroma = PROCESSED_DIR / "chroma"
    items = [
        ("SQLite DB", DB_PATH, DB_PATH.exists()),
        ("Features parquet", PROCESSED_DIR / "features.parquet", (PROCESSED_DIR / "features.parquet").exists()),
        ("Predictions parquet", PROCESSED_DIR / "predictions.parquet", (PROCESSED_DIR / "predictions.parquet").exists()),
        ("Logic graph JSON", GRAPH_PATH, GRAPH_PATH.exists()),
        ("Vector fallback JSON", VECTOR_PATH, VECTOR_PATH.exists()),
        ("ChromaDB vectors", chroma, chroma.exists()),
    ]
    rows = []
    for name, path, ok in items:
        size = path.stat().st_size if ok and path.is_file() else 0
        rows.append({"Store": name, "Path": str(path), "Saved": "Yes" if ok else "No", "Size": f"{size / 1024:.1f} KB" if size else "—"})
    core_items = [(n, p, ok) for n, p, ok in items if "Chroma" not in n]
    all_core = all(ok for _, _, ok in core_items) if core_items else False
    return {"rows": rows, "all_core": all_core}


def _hover_text(node: dict) -> str:
    meta = node.get("meta") or {}
    lines = [f"<b>{node.get('label', node['id'])}</b>", f"Type: {node.get('kind', '')}", f"ID: {node['id']}"]
    for k, v in list(meta.items())[:12]:
        lines.append(f"{k}: {v}")
    return "<br>".join(lines)


def build_comprehensive_figure(sub: dict, title: str, filter_kinds: list[str] | None = None) -> go.Figure | None:
    nodes = sub.get("nodes", [])
    edges = sub.get("edges", [])
    if not nodes:
        return None

    try:
        import networkx as nx
    except ImportError:
        st.error("Install networkx: pip install networkx")
        return None

    if filter_kinds:
        nodes = [n for n in nodes if n.get("kind") in filter_kinds or n.get("kind") == "symbol"]
        nids = {n["id"] for n in nodes}
        edges = [e for e in edges if e["source"] in nids and e["target"] in nids]

    g = nx.Graph()
    for n in nodes:
        g.add_node(n["id"], **n)
    for e in edges:
        w = float(e.get("weight", 0.5))
        g.add_edge(e["source"], e["target"], weight=max(w, 0.1), relation=e.get("relation", ""))

    pos = nx.spring_layout(g, seed=42, k=2.2 / max(len(g.nodes) ** 0.5, 1), iterations=80)

    fig = go.Figure()
    # Edges with relation coloring
    for e in edges:
        if e["source"] not in pos or e["target"] not in pos:
            continue
        x0, y0 = pos[e["source"]]
        x1, y1 = pos[e["target"]]
        rel = e.get("relation", "")
        color = "#3fb950" if rel in ("supports", "confirms", "narrative_supports", "supports_long") else (
            "#f85149" if rel in ("contradicts", "warns", "threatens", "pressured_by", "invalidates_reliability") else "#484f58"
        )
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=1 + float(e.get("weight", 0.5)) * 2, color=color),
                hoverinfo="text",
                hovertext=f"{rel}: {e.get('rationale', '')[:100]}",
                showlegend=False,
            )
        )

    # Nodes by kind (legend) — labels as axis annotations so text scales when zooming
    kinds_seen = set()
    label_annotations: list[dict] = []
    for n in nodes:
        kind = n.get("kind", "default")
        if kind in kinds_seen:
            continue
        kinds_seen.add(kind)
        subset = [x for x in nodes if x.get("kind") == kind]
        xs, ys, hovers, sizes, colors = [], [], [], [], []
        for node in subset:
            nid = node["id"]
            if nid not in pos:
                continue
            x, y = pos[nid]
            xs.append(x)
            ys.append(y)
            hovers.append(_hover_text(node))
            sizes.append(NODE_SIZES.get(kind, 10) + 2)
            colors.append(NODE_COLORS.get(kind, NODE_COLORS["default"]))
            label = str(node.get("label", nid.split(":")[-1]))[:22]
            label_annotations.append(
                dict(
                    x=x,
                    y=y,
                    text=label,
                    showarrow=False,
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="bottom",
                    yshift=10,
                    font=dict(size=11, color="#e6edf3"),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=kind.replace("_", " ").title(),
                marker=dict(size=sizes, color=colors, line=dict(width=1, color="#fff")),
                hovertext=hovers,
                hoverinfo="text",
            )
        )

    if label_annotations:
        fig.update_layout(annotations=label_annotations)

    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=640,
        dragmode="pan",
    )
    return fig


def _render_conclusion_panel(sym: str, sub: dict) -> None:
    """Fleet + quant synthesized verdict shown above the graph."""
    concl = sub.get("conclusion") or {}
    if not concl:
        st.info(
            f"No stored conclusion for **{sym}** yet. Click **Rebuild comprehensive graph** to run "
            "162 agents and generate the summary verdict."
        )
        return

    st.markdown("### Conclusion (quant + agents + LLM rules)")
    st.markdown(concl.get("summary", ""))
    st.markdown(f"**Recommended action:** {concl.get('action', '—')}")

    c1, c2, c3, c4 = st.columns(4)
    dom_s = concl.get("domain_signals") or {}
    dom_sc = concl.get("domain_scores") or {}
    c1.metric("Quant desk", f"{dom_sc.get('quant', 0):.0f}", dom_s.get("quant", "—"))
    c2.metric("Financial", f"{dom_sc.get('financial', 0):.0f}", dom_s.get("financial", "—"))
    c3.metric("Broker", f"{dom_sc.get('broker', 0):.0f}", dom_s.get("broker", "—"))
    c4.metric("LLM rules", f"{dom_sc.get('llm', 0):.0f}", dom_s.get("llm", "—"))

    col_d, col_r = st.columns(2)
    with col_d:
        st.markdown("**Key drivers**")
        for d in concl.get("drivers") or []:
            st.markdown(f"- {d}")
    with col_r:
        st.markdown("**Risks**")
        for r in concl.get("risks") or []:
            st.markdown(f"- {r}")
    if concl.get("quant_verdict"):
        st.caption(f"Quant pipeline verdict: *{concl['quant_verdict']}* · Tier: **{concl.get('tier', '—')}**")


def render_node_inspector(sub: dict) -> None:
    nodes = sub.get("nodes", [])
    if not nodes:
        return
    options = {f"{n.get('kind')}: {n.get('label', n['id'])[:40]}": n["id"] for n in nodes}
    pick = st.selectbox("Inspect node", list(options.keys()))
    nid = options[pick]
    node = next((n for n in nodes if n["id"] == nid), None)
    if not node:
        return
    st.json(node)
    related = [e for e in sub.get("edges", []) if e["source"] == nid or e["target"] == nid]
    if related:
        st.dataframe(pd.DataFrame(related), hide_index=True, use_container_width=True)


def _render_kg_guide() -> None:
    with st.expander("What this graph means (findings & how to act)", expanded=False):
        st.markdown(
            """
**One symbol only (not other stocks)**  
The graph should show **only the symbol you selected** (e.g. NGPL). If you see BUNGAL or AKJCL, that was a **bug**: old shared agent nodes linked multiple tickers. Click **Rebuild comprehensive graph** for your symbol to refresh. After the fix, agents are stored as `agent:name:SYMBOL` so other stocks cannot appear.

**What you are looking at**  
A single-stock **map of evidence**: quant math, broker desks, financial risk, and LLM-style rules, all linked to one ticker.

| Node type | Meaning | How to read it |
|-----------|---------|----------------|
| **Symbol** | Your company | Center of the story — tier, turnover, EMS, P(long) in hover |
| **Domain** (QUANT / FINANCIAL / BROKER / LLM) | Specialist “desks” | Score + bullish/bearish/neutral — which side of the house agrees |
| **Agent** | One automated check (162 total) | Green link → supports long · Red → warns · Skip = no data that day |
| **Metric** | Hard numbers (EMS, dist risk, turnover) | Linked to the domain that owns that metric |
| **Broker** | A broker desk **with real trades on this symbol** | Not the whole market top-10 — only desks that traded this name |
| **Pipeline step** | Volumetric / broker / price action / momentum | Pass/fail and score from the 4-step quant pipeline |
| **Signal tier** | Setup / Trigger / Watch / … | Final scanner classification |

**Edge colors**  
- **Green** — supports long (absorption, confirms, bullish agent)  
- **Red** — conflict (distribution pressure, bearish agent, circular risk)  
- **Gray** — neutral association  

**Your position (how to decide)**  
1. **Broker + QUANT agree (green)** and financial dist risk not extreme → *consider long / add on confirmation*.  
2. **Red edges dominate** or tier **Invalidated** → *avoid new longs; exit or reduce if already in*.  
3. **Mixed graph** (Setup tier, split agents) → *hold / watch — wait for turnover + tier upgrade*.  

**What to do with findings**  
- Use the graph **before** LLM chat: it shows *why* a score exists.  
- Rebuild after new CSV upload or when switching symbol.  
- Cross-check **Momentum Scanner** turnover (Lac) — graph does not replace liquidity filters.
            """
        )


def render_knowledge_graph_page(symbol: str | None = None) -> None:
    st.markdown("### Comprehensive knowledge graph")
    st.caption(
        "Dynamic graph from **162 agents** + quant pipeline + financial metrics + broker desks + cross-domain associations. "
        "Green edges = support · Red = conflict · Deploy agents on a symbol to rebuild."
    )
    _render_kg_guide()

    stat = storage_status()
    with st.expander("Storage & vector DB", expanded=False):
        st.dataframe(pd.DataFrame(stat["rows"]), hide_index=True, use_container_width=True)

    graph = LogicGraphStore()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total nodes", len(graph.nodes))
    c2.metric("Total edges", len(graph.edges))
    kinds = pd.Series([n.get("kind") for n in graph.nodes]).value_counts().head(5)
    c3.metric("Top node types", kinds.index[0] if len(kinds) else "—")
    rag = VectorLogicRAG()
    c4.metric("Vector docs", len(rag._fallback))

    from backend.scanner.symbol_lookup import all_tracked_symbols
    from frontend.symbol_search import symbol_picker

    store_syms: list[str] = []
    try:
        from backend.db.store import DataStore

        s = DataStore()
        store_syms = all_tracked_symbols(s.load_predictions(), s.load_features(), s.load_broker_panel())
    except Exception:
        pass

    default_sym = (symbol or "").strip().upper()
    if store_syms:
        sym = symbol_picker(
            "Graph symbol",
            store_syms,
            key_prefix="kg",
            default=default_sym or store_syms[0],
            help_text="Search by ticker or full company name. Rebuild graph after changing symbol.",
        )
    else:
        sym = (symbol or st.text_input("Focus symbol", value=default_sym or "NGPL", key="kg_sym")).strip().upper()
    depth = st.slider("Graph depth (hops from symbol)", 1, 3, 2)
    filter_dom = st.multiselect(
        "Show node types",
        ["domain", "agent", "metric", "broker", "pipeline_step", "pattern", "signal_tier", "quant_verdict"],
        default=["domain", "agent", "metric", "broker", "pipeline_step"],
    )

    use_llm_rebuild = st.checkbox("Add LLM association edges on rebuild", value=llm_status().get("ready", False), key="kg_llm_rebuild")
    if st.button("Rebuild comprehensive graph", type="primary"):
        from backend.scanner.symbol_lookup import enrich_symbol_row
        from backend.db.store import DataStore
        from backend.agents.fleet import deploy_agent_fleet
        from backend.knowledge.comprehensive_graph import build_comprehensive_knowledge
        from frontend.data_access import load_panel_safe

        progress = st.progress(0.0, text="Starting…")
        status = st.empty()

        def _step(pct: float, msg: str) -> None:
            progress.progress(min(max(pct, 0.02), 0.98), text=msg)
            status.caption(msg)

        try:
            _step(0.05, "Loading data…")
            store = DataStore()
            preds = store.load_predictions()
            panel = load_panel_safe(store, repair=True)
            bp = store.load_broker_panel()
            feat = store.load_features()
            from backend.scanner.volume_universe import get_latest_scanner_universe

            _step(0.25, "Building scanner context…")
            universe = get_latest_scanner_universe(preds, panel=panel, broker_panel=bp, top_n=0, features=feat)
            row_df = enrich_symbol_row(sym, preds, panel, bp, features=feat, universe_df=universe)
            if row_df.empty:
                st.error("No symbol data — run pipeline")
            else:
                sym_panel = panel[panel["symbol"].astype(str).str.upper() == sym]
                _step(0.45, f"Running {sym} agent fleet (parallel)…")
                report = deploy_agent_fleet(sym, row_df.iloc[0], sym_panel, bp, features=feat)
                _step(0.75, "Writing knowledge graph + vectors…")
                build_comprehensive_knowledge(
                    sym, row_df.iloc[0], report, report.quant_pipeline, use_llm_associations=use_llm_rebuild
                )
                progress.progress(1.0, text="Done")
                status.empty()
                from backend.knowledge.comprehensive_graph import build_graph_conclusion

                concl = build_graph_conclusion(sym, row_df.iloc[0], report, report.quant_pipeline)
                st.session_state[f"kg_conclusion_{sym}"] = concl
                st.success(f"Graph built: **{report.agent_count}** agents · composite **{report.composite_score:.0f}/100**")
                st.balloons()
                st.rerun()
        except Exception as exc:
            progress.empty()
            status.empty()
            st.error(f"Graph build failed: {exc}")

    if sym:
        from backend.knowledge.comprehensive_graph import subgraph_for_symbol

        sub = subgraph_for_symbol(sym, depth=depth)
    else:
        sub = {"nodes": [], "edges": []}
    other_symbols = sorted(
        {
            n.get("label", n["id"].split(":")[-1])
            for n in sub.get("nodes", [])
            if n.get("kind") == "symbol" and n["id"] != f"symbol:{sym}"
        }
    )
    if other_symbols:
        st.warning(
            f"Stale graph data still lists other tickers: **{', '.join(other_symbols[:6])}**. "
            f"Click **Rebuild comprehensive graph** for **{sym}** only."
        )

    st.caption(
        f"**{sym} only** — {len(sub.get('nodes', []))} nodes · {len(sub.get('edges', []))} edges (depth {depth}). "
        "Rebuild after changing symbol."
    )

    if sub.get("conclusion"):
        _render_conclusion_panel(sym, sub)
    elif st.session_state.get(f"kg_conclusion_{sym}"):
        _render_conclusion_panel(sym, {"conclusion": st.session_state[f"kg_conclusion_{sym}"]})

    fig = build_comprehensive_figure(sub, f"{sym} — quant · financial · broker · LLM", filter_kinds=filter_dom or None)
    if fig:
        st.caption("Scroll to zoom · drag to pan · double-click to reset · use toolbar for box zoom.")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_ZOOM_CONFIG)
    elif sym:
        st.info(f"Deploy **{sym}** agent fleet in Symbol Deep Dive → Brokers to build the comprehensive graph.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Associations (edges)**")
        if sub.get("edges"):
            edf = pd.DataFrame(sub["edges"])
            st.dataframe(
                edf[["relation", "weight", "source", "target", "rationale"]].head(80),
                hide_index=True,
                use_container_width=True,
                height=320,
            )
    with col_b:
        render_node_inspector(sub)

    st.markdown("**Vector memory search** (comprehensive agent + quant docs)")
    q = st.text_input("Query", value=f"{sym} comprehensive broker quant financial")
    if st.button("Search Chroma / fallback"):
        for i, h in enumerate(rag.query(q, n=8), 1):
            st.markdown(f"**{i}.** {h.get('text', '')[:500]}")
            if h.get("metadata"):
                st.caption(json.dumps(h.get("metadata")))
