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

    # Nodes by kind (legend)
    kinds_seen = set()
    for n in nodes:
        kind = n.get("kind", "default")
        if kind in kinds_seen:
            continue
        kinds_seen.add(kind)
        subset = [x for x in nodes if x.get("kind") == kind]
        xs, ys, texts, hovers, sizes, colors = [], [], [], [], [], []
        for node in subset:
            nid = node["id"]
            if nid not in pos:
                continue
            xs.append(pos[nid][0])
            ys.append(pos[nid][1])
            texts.append(node.get("label", nid.split(":")[-1])[:18])
            hovers.append(_hover_text(node))
            sizes.append(NODE_SIZES.get(kind, 10))
            colors.append(NODE_COLORS.get(kind, NODE_COLORS["default"]))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name=kind.replace("_", " ").title(),
                text=texts,
                textposition="top center",
                textfont=dict(size=9, color="#e6edf3"),
                marker=dict(size=sizes, color=colors, line=dict(width=1, color="#fff")),
                hovertext=hovers,
                hoverinfo="text",
            )
        )

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


def render_knowledge_graph_page(symbol: str | None = None) -> None:
    st.markdown("### Comprehensive knowledge graph")
    st.caption(
        "Dynamic graph from **162 agents** + quant pipeline + financial metrics + broker desks + cross-domain associations. "
        "Green edges = support · Red = conflict · Deploy agents on a symbol to rebuild."
    )

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

    default_sym = (symbol or "").strip().upper()
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

        store = DataStore()
        preds, panel, bp, feat = store.load_predictions(), store.load_panel(), store.load_broker_panel(), store.load_features()
        from backend.scanner.volume_universe import get_latest_scanner_universe

        universe = get_latest_scanner_universe(preds, panel=panel, broker_panel=bp, top_n=120, features=feat)
        row_df = enrich_symbol_row(sym, preds, panel, bp, features=feat, universe_df=universe)
        if row_df.empty:
            st.error("No symbol data — run pipeline")
        else:
            sym_panel = panel[panel["symbol"].astype(str).str.upper() == sym]
            report = deploy_agent_fleet(sym, row_df.iloc[0], sym_panel, bp, features=feat)
            build_comprehensive_knowledge(sym, row_df.iloc[0], report, report.quant_pipeline, use_llm_associations=use_llm_rebuild)
            st.success(f"Graph built: {report.agent_count} agents indexed")
            st.rerun()

    if sym:
        from backend.knowledge.comprehensive_graph import subgraph_for_symbol

        sub = subgraph_for_symbol(sym, depth=depth)
    else:
        sub = {"nodes": [], "edges": []}
    st.caption(f"**{sym}** subgraph: {len(sub.get('nodes', []))} nodes · {len(sub.get('edges', []))} edges (depth {depth})")

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
