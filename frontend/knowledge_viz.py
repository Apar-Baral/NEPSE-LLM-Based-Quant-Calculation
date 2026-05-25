"""Interactive knowledge graph + vector DB status (MiroFish-style)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.config import DB_PATH, PROCESSED_DIR
from backend.knowledge.graph_store import GRAPH_PATH, LogicGraphStore
from backend.knowledge.vector_rag import VECTOR_PATH, VectorLogicRAG

NODE_COLORS = {
    "symbol": "#58a6ff",
    "broker": "#f0883e",
    "pipeline_step": "#3fb950",
    "signal_tier": "#d2a8ff",
    "pattern": "#ffa657",
    "default": "#8b949e",
}


def storage_status() -> dict:
    """What is persisted on disk."""
    chroma = PROCESSED_DIR / "chroma"
    items = [
        ("SQLite DB", DB_PATH, DB_PATH.exists()),
        ("Features parquet", PROCESSED_DIR / "features.parquet", (PROCESSED_DIR / "features.parquet").exists()),
        ("Predictions parquet", PROCESSED_DIR / "predictions.parquet", (PROCESSED_DIR / "predictions.parquet").exists()),
        ("Broker panel parquet", PROCESSED_DIR / "broker_panel.parquet", (PROCESSED_DIR / "broker_panel.parquet").exists()),
        ("Symbol panel parquet", PROCESSED_DIR / "symbol_panel.parquet", (PROCESSED_DIR / "symbol_panel.parquet").exists()),
        ("Logic graph JSON", GRAPH_PATH, GRAPH_PATH.exists()),
        ("Vector fallback JSON", VECTOR_PATH, VECTOR_PATH.exists()),
        ("ChromaDB vectors", chroma, chroma.exists()),
    ]
    rows = []
    for name, path, ok in items:
        size = path.stat().st_size if ok and path.is_file() else 0
        rows.append({"Store": name, "Path": str(path), "Saved": "Yes" if ok else "No", "Size": f"{size / 1024:.1f} KB" if size else "—"})
    return {"rows": rows, "all_core": ok for _, p, ok in items[:5] if "Chroma" not in str(p)}


def _build_network_figure(sub: dict, title: str) -> go.Figure | None:
    nodes = sub.get("nodes", [])
    edges = sub.get("edges", [])
    if not nodes:
        return None

    try:
        import networkx as nx
    except ImportError:
        return None

    g = nx.Graph()
    labels = {}
    colors = []
    for n in nodes:
        nid = n["id"]
        g.add_node(nid)
        labels[nid] = n.get("label", nid.split(":")[-1][:12])
        colors.append(NODE_COLORS.get(n.get("kind", "default"), NODE_COLORS["default"]))
    for e in edges:
        g.add_edge(e["source"], e["target"])

    pos = nx.spring_layout(g, seed=42, k=1.8)
    edge_x, edge_y = [], []
    for e in edges:
        x0, y0 = pos[e["source"]]
        x1, y1 = pos[e["target"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = [pos[n][0] for n in g.nodes()]
    node_y = [pos[n][1] for n in g.nodes()]
    node_text = [labels.get(n, n) for n in g.nodes()]
    node_ids = list(g.nodes())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="#484f58"),
            hoverinfo="none",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont=dict(size=10, color="#e6edf3"),
            marker=dict(size=18, color=colors, line=dict(width=1, color="#fff")),
            customdata=node_ids,
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=520,
    )
    return fig


def render_knowledge_graph_page(symbol: str | None = None) -> None:
    st.markdown("### Knowledge graph & vector memory")
    st.caption("Relationship view (like MiroFish graph) — symbols, brokers, quant steps, signals. Vectors power LLM retrieval.")

    stat = storage_status()
    st.dataframe(pd.DataFrame(stat["rows"]), hide_index=True, use_container_width=True)
    if not stat["all_core"]:
        st.warning("Some core stores missing — run **Run Pipeline** and upload daily data.")

    c1, c2, c3 = st.columns(3)
    graph = LogicGraphStore()
    rag = VectorLogicRAG()
    c1.metric("Graph nodes", len(graph.nodes))
    c2.metric("Graph edges", len(graph.edges))
    vec_n = len(rag._fallback)
    if rag._collection is not None:
        try:
            vec_n = max(vec_n, rag._collection.count())
        except Exception:
            pass
    c3.metric("Vector docs", vec_n)

    sym = (symbol or "").strip().upper()
    if not sym:
        sym = st.text_input("Focus symbol", placeholder="BUNGAL").strip().upper()

    if sym:
        sub = graph.subgraph_symbol(sym)
        fig = _build_network_figure(sub, f"{sym} — logic graph")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No graph for **{sym}** yet — open Symbol Deep Dive → Brokers → Deploy agents.")

        st.markdown("**Edges**")
        if sub.get("edges"):
            st.dataframe(pd.DataFrame(sub["edges"]), hide_index=True, use_container_width=True)
        else:
            st.caption("Deploy agent fleet once to build edges.")

    with st.expander("Full graph (sampled)", expanded=False):
        if len(graph.edges) > 400:
            edges = graph.edges[-400:]
            nids = {e["source"] for e in edges} | {e["target"] for e in edges}
            nodes = [n for n in graph.nodes if n["id"] in nids]
        else:
            edges, nodes = graph.edges, graph.nodes
        fig_all = _build_network_figure({"nodes": nodes, "edges": edges}, "NEPSE logic graph (subset)")
        if fig_all:
            st.plotly_chart(fig_all, use_container_width=True)

    st.markdown("**Vector search test**")
    q = st.text_input("Query logic memory", value=f"{sym or 'NGPL'} early momentum broker")
    if st.button("Search vectors"):
        hits = rag.query(q, n=5)
        if hits:
            for i, h in enumerate(hits, 1):
                st.markdown(f"**{i}.** {h.get('text', '')[:300]}")
        else:
            st.caption("No hits — deploy agents or run pipeline to index.")
