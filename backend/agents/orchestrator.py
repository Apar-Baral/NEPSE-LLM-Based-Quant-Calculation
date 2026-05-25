"""Agent fleet orchestrator — deploys 100+ quant · financial · broker · LLM agents."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.agents.fleet import deploy_agent_fleet, fleet_status
from backend.knowledge.graph_store import LogicGraphStore
from backend.knowledge.vector_rag import VectorLogicRAG


def _index_knowledge(sym: str, fleet_report) -> dict:
    quant = fleet_report.quant_pipeline or {}
    graph = LogicGraphStore()
    graph.add_symbol_analysis(sym, quant.get("steps", []), tier=str(quant.get("verdict", fleet_report.domain_signals.get("quant", ""))))
    if fleet_report.broker_table:
        graph.add_broker_flow_edges(sym, fleet_report.broker_table)
    graph.save()

    rag = VectorLogicRAG()
    chain = " | ".join(
        f"{a.agent_id}:{a.score}" for a in fleet_report.agents[:40] if a.status == "ok"
    )
    rag.index_logic_chain(
        f"{sym}:fleet",
        f"{sym} agent fleet ({fleet_report.agent_count} agents): {chain}. Composite {fleet_report.composite_score}.",
        {"symbol": sym, "composite": fleet_report.composite_score, "agents": fleet_report.agent_count},
    )
    rag.save_fallback()
    sub = graph.subgraph_symbol(sym)
    return {"agent": "knowledge", "graph_nodes": len(sub["nodes"]), "graph_edges": len(sub["edges"])}


def run_analysis_swarm(
    sym: str,
    row: pd.Series,
    panel_sym: pd.DataFrame,
    broker_panel: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Deploy full agent fleet (≥100 agents) in parallel.
    Returns legacy-compatible dict plus fleet report.
    """
    sym = str(sym).strip().upper()
    report = deploy_agent_fleet(sym, row, panel_sym, broker_panel, universe, features=features)

    knowledge = {}
    try:
        knowledge = _index_knowledge(sym, report)
    except Exception as exc:
        knowledge = {"error": str(exc)}

    by_domain: dict[str, list] = {"quant": [], "financial": [], "broker": [], "llm": []}
    for a in report.agents:
        by_domain.setdefault(a.domain, []).append(
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "score": a.score,
                "signal": a.signal,
                "status": a.status,
                "summary": a.summary,
            }
        )

    return {
        "symbol": sym,
        "fleet": report.to_dict(),
        "fleet_status": fleet_status(),
        "agent_count": report.agent_count,
        "composite_score": report.composite_score,
        "consensus_long_pct": report.consensus_long_pct,
        "domain_scores": report.domain_scores,
        "domain_signals": report.domain_signals,
        "agents": by_domain,
        "agents_flat": [
            {
                "agent_id": a.agent_id,
                "domain": a.domain,
                "name": a.name,
                "score": a.score,
                "signal": a.signal,
                "status": a.status,
                "summary": a.summary,
            }
            for a in report.agents
        ],
        "quant": report.quant_pipeline,
        "broker_table": report.broker_table,
        "knowledge": knowledge,
        # Legacy keys
        "ok_count": report.ok_count,
        "error_count": report.error_count,
        "skip_count": report.skip_count,
    }
