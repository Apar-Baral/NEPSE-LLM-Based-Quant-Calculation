"""Agent fleet orchestrator — deploys 100+ quant · financial · broker · LLM agents."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.agents.fleet import deploy_agent_fleet, fleet_status
from backend.knowledge.comprehensive_graph import build_comprehensive_knowledge, subgraph_for_symbol
from backend.llm.analyst import llm_status


def _index_knowledge(sym: str, fleet_report, row: pd.Series, use_llm: bool = False) -> dict:
    quant = fleet_report.quant_pipeline or {}
    use_llm = use_llm or llm_status().get("ready", False)
    build_comprehensive_knowledge(
        sym,
        row,
        fleet_report,
        quant_pipeline=quant,
        use_llm_associations=use_llm,
    )
    sub = subgraph_for_symbol(sym, depth=2)
    return {
        "agent": "knowledge",
        "graph_nodes": len(sub["nodes"]),
        "graph_edges": len(sub["edges"]),
        "llm_associations": use_llm,
        "comprehensive": True,
    }


def run_analysis_swarm(
    sym: str,
    row: pd.Series,
    panel_sym: pd.DataFrame,
    broker_panel: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    use_llm_graph: bool = False,
) -> dict[str, Any]:
    """
    Deploy full agent fleet (≥100 agents) in parallel.
    Returns legacy-compatible dict plus fleet report.
    """
    sym = str(sym).strip().upper()
    report = deploy_agent_fleet(sym, row, panel_sym, broker_panel, universe, features=features)

    knowledge = {}
    try:
        knowledge = _index_knowledge(sym, report, row, use_llm=use_llm_graph)
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
