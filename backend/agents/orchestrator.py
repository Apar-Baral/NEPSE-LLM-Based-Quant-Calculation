"""Parallel analysis agents — volumetric, broker, momentum, knowledge indexing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from backend.knowledge.graph_store import LogicGraphStore
from backend.knowledge.vector_rag import VectorLogicRAG
from backend.quant.engine import run_quant_analysis
from backend.scanner.broker_top10 import discover_top_brokers, symbol_top_brokers_table


def _agent_volumetric(sym: str, row: pd.Series, universe: pd.DataFrame | None) -> dict:
    from backend.quant.volumetric import analyze_volume

    return {"agent": "volumetric", "result": analyze_volume(row, universe)}


def _agent_broker(sym: str, broker_panel: pd.DataFrame) -> dict:
    table = symbol_top_brokers_table(sym, broker_panel)
    top_ids = discover_top_brokers(broker_panel)
    return {
        "agent": "broker_top10",
        "top_brokers": top_ids,
        "table_rows": len(table),
        "result": table.to_dict(orient="records") if not table.empty else [],
    }


def _agent_momentum(sym: str, row: pd.Series, panel_sym: pd.DataFrame, broker_panel: pd.DataFrame, universe) -> dict:
    return {"agent": "quant_pipeline", "result": run_quant_analysis(sym, row, panel_sym, broker_panel, universe)}


def _agent_index(sym: str, quant: dict) -> dict:
    graph = LogicGraphStore()
    graph.add_symbol_analysis(sym, quant.get("steps", []), tier=str(quant.get("verdict", "")))
    graph.save()

    rag = VectorLogicRAG()
    chain = " -> ".join(f"{s['step']}:{s['score']}" for s in quant.get("steps", []))
    rag.index_logic_chain(
        f"{sym}:latest",
        f"{sym} early momentum logic chain: {chain}. Verdict: {quant.get('verdict')}",
        {"symbol": sym, "composite": quant.get("composite_score", 0)},
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
) -> dict[str, Any]:
    """Run 4 agents in parallel; returns merged results."""
    sym = str(sym).strip().upper()
    results: dict[str, Any] = {"symbol": sym, "agents": {}}

    tasks = {
        "volumetric": lambda: _agent_volumetric(sym, row, universe),
        "broker": lambda: _agent_broker(sym, broker_panel),
        "momentum": lambda: _agent_momentum(sym, row, panel_sym, broker_panel, universe),
    }

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results["agents"][name] = fut.result()
            except Exception as exc:
                results["agents"][name] = {"error": str(exc)}

    quant = results["agents"].get("momentum", {}).get("result", {})
    if quant:
        try:
            results["agents"]["knowledge"] = _agent_index(sym, quant)
        except Exception as exc:
            results["agents"]["knowledge"] = {"error": str(exc)}

    results["quant"] = quant
    results["broker_table"] = results["agents"].get("broker", {}).get("result", [])
    return results
