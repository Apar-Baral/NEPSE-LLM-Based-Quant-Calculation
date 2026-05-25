"""Build dynamic comprehensive knowledge graph from agents, quant, financial, broker, LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from backend.knowledge.graph_store import LogicGraphStore

if TYPE_CHECKING:
    from backend.agents.base import FleetReport
from backend.knowledge.llm_associations import llm_graph_associations, rule_based_associations
from backend.knowledge.vector_rag import VectorLogicRAG


DOMAIN_HUBS = ("quant", "financial", "broker", "llm")


def _metric_node(sym: str, key: str, label: str, value: Any, domain: str) -> tuple[str, dict]:
    nid = f"metric:{key}:{sym}"
    return nid, {
        "id": nid,
        "kind": "metric",
        "label": f"{label}: {value}",
        "meta": {"key": key, "value": value, "domain": domain, "symbol": sym},
    }


def build_comprehensive_knowledge(
    sym: str,
    row: pd.Series,
    fleet_report: "FleetReport",
    quant_pipeline: dict | None = None,
    use_llm_associations: bool = False,
) -> LogicGraphStore:
    """Populate graph + vector index for one symbol (call after agent fleet)."""
    sym = str(sym).strip().upper()
    g = LogicGraphStore()
    g.prune_symbol(sym)
    sid = g._nid("symbol", sym)

    tier = str(row.get("signal_tier", "Neutral"))
    g._upsert_node(
        sid,
        "symbol",
        sym,
        {
            "tier": tier,
            "ltp": row.get("ltp"),
            "turnover_lac": row.get("daily_turnover_lac"),
            "p_long": row.get("p_long_momentum"),
            "ems": row.get("early_momentum_score"),
            "floorsheet": row.get("floorsheet_momentum_score"),
            "fleet_composite": fleet_report.composite_score,
            "consensus_long_pct": fleet_report.consensus_long_pct,
        },
    )

    # Domain hub nodes (per-symbol so graphs do not cross-link tickers)
    for dom in DOMAIN_HUBS:
        hid = g._nid("domain", f"{dom}:{sym}")
        score = fleet_report.domain_scores.get(dom, 50)
        sig = fleet_report.domain_signals.get(dom, "neutral")
        g._upsert_node(
            hid,
            "domain",
            dom.upper(),
            {"score": score, "signal": sig, "agent_count": sum(1 for a in fleet_report.agents if a.domain == dom)},
        )
        g._add_edge(sid, hid, "analyzed_by", weight=1.0, rationale=f"{dom} domain score {score}")

    # Top agents per domain (dynamic from fleet)
    for dom in DOMAIN_HUBS:
        hid = g._nid("domain", f"{dom}:{sym}")
        agents = [a for a in fleet_report.agents if a.domain == dom and a.status == "ok"]
        agents.sort(key=lambda a: a.score, reverse=True)
        for a in agents[:12]:
            aid = g._nid("agent", a.agent_id)
            g._upsert_node(
                aid,
                "agent",
                a.name[:40],
                {
                    "agent_id": a.agent_id,
                    "score": a.score,
                    "signal": a.signal,
                    "summary": a.summary,
                    "domain": dom,
                },
            )
            rel = "supports_long" if a.signal == "bullish" else ("pressures_short" if a.signal == "bearish" else "neutral_check")
            g._add_edge(hid, aid, "deploys", weight=a.score / 100, rationale=a.summary[:120])
            if a.signal == "bullish":
                g._add_edge(aid, sid, "supports", weight=a.score / 100, rationale=a.summary[:80])
            elif a.signal == "bearish":
                g._add_edge(aid, sid, "warns", weight=a.score / 100, rationale=a.summary[:80])

    # Quant pipeline steps
    if quant_pipeline:
        for step in quant_pipeline.get("steps", []):
            step_name = step.get("step", "step")
            pid = g._nid("pipeline", f"{sym}:{step_name}")
            g._upsert_node(
                pid,
                "pipeline_step",
                step_name,
                {"score": step.get("score"), "pass": step.get("pass"), "notes": step.get("notes", [])[:3]},
            )
            g._add_edge(g._nid("domain", f"quant:{sym}"), pid, "includes_step", weight=(step.get("score") or 50) / 100)
            g._add_edge(sid, pid, "has_quant_step", weight=0.9)
            verdict = quant_pipeline.get("verdict", "")
            vid = g._nid("quant_verdict", sym)
            g._upsert_node(vid, "quant_verdict", str(verdict)[:50], {"composite": quant_pipeline.get("composite_score")})
            g._add_edge(pid, vid, "contributes_to", weight=0.85)

    # Key metrics (quant + financial)
    metrics = [
        ("p_long", "P(long)", row.get("p_long_momentum"), "quant"),
        ("ems", "EMS", row.get("early_momentum_score"), "quant"),
        ("floorsheet", "Floorsheet", row.get("floorsheet_momentum_score"), "quant"),
        ("dist_risk", "Dist risk", row.get("distribution_risk_score"), "financial"),
        ("turnover", "1D Turnover", row.get("daily_turnover_lac"), "financial"),
        ("broker_pressure", "Broker pressure", row.get("broker_pressure"), "broker"),
        ("exp_ret", "Exp return 10D", row.get("expected_return_10d"), "financial"),
        ("wash", "Wash score", row.get("wash_score"), "broker"),
        ("circular", "Circular risk", row.get("circular_risk"), "broker"),
    ]
    for key, label, val, dom in metrics:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        nid, node = _metric_node(sym, key, label, val, dom)
        g._upsert_node(nid, node["kind"], node["label"], node["meta"])
        g._add_edge(g._nid("domain", f"{dom}:{sym}"), nid, "measures", weight=0.7)
        g._add_edge(sid, nid, "has_metric", weight=0.6)

    # Brokers from table
    for brow in fleet_report.broker_table[:15]:
        bid = str(brow.get("broker_id", ""))
        if not bid or bid == "—":
            continue
        bid_node = g._nid("broker", bid)
        g._upsert_node(bid_node, "broker", f"Broker {bid}", {k: v for k, v in brow.items() if k != "broker_id"})
        g._add_edge(g._nid("domain", f"broker:{sym}"), bid_node, "desk_activity", weight=(brow.get("conviction_score") or 0) / 100)
        g._add_edge(sid, bid_node, brow.get("bias", "flow"), weight=(brow.get("conviction_score") or 30) / 100, rationale=brow.get("flow_label", ""))

    # Tier + patterns
    tier_nid = g._nid("signal", f"{tier}:{sym}")
    g._upsert_node(tier_nid, "signal_tier", tier, {"source": "scanner", "symbol": sym})
    g._add_edge(sid, tier_nid, "classified_as", weight=1.0)

    if row.get("pattern_dist_shakeout") or row.get("dist_shakeout_flag"):
        pid = g._nid("pattern", f"shakeout:{sym}")
        g._upsert_node(pid, "pattern", "Dist shakeout", {"active": True})
        g._add_edge(sid, pid, "exhibits_pattern", weight=0.85)
        g._add_edge(pid, g._nid("domain", f"quant:{sym}"), "informs", weight=0.7)

    # Rule + optional LLM cross-links
    for link in rule_based_associations(
        sym, row, fleet_report.domain_scores, fleet_report.domain_signals,
        fleet_report.composite_score, fleet_report.consensus_long_pct,
    ):
        g._add_edge(
            link["source"], link["target"], link["relation"],
            weight=link.get("weight", 0.5), rationale=link.get("rationale", ""),
        )

    if use_llm_associations:
        ctx = {
            "symbol": sym,
            "row": {k: row.get(k) for k in ("signal_tier", "p_long_momentum", "early_momentum_score", "distribution_risk_score", "broker_pressure")},
            "domain_scores": fleet_report.domain_scores,
            "domain_signals": fleet_report.domain_signals,
            "composite": fleet_report.composite_score,
            "top_agents": [
                {"id": a.agent_id, "domain": a.domain, "score": a.score, "signal": a.signal, "summary": a.summary}
                for a in sorted(fleet_report.agents, key=lambda x: -x.score)[:25]
                if a.status == "ok"
            ],
        }
        for link in llm_graph_associations(sym, ctx):
            g._add_edge(
                link.get("source", ""), link.get("target", ""),
                link.get("relation", "associates"),
                weight=float(link.get("weight", 0.5)),
                rationale=str(link.get("rationale", ""))[:200],
            )

    g.save()
    _index_comprehensive_vectors(sym, row, fleet_report, quant_pipeline)
    return g


def _index_comprehensive_vectors(sym: str, row: pd.Series, fleet: "FleetReport", quant: dict | None) -> None:
    rag = VectorLogicRAG()
    sym = sym.upper()

    agent_lines = []
    for dom in DOMAIN_HUBS:
        ok_agents = [a for a in fleet.agents if a.domain == dom and a.status == "ok"]
        ok_agents.sort(key=lambda a: -a.score)
        agent_lines.append(f"\n## {dom.upper()} ({fleet.domain_scores.get(dom, 0):.0f}/100, {fleet.domain_signals.get(dom, 'neutral')})")
        for a in ok_agents[:15]:
            agent_lines.append(f"- {a.agent_id} ({a.score:.0f}): {a.signal} — {a.summary}")

    broker_lines = []
    for b in fleet.broker_table[:10]:
        broker_lines.append(
            f"Broker {b.get('broker_id')}: {b.get('bias')} buy_share={b.get('buy_share_pct')}% — {b.get('flow_label', '')}"
        )

    quant_lines = []
    if quant:
        quant_lines.append(f"Verdict: {quant.get('verdict')} composite {quant.get('composite_score')}")
        for s in quant.get("steps", []):
            quant_lines.append(f"- {s.get('step')}: {s.get('score')}/100 pass={s.get('pass')}")

    doc = f"""# {sym} comprehensive NEPSE knowledge
Tier: {row.get('signal_tier')} | LTP: {row.get('ltp')} | Turnover: {row.get('daily_turnover_lac')} Lac
P(long): {row.get('p_long_momentum')} | EMS: {row.get('early_momentum_score')} | Floorsheet: {row.get('floorsheet_momentum_score')}
Fleet: {fleet.agent_count} agents | composite {fleet.composite_score}/100 | long consensus {fleet.consensus_long_pct}%
Dist risk: {row.get('distribution_risk_score')} | Broker pressure: {row.get('broker_pressure')}

## Quant pipeline
{chr(10).join(quant_lines) or 'n/a'}

## Agent fleet
{chr(10).join(agent_lines)}

## Broker desks
{chr(10).join(broker_lines) or 'n/a'}
"""
    rag.index_logic_chain(f"{sym}:comprehensive", doc, {"symbol": sym, "type": "comprehensive", "composite": fleet.composite_score})
    rag.index_logic_chain(
        f"{sym}:associations",
        f"Cross-domain: quant={fleet.domain_signals.get('quant')} financial={fleet.domain_signals.get('financial')} "
        f"broker={fleet.domain_signals.get('broker')} llm={fleet.domain_signals.get('llm')}",
        {"symbol": sym, "type": "associations"},
    )
    rag.save_fallback()


def _blocks_symbol_hop(n: dict, sym: str) -> bool:
    """True if BFS must not traverse this node (another ticker)."""
    if n.get("kind") != "symbol":
        return False
    return n["id"] != f"symbol:{sym.upper()}"


def subgraph_for_symbol(sym: str, depth: int = 2) -> dict:
    """BFS subgraph for one symbol — never pulls in other tickers via shared hubs."""
    g = LogicGraphStore()
    sym = sym.upper()
    sid = g._nid("symbol", sym)
    if not any(n["id"] == sid for n in g.nodes):
        return {"nodes": [], "edges": [], "symbol": sym, "depth": depth}

    node_by_id = {n["id"]: n for n in g.nodes}
    seen_n = {sid}
    seen_e: set[tuple] = set()
    frontier = {sid}
    all_edges: list[dict] = []

    for _ in range(depth):
        next_frontier: set[str] = set()
        for e in g.edges:
            src, dst = e["source"], e["target"]
            if src not in frontier and dst not in frontier:
                continue
            other = dst if src in frontier else src
            other_node = node_by_id.get(other)
            if other_node and other_node.get("kind") == "symbol" and other != sid:
                continue
            key = (src, dst, e.get("relation", ""))
            if key not in seen_e:
                seen_e.add(key)
                all_edges.append(e)
            for nid in (src, dst):
                if nid in seen_n:
                    continue
                n = node_by_id.get(nid)
                if n and _blocks_symbol_hop(n, sym):
                    continue
                seen_n.add(nid)
                next_frontier.add(nid)
        frontier = next_frontier

    sid = g._nid("symbol", sym)
    nodes = [n for n in g.nodes if n["id"] in seen_n and (n.get("kind") != "symbol" or n["id"] == sid)]
    nids = {n["id"] for n in nodes}
    edges = [e for e in all_edges if e["source"] in nids and e["target"] in nids]
    return {"nodes": nodes, "edges": edges, "symbol": sym, "depth": depth}
