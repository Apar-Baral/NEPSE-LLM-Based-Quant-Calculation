"""LLM + rule-based association edges for the knowledge graph."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from backend.llm.analyst import llm_status


def rule_based_associations(
    sym: str,
    row: pd.Series,
    domain_scores: dict[str, float],
    domain_signals: dict[str, str],
    fleet_composite: float,
    consensus_long: float,
) -> list[dict]:
    """Cross-domain links when metrics/agents agree or conflict."""
    sym = sym.upper()
    links: list[dict] = []
    q_sig = domain_signals.get("quant", "neutral")
    f_sig = domain_signals.get("financial", "neutral")
    b_sig = domain_signals.get("broker", "neutral")
    l_sig = domain_signals.get("llm", "neutral")

    def _link(src: str, dst: str, rel: str, weight: float, rationale: str) -> None:
        links.append({"source": src, "target": dst, "relation": rel, "weight": weight, "rationale": rationale})

    sid = f"symbol:{sym}"
    qh, fh, bh, lh = f"domain:quant:{sym}", f"domain:financial:{sym}", f"domain:broker:{sym}", f"domain:llm:{sym}"

    if q_sig == f_sig == "bullish":
        _link(qh, fh, "confirms", 0.85, "Quant and financial domains aligned bullish")
    if q_sig == "bullish" and b_sig == "bearish":
        _link(bh, qh, "contradicts", 0.75, "Broker desks distribution-heavy vs quant optimism")
    if l_sig == q_sig == "bullish":
        _link(lh, qh, "narrative_supports", 0.8, "LLM narratives agree with quant")
    if fleet_composite >= 58 and consensus_long >= 45:
        _link(sid, qh, "supported_by", 0.7, f"Fleet composite {fleet_composite:.0f} with {consensus_long:.0f}% long consensus")
    if fleet_composite < 45:
        _link(sid, f"metric:fleet_weak:{sym}", "pressured_by", 0.65, "Low fleet composite — mixed/weak conviction")

    drs = float(row.get("distribution_risk_score") or 0)
    if drs >= 70:
        _link(f"metric:dist_risk:{sym}", sid, "threatens", 0.8, f"Distribution risk {drs:.0f}")
    shake = row.get("pattern_dist_shakeout") or row.get("dist_shakeout_flag")
    if shake in (True, 1, "True"):
        _link(f"pattern:shakeout:{sym}", sid, "enables_entry", 0.72, "Shakeout pattern may precede reversal")

    circ = row.get("circular_confirmed") in (True, 1)
    if circ:
        _link(bh, sid, "invalidates_reliability", 0.9, "Confirmed circular trading on broker flow")

    tier = str(row.get("signal_tier", "Neutral"))
    _link(sid, f"signal:{tier}:{sym}", "classified_as", 1.0, f"Scanner tier {tier}")

    return links


def llm_graph_associations(sym: str, context: dict[str, Any]) -> list[dict]:
    """
    Optional DeepSeek/OpenAI pass: returns extra edges as JSON list.
    Falls back to empty if API unavailable.
    """
    if not llm_status().get("ready"):
        return []

    prompt = f"""You are building a financial knowledge graph for NEPSE symbol {sym}.
Given this JSON context, output ONLY a JSON array (max 12 items) of associations:
[{{"source": "node_id", "target": "node_id", "relation": "verb", "weight": 0.0-1.0, "rationale": "short"}}]

Use node ids like symbol:{sym}, domain:quant, domain:broker, broker:58, metric:p_long:{sym}, agent:q_rsi, signal:Trigger.

Context:
{json.dumps(context, default=str)[:6000]}
"""
    try:
        from backend.llm.analyst import _call_llm

        raw = _call_llm(prompt)
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            return []
        data = json.loads(m.group(0))
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict) and "source" in x and "target" in x]
    except Exception:
        return []
    return []
