"""Logic graph — nodes and edges linking quant, financial, broker, agents, LLM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import PROCESSED_DIR


def _json_default(obj: object):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _sanitize_meta(meta: dict) -> dict:
    out = {}
    for k, v in meta.items():
        try:
            json.dumps(v, default=_json_default)
            out[k] = v
        except TypeError:
            out[k] = str(v)
    return out

GRAPH_PATH = PROCESSED_DIR / "logic_graph.json"


class LogicGraphStore:
    """
    Knowledge graph: Symbol ↔ domains ↔ agents ↔ metrics ↔ brokers ↔ signals.
    """

    def __init__(self):
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self.meta: dict = {}
        if GRAPH_PATH.exists():
            try:
                data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
                self.nodes = data.get("nodes", [])
                self.edges = data.get("edges", [])
                self.meta = data.get("meta", {})
            except json.JSONDecodeError:
                pass

    def _nid(self, kind: str, key: str) -> str:
        return f"{kind}:{key}"

    def _upsert_node(self, nid: str, kind: str, label: str, meta: dict) -> None:
        for n in self.nodes:
            if n["id"] == nid:
                n["meta"] = {**n.get("meta", {}), **meta}
                n["label"] = label
                return
        self.nodes.append({"id": nid, "kind": kind, "label": label, "meta": _sanitize_meta(meta)})

    def _add_edge(
        self,
        src: str,
        dst: str,
        rel: str,
        weight: float = 0.5,
        rationale: str = "",
    ) -> None:
        self.edges.append(
            {
                "source": src,
                "target": dst,
                "relation": rel,
                "weight": round(float(weight), 3),
                "rationale": rationale[:300] if rationale else "",
            }
        )

    def add_symbol_analysis(self, sym: str, quant_steps: list[dict], tier: str) -> None:
        sym = sym.upper()
        sid = self._nid("symbol", sym)
        self._upsert_node(sid, "symbol", sym, {"tier": tier})

        for step in quant_steps:
            step_name = step.get("step", "step")
            pid = self._nid("pipeline", f"{sym}:{step_name}")
            self._upsert_node(pid, "pipeline_step", step_name, {"score": step.get("score"), "pass": step.get("pass")})
            self._add_edge(sid, pid, "has_step", weight=0.9)

        tier_node = self._nid("signal", tier)
        self._upsert_node(tier_node, "signal_tier", tier, {})
        self._add_edge(sid, tier_node, "classified_as", weight=1.0)

    def add_broker_flow_edges(self, sym: str, broker_table: list[dict]) -> None:
        sym = sym.upper()
        sid = self._nid("symbol", sym)
        self._upsert_node(sid, "symbol", sym, {})
        for row in broker_table[:20]:
            bid = str(row.get("broker_id", ""))
            if not bid or bid == "—":
                continue
            act = float(row.get("activity_qty") or 0)
            if act <= 0 and float(row.get("buy_qty") or 0) + float(row.get("sell_qty") or 0) <= 0:
                continue
            bid_node = self._nid("broker", f"{sym}:{bid}")
            self._upsert_node(bid_node, "broker", f"Broker {bid}", {k: v for k, v in row.items() if k != "broker_id"})
            self._add_edge(
                sid,
                bid_node,
                str(row.get("bias", "flow")),
                weight=(float(row.get("conviction_score") or 30)) / 100,
                rationale=str(row.get("flow_label", "")),
            )

    def prune_symbol(self, sym: str) -> int:
        """Remove one symbol's nodes/edges so rebuild does not leak other tickers."""
        sym = str(sym).strip().upper()
        sid = self._nid("symbol", sym)
        remove: set[str] = set()

        for n in self.nodes:
            nid = n["id"]
            if nid == sid:
                remove.add(nid)
                continue
            if n.get("kind") == "symbol" and nid != sid:
                continue
            meta = n.get("meta") or {}
            if meta.get("symbol") == sym or f":{sym}" in nid or nid.endswith(f":{sym}"):
                remove.add(nid)
            if n.get("kind") == "agent" and nid.startswith("agent:") and nid.count(":") == 1:
                remove.add(nid)

        before = len(self.nodes)
        self.nodes = [n for n in self.nodes if n["id"] not in remove]
        self.edges = [e for e in self.edges if e["source"] not in remove and e["target"] not in remove]
        return before - len(self.nodes)

    def save(self) -> None:
        GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.meta["node_count"] = len(self.nodes)
        self.meta["edge_count"] = len(self.edges)
        GRAPH_PATH.write_text(
            json.dumps(
                {"meta": self.meta, "nodes": self.nodes[-12000:], "edges": self.edges[-30000:]},
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )

    def subgraph_symbol(self, sym: str) -> dict:
        from backend.knowledge.comprehensive_graph import subgraph_for_symbol

        return subgraph_for_symbol(sym, depth=2)

    def node_by_id(self, nid: str) -> dict | None:
        for n in self.nodes:
            if n["id"] == nid:
                return n
        return None
