"""Logic graph — nodes and edges linking quant pipeline steps."""

from __future__ import annotations

import json
from pathlib import Path

from backend.config import PROCESSED_DIR

GRAPH_PATH = PROCESSED_DIR / "logic_graph.json"


class LogicGraphStore:
    """
    Lightweight knowledge graph for relationships:
    Symbol -> Broker -> Signal -> Pattern -> Horizon
    """

    def __init__(self):
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        if GRAPH_PATH.exists():
            try:
                data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
                self.nodes = data.get("nodes", [])
                self.edges = data.get("edges", [])
            except json.JSONDecodeError:
                pass

    def _nid(self, kind: str, key: str) -> str:
        return f"{kind}:{key}"

    def add_symbol_analysis(self, sym: str, quant_steps: list[dict], tier: str) -> None:
        sym = sym.upper()
        sid = self._nid("symbol", sym)
        self._upsert_node(sid, "symbol", sym, {"tier": tier})

        for step in quant_steps:
            step_name = step.get("step", "step")
            pid = self._nid("pipeline", f"{sym}:{step_name}")
            self._upsert_node(pid, "pipeline_step", step_name, {"score": step.get("score"), "pass": step.get("pass")})
            self._add_edge(sid, pid, "has_step")

            if step_name == "Broker analysis" and step.get("top_broker_ids"):
                for part in str(step["top_broker_ids"]).split(","):
                    bid = part.split("(")[0].strip()
                    if not bid:
                        continue
                    bid_node = self._nid("broker", bid)
                    self._upsert_node(bid_node, "broker", bid, {})
                    self._add_edge(sid, bid_node, "traded_by")
                    self._add_edge(bid_node, pid, "influences")

    def add_broker_flow_edges(self, sym: str, broker_table: list[dict]) -> None:
        """Link symbol to brokers with flow metadata (post fleet deploy)."""
        sym = sym.upper()
        sid = self._nid("symbol", sym)
        self._upsert_node(sid, "symbol", sym, {})
        for row in broker_table[:15]:
            bid = str(row.get("broker_id", ""))
            if not bid or bid == "—":
                continue
            bid_node = self._nid("broker", bid)
            self._upsert_node(
                bid_node,
                "broker",
                bid,
                {
                    "bias": row.get("bias"),
                    "buy_share_pct": row.get("buy_share_pct"),
                    "conviction": row.get("conviction_score"),
                },
            )
            rel = row.get("bias", "flow")
            self._add_edge(sid, bid_node, str(rel))

        tier_node = self._nid("signal", tier)
        self._upsert_node(tier_node, "signal_tier", tier, {})
        self._add_edge(sid, tier_node, "classified_as")

    def _upsert_node(self, nid: str, kind: str, label: str, meta: dict) -> None:
        for n in self.nodes:
            if n["id"] == nid:
                n["meta"] = {**n.get("meta", {}), **meta}
                return
        self.nodes.append({"id": nid, "kind": kind, "label": label, "meta": meta})

    def _add_edge(self, src: str, dst: str, rel: str) -> None:
        self.edges.append({"source": src, "target": dst, "relation": rel})

    def save(self) -> None:
        GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
        GRAPH_PATH.write_text(
            json.dumps({"nodes": self.nodes[-8000:], "edges": self.edges[-20000:]}, indent=2),
            encoding="utf-8",
        )

    def subgraph_symbol(self, sym: str) -> dict:
        sym = sym.upper()
        sid = self._nid("symbol", sym)
        related_edges = [e for e in self.edges if e["source"] == sid or e["target"] == sid]
        nids = {sid}
        for e in related_edges:
            nids.add(e["source"])
            nids.add(e["target"])
        nodes = [n for n in self.nodes if n["id"] in nids]
        return {"nodes": nodes, "edges": related_edges}
