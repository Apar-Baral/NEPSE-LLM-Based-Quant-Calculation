"""Agent fleet primitives — context, results, aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

AgentDomain = Literal["quant", "financial", "broker", "llm"]
AgentSignal = Literal["bullish", "bearish", "neutral", "skip"]


@dataclass
class AgentContext:
    symbol: str
    row: pd.Series
    panel_sym: pd.DataFrame
    broker_panel: pd.DataFrame
    universe: pd.DataFrame | None = None
    features_row: pd.Series | None = None
    features: pd.DataFrame | None = None


@dataclass
class AgentResult:
    agent_id: str
    domain: AgentDomain
    name: str
    status: Literal["ok", "skip", "error"] = "ok"
    score: float = 50.0
    signal: AgentSignal = "neutral"
    summary: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class FleetReport:
    symbol: str
    agent_count: int
    ok_count: int
    error_count: int
    skip_count: int
    composite_score: float
    consensus_long_pct: float
    domain_scores: dict[str, float]
    domain_signals: dict[str, AgentSignal]
    agents: list[AgentResult]
    quant_pipeline: dict | None = None
    broker_table: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "agent_count": self.agent_count,
            "ok_count": self.ok_count,
            "composite_score": self.composite_score,
            "consensus_long_pct": self.consensus_long_pct,
            "domain_scores": self.domain_scores,
            "domain_signals": self.domain_signals,
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "domain": a.domain,
                    "name": a.name,
                    "status": a.status,
                    "score": a.score,
                    "signal": a.signal,
                    "summary": a.summary,
                }
                for a in self.agents
            ],
            "quant": self.quant_pipeline,
            "broker_table": self.broker_table,
        }
