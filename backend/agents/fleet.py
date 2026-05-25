"""Deploy 100+ agents in parallel and aggregate consensus."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from backend.agents.base import AgentContext, AgentResult, AgentSignal, FleetReport
from backend.agents.catalog import MIN_AGENT_COUNT, build_agent_catalog, catalog_summary
from backend.config import load_yaml_config
from backend.quant.engine import run_quant_analysis
from backend.scanner.broker_top10 import discover_top_brokers, symbol_top_brokers_table


def _fleet_cfg() -> dict:
    return load_yaml_config("settings.yaml").get("agents", {})


def _run_one(defn, ctx: AgentContext) -> AgentResult:
    try:
        return defn.run(ctx)
    except Exception as exc:
        return AgentResult(
            agent_id=defn.agent_id,
            domain=defn.domain,
            name=defn.name,
            status="error",
            score=0.0,
            signal="neutral",
            summary=str(exc)[:120],
            error=str(exc),
        )


def _aggregate_domain(results: list[AgentResult], domain: str) -> tuple[float, AgentSignal]:
    sub = [r for r in results if r.domain == domain and r.status == "ok"]
    if not sub:
        return 50.0, "neutral"
    avg = sum(r.score for r in sub) / len(sub)
    bulls = sum(1 for r in sub if r.signal == "bullish")
    bears = sum(1 for r in sub if r.signal == "bearish")
    if bulls > bears * 1.4 and avg >= 55:
        sig: AgentSignal = "bullish"
    elif bears > bulls * 1.2 and avg <= 48:
        sig = "bearish"
    else:
        sig = "neutral"
    return float(round(avg, 1)), sig


def deploy_agent_fleet(
    sym: str,
    row: pd.Series,
    panel_sym: pd.DataFrame,
    broker_panel: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    max_workers: int | None = None,
) -> FleetReport:
    """Run all registered agents (100+) concurrently."""
    sym = str(sym).strip().upper()
    cfg = _fleet_cfg()
    max_workers = max_workers or int(cfg.get("max_workers", 48))

    top_brokers = discover_top_brokers(broker_panel, top_n=int(cfg.get("broker_agent_count", 50))) if not broker_panel.empty else []
    catalog = build_agent_catalog(broker_ids=top_brokers)

    ctx = AgentContext(
        symbol=sym,
        row=row,
        panel_sym=panel_sym,
        broker_panel=broker_panel,
        universe=universe,
        features=features,
    )

    results: list[AgentResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, defn, ctx): defn for defn in catalog}
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: (r.domain, r.agent_id))
    ok = [r for r in results if r.status == "ok"]
    errs = [r for r in results if r.status == "error"]
    skips = [r for r in results if r.status == "skip"]

    domain_scores = {}
    domain_signals = {}
    for dom in ("quant", "financial", "broker", "llm"):
        domain_scores[dom], domain_signals[dom] = _aggregate_domain(results, dom)

    weights = cfg.get(
        "domain_weights",
        {"quant": 0.30, "financial": 0.22, "broker": 0.28, "llm": 0.20},
    )
    composite = sum(domain_scores.get(d, 50) * float(weights.get(d, 0.25)) for d in domain_scores)

    bullish = sum(1 for r in ok if r.signal == "bullish")
    consensus = (bullish / len(ok) * 100) if ok else 0.0

    quant_pipeline = None
    broker_table: list[dict] = []
    try:
        quant_pipeline = run_quant_analysis(sym, row, panel_sym, broker_panel, universe, run_llm=False)
    except Exception:
        pass
    try:
        tbl = symbol_top_brokers_table(sym, broker_panel, top_n=10)
        if not tbl.empty:
            broker_table = tbl.to_dict(orient="records")
    except Exception:
        pass

    return FleetReport(
        symbol=sym,
        agent_count=len(catalog),
        ok_count=len(ok),
        error_count=len(errs),
        skip_count=len(skips),
        composite_score=round(composite, 1),
        consensus_long_pct=round(consensus, 1),
        domain_scores=domain_scores,
        domain_signals=domain_signals,
        agents=results,
        quant_pipeline=quant_pipeline,
        broker_table=broker_table,
    )


def fleet_status() -> dict[str, Any]:
    """Expose fleet size for UI / tests."""
    summary = catalog_summary()
    return {
        "min_required": MIN_AGENT_COUNT,
        **summary,
        "meets_minimum": summary["total"] >= MIN_AGENT_COUNT,
    }
