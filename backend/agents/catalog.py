"""Agent catalog — 100+ registered agents across quant, financial, broker, LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from backend.agents.base import AgentContext, AgentDomain, AgentResult
from backend.agents.executors import run_broker_agent, run_financial_agent, run_llm_agent, run_quant_agent

MIN_AGENT_COUNT = 100


@dataclass(frozen=True)
class AgentDefinition:
    agent_id: str
    domain: AgentDomain
    name: str
    run: Callable[[AgentContext], AgentResult]


QUANT_AGENT_IDS = [
    "q_turnover_abs",
    "q_turnover_pct",
    "q_daily_qty",
    "q_float_z",
    "q_float_turn_1d",
    "q_p_long_eff",
    "q_p_long_raw",
    "q_ems",
    "q_early_rank",
    "q_mtf",
    "q_dist_risk",
    "q_shakeout",
    "q_acc_dist",
    "q_ofi",
    "q_ob_demand",
    "q_ob_supply",
    "q_fvg_bull",
    "q_fvg_bear",
    "q_dist_1d",
    "q_dist_1w",
    "q_acc_1d",
    "q_acc_1w",
    "q_smart_money",
    "q_analog",
    "q_anomaly",
    "q_exp_return",
    "q_ml_conf",
    "q_vol_pipeline",
    "q_broker_pipeline",
    "q_pa_pipeline",
    "q_momentum_pipeline",
    "q_rsi",
    "q_roc_momentum",
    "q_bollinger",
    "q_turnover_zscore",
]

FINANCIAL_TECHNIQUE_IDS = [
    "f_sharpe_proxy",
    "f_max_drawdown",
    "f_herfindahl",
]

FINANCIAL_AGENT_IDS = [
    "f_ltp",
    "f_return_10d",
    "f_broker_pressure",
    "f_liquidity_turn",
    "f_liquidity_qty",
    "f_risk_drs",
    "f_risk_invalidate",
    "f_capital_eff",
    "f_dist_3y",
    "f_dist_1m",
    "f_dist_3m",
    "f_acc_1w",
    "f_acc_1m",
    "f_float_z_fin",
    "f_turnover_rank",
    "f_early_pick",
    "f_tier_bull",
    "f_llm_p_long",
    "f_price_demand",
    "f_price_supply",
    "f_anomaly_fin",
    "f_drawdown_guard",
    "f_turnover_eff",
    "f_net_1d",
    "f_hold_score",
    "f_entry_timing",
    "f_exit_risk",
    "f_portfolio_fit",
    "f_margin_safety",
    "f_stress_pass",
    "f_circular_fin",
    "f_wash_fin",
]

BROKER_META_IDS = [
    "b_circular_risk",
    "b_circular_flag",
    "b_wash",
    "b_directional",
    "b_reciprocal",
    "b_activity",
    "b_top_net",
    "b_pressure",
    "b_conviction_max",
    "b_conviction_avg",
    "b_buy_dom",
    "b_sell_dom",
    "b_churn",
    "b_net_skew",
    "b_dispersion",
    "b_watch_aggregate",
    "b_market_top10",
]

# Extra broker slots for top market brokers (58, 49, …) — filled at catalog build
BROKER_SLOT_IDS = [f"b_broker_{i:02d}" for i in range(1, 51)]

LLM_AGENT_IDS = [
    "llm_trend",
    "llm_risk",
    "llm_broker",
    "llm_volume",
    "llm_distribution",
    "llm_accumulation",
    "llm_shakeout",
    "llm_circular",
    "llm_entry",
    "llm_exit",
    "llm_tier",
    "llm_p_long",
    "llm_ems",
    "llm_analog",
    "llm_fvg",
    "llm_orderblock",
    "llm_turnover",
    "llm_watchlist",
    "llm_long_bias",
    "llm_short_bias",
    "llm_hold",
    "llm_verdict",
    "llm_momentum",
    "llm_financial",
    "llm_quant_sync",
]


def _wrap_quant(agent_id: str) -> Callable[[AgentContext], AgentResult]:
    def _run(ctx: AgentContext) -> AgentResult:
        return run_quant_agent(agent_id, ctx)

    return _run


def _wrap_fin(agent_id: str) -> Callable[[AgentContext], AgentResult]:
    def _run(ctx: AgentContext) -> AgentResult:
        return run_financial_agent(agent_id, ctx)

    return _run


def _wrap_broker(agent_id: str) -> Callable[[AgentContext], AgentResult]:
    def _run(ctx: AgentContext) -> AgentResult:
        return run_broker_agent(agent_id, ctx)

    return _run


def _wrap_llm(agent_id: str) -> Callable[[AgentContext], AgentResult]:
    def _run(ctx: AgentContext) -> AgentResult:
        return run_llm_agent(agent_id, ctx)

    return _run


def build_agent_catalog(broker_ids: list[str] | None = None) -> list[AgentDefinition]:
    """
    Build full agent fleet (≥ MIN_AGENT_COUNT).
    broker_ids: optional top-N broker IDs for per-broker agents (defaults to 58,49,45,55,34,6,38,43,70 + extras).
    """
    catalog: list[AgentDefinition] = []

    for aid in QUANT_AGENT_IDS:
        catalog.append(
            AgentDefinition(aid, "quant", aid.replace("_", " ").title(), _wrap_quant(aid))
        )

    for aid in FINANCIAL_AGENT_IDS + FINANCIAL_TECHNIQUE_IDS:
        catalog.append(
            AgentDefinition(aid, "financial", aid.replace("_", " ").title(), _wrap_fin(aid))
        )

    for aid in LLM_AGENT_IDS:
        catalog.append(
            AgentDefinition(aid, "llm", aid.replace("llm_", "LLM ").replace("_", " ").title(), _wrap_llm(aid))
        )

    for aid in BROKER_META_IDS:
        catalog.append(
            AgentDefinition(aid, "broker", aid.replace("_", " ").title(), _wrap_broker(aid))
        )

    default_brokers = [
        "58", "49", "45", "55", "34", "6", "38", "43", "70",
        "33", "42", "44", "62", "64", "17", "14", "4", "26", "36", "39",
        "41", "48", "50", "51", "52", "53", "56", "57", "60", "61",
        "63", "65", "66", "67", "68", "69", "71", "74", "75", "77",
        "80", "82", "83", "84", "88", "91", "95", "98", "100", "101",
    ]
    bids = broker_ids or default_brokers
    for bid in bids:
        aid = f"b_broker_{bid}"
        if any(a.agent_id == aid for a in catalog):
            continue
        catalog.append(
            AgentDefinition(aid, "broker", f"Broker {bid} desk", _wrap_broker(aid))
        )

    assert len(catalog) >= MIN_AGENT_COUNT, f"Catalog has {len(catalog)} agents, need {MIN_AGENT_COUNT}+"
    return catalog


def catalog_summary(catalog: list[AgentDefinition] | None = None) -> dict[str, int]:
    cat = catalog or build_agent_catalog()
    out: dict[str, int] = {"total": len(cat), "quant": 0, "financial": 0, "broker": 0, "llm": 0}
    for a in cat:
        out[a.domain] = out.get(a.domain, 0) + 1
    return out
