"""UI metadata: what each quant algorithm uses and what it returns."""

from __future__ import annotations

from typing import Any

ALGORITHM_SPECS: dict[str, dict[str, Any]] = {
    "Volumetric": {
        "title": "Volumetric analysis",
        "uses": [
            "1D turnover (Lac) vs scanner universe",
            "Daily traded quantity",
            "Float turnover z-score (unusual activity)",
        ],
        "gives": "Liquidity score 0–100 — is there enough two-sided flow to trade this name?",
    },
    "Broker analysis": {
        "title": "Broker desk analysis",
        "uses": [
            "Broker pressure (1D buy vs sell skew)",
            "Top broker desks on this symbol",
            "Circular / wash detection",
            "Directional % of broker activity",
        ],
        "gives": "Desk conviction score — whether smart brokers are skewed long without circular churn.",
    },
    "Price action": {
        "title": "Price action — order blocks & FVG",
        "uses": [
            "Demand / supply zones (order-block proxy)",
            "LTP vs zones",
            "Order-flow imbalance (OFI)",
            "1D net amount & distribution power",
            "Multi-horizon LTP gaps",
        ],
        "gives": "Structure score — demand OB support, supply overhead, bullish/bearish FVG-style hints.",
    },
    "Quant momentum": {
        "title": "Momentum & ML blend",
        "uses": [
            "Effective P(long) after distribution calibration",
            "Early momentum score (EMS)",
            "Floorsheet momentum score",
            "Multi-timeframe convergence",
            "Distribution risk & shakeout pattern",
        ],
        "gives": "Trend score — calibrated probability of a 10D long working on this setup.",
    },
    "LLM verification": {
        "title": "LLM verification",
        "uses": [
            "Outputs from volumetric, broker, PA, momentum steps",
            "Cached LLM narrative / P(long) if available",
            "DeepSeek API (optional) cross-check",
        ],
        "gives": "Narrative verification — does language agree with quant steps?",
    },
}


def spec_for_step(step_name: str) -> dict[str, Any]:
    return ALGORITHM_SPECS.get(
        step_name,
        {
            "title": step_name,
            "uses": ["Symbol metrics from pipeline"],
            "gives": "Step score and pass/fail",
        },
    )
