"""Human-readable catalog of implemented quant / ML / financial techniques."""

from __future__ import annotations

ALGORITHM_SECTIONS = [
    {
        "title": "Floorsheet feature engineering",
        "items": [
            "Multi-horizon acc/dist net amount & float turnover (1D→3Y)",
            "Power scores (Strong Buy → Strong Sell ladder)",
            "MTF convergence — short horizons aligned",
            "Acc/Dist ratio, OFI (order flow imbalance)",
            "Demand/supply zone distance %",
            "Cross-sectional float turnover z-score",
            "Smart money score, floorsheet momentum, distribution risk score",
            "Pattern flags: dist shakeout, float spike, zone+power",
        ],
    },
    {
        "title": "Machine learning",
        "items": [
            "LightGBM classifier + isotonic calibration → P(long 10D)",
            "XGBoost ensemble member",
            "Isolation Forest anomaly score",
            "Historical analog matching (pattern library)",
            "Multimodal: temporal CNN + broker graph + cross-modal attention",
        ],
    },
    {
        "title": "Quant pipeline (5 steps)",
        "items": [
            "Volumetric — turnover rank, float z, daily qty",
            "Broker — pressure, circular/wash, top-10 conviction",
            "Price action — order blocks (demand/supply zones), FVG proxy",
            "Momentum — effective P(long), EMS, MTF, shakeout",
            "LLM verify — optional DeepSeek check on step outputs",
        ],
    },
    {
        "title": "Financial / technical (classical)",
        "items": [
            "RSI (14), ROC momentum, Bollinger %B",
            "Sharpe ratio proxy, max drawdown on LTP series",
            "Broker Herfindahl concentration (HHI)",
            "Z-score mean reversion on turnover",
            "Expected return 10D from ML + effective score blending",
        ],
    },
    {
        "title": "Broker desk",
        "items": [
            "Top-N broker discovery by market activity",
            "Per-broker conviction (share, directional %, two-side churn)",
            "Circular trading: wash score, reciprocal brokers, universe percentile",
        ],
    },
    {
        "title": "Agent fleet (155)",
        "items": [
            "31 quant + 32 financial + 67 broker + 25 LLM rule-analysts",
            "Parallel deploy → domain scores + long consensus %",
            "Logic graph: symbol → pipeline steps → brokers → signal",
        ],
    },
]
