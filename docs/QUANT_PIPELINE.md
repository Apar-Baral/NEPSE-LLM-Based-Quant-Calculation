# NEPSE Quant Pipeline

## Implemented algorithms

| Layer | Module | What it does |
|-------|--------|----------------|
| Volumetric | `backend/quant/volumetric.py` | 1D turnover, daily qty, float z-score, universe percentile |
| Broker | `backend/quant/broker_quant.py` | Watch-broker skew, circular flags, directional flow |
| Price action | `backend/quant/price_action.py` | Order blocks (tech demand/supply), OFI, bullish/bearish FVG proxies |
| Momentum | `backend/quant/momentum_quant.py` | Effective P(long) & EMS (distribution-mode aware), MTF, shakeout |
| LLM verify | `backend/quant/llm_verify.py` | Cross-checks quant steps + cached/API narrative |
| Multimodal ML | `backend/models/multimodal/` | Temporal CNN, graph propagation, cross-modal attention |
| Classic ML | `backend/models/trainer.py` | LightGBM + XGBoost on floorsheet features |

## Symbol Deep Dive

`run_quant_analysis()` runs all steps and returns a composite 0–100 score.

Effective scores use `backend.signals.momentum_rules._effective_scores` so Trigger tier and high turnover are not shown as flat 25% P(long).

## Data requirements

- Daily distribution/accumulation Excel → panel → features
- Broker CSV → broker panel (58, 49, …)
- Optional OHLCV CSV for backtest forward returns
