# NEPSE LLM-Based Quant Calculation

**Early long-momentum detection** for the Nepal Stock Exchange (NEPSE), powered by Floorsheet Analytics accumulation/distribution data, quant feature engineering, ML probability models, and LLM research briefs.

[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](frontend/streamlit_app.py)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](backend/api/main.py)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB)](requirements.txt)

**Author:** [Apar-Baral](https://github.com/Apar-Baral) · dedsecaparb@gmail.com

**Repository:** [NEPSE-LLM-Based-Quant-Calculation](https://github.com/Apar-Baral/NEPSE-LLM-Based-Quant-Calculation)

---

## What it does

| Layer | Capability |
|-------|------------|
| **Ingestion** | Daily Accumulation + Distribution Excel (multi-sheet), CSV backfill |
| **Quant** | Multi-horizon features (1D→3Y), Smart Money, Early Momentum, distribution risk, broker pressure |
| **ML** | LightGBM long-momentum probability, walk-forward validation, SHAP, analog patterns |
| **Signals** | Watch → Setup → Trigger → Confirmed (distribution-aware when acc sheets are missing) |
| **Scanner** | Top **120 high-volume** symbols ranked for early movement |
| **LLM** | DeepSeek / OpenAI / Ollama daily briefs and symbol reports |
| **UI** | Streamlit dashboard + React/FastAPI production stack |

---

## Quick start

```powershell
cd "E:\Major Project - Nepse Data LLM"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Copy and edit secrets
copy .env.example .env

# Run pipeline on distribution CSVs in Data/ (or upload Excel in UI)
python scripts/run_pipeline.py

# Dashboard
streamlit run frontend/streamlit_app.py

# API (optional)
uvicorn backend.api.main:app --reload --port 8000
```

### Daily workflow

1. Drop **Accumulation** and **Distribution** Excel files via **Daily Upload** (or `data/inbox/`).
2. Click **Run Pipeline** in the sidebar.
3. Use **Momentum Scanner** — filter **Trigger / Setup / Watch** for actionable names.
4. Open **Symbol Deep Dive** for horizon ladder, broker buy/sell skew, zones, and LLM report.

---

## Signal tiers (Momentum Scanner)

| Tier | Meaning |
|------|---------|
| **Confirmed** | Strong early momentum + probability (best when accumulation data is loaded) |
| **Trigger** | Actionable early setup — review for entry |
| **Setup** | Distribution shakeout / broker skew building |
| **Watch** | High volume + early rank — monitor |
| **Neutral** | No clear edge |
| **Invalidated** | Heavy long-horizon distribution — avoid fresh longs |

> With **distribution-only** data, the engine uses floorsheet momentum, broker pressure, and dist shakeout patterns. Upload **both** Excel workbooks for full early-momentum scores.

---

## Project layout

```
backend/          # ingest, features, ML, signals, scanner, LLM, API
frontend/         # Streamlit dashboard
frontend_react/   # React + Vite UI
config/           # horizons.yaml, settings.yaml
scripts/          # pipeline, train, verify_imports
data/             # processed parquet (local, gitignored)
```

---

## Configuration

- `config/settings.yaml` — scanner size (120), signal thresholds, LLM provider
- `config/horizons.yaml` — 1D … 3Y horizon mapping
- `.env` — `DEEPSEEK_API_KEY`, `LLM_PROVIDER`, etc.

---

## License

MIT — see [LICENSE](LICENSE).

Built for quantitative research on NEPSE floorsheet data. **Not financial advice.**
