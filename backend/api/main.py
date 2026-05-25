from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.backtest.engine import run_backtest
from backend.config import ensure_dirs, load_yaml_config
from backend.db.store import DataStore
from backend.llm.analyst import chat_query, generate_daily_brief, generate_symbol_report, llm_status, test_llm_connection
from backend.features.pattern_library import find_historical_analogs
from backend.models.trainer import compute_shap_values
from backend.pipeline import run_pipeline
from backend.scanner.volume_universe import get_latest_scanner_universe

ensure_dirs()
app = FastAPI(title="NEPSE Quant API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/scan")
def scan(top_n: int = 120):
    store = DataStore()
    df = store.load_predictions()
    panel = store.load_panel()
    if df.empty:
        return {"data": [], "count": 0, "universe": "high_volume"}
    universe = get_latest_scanner_universe(df, panel=panel, top_n=top_n)
    latest = universe["report_date"].max() if not universe.empty else None
    return {
        "data": universe.to_dict(orient="records"),
        "count": len(universe),
        "report_date": str(latest.date()) if latest is not None else None,
        "universe": f"top_{top_n}_high_volume",
    }


@app.get("/symbol/{symbol}")
def symbol_detail(symbol: str):
    store = DataStore()
    sym = symbol.upper()
    features = store.load_features()
    panel = store.load_panel()
    preds = store.load_predictions()

    sym_feat = features[features["symbol"] == sym].sort_values("report_date")
    sym_panel = panel[panel["symbol"] == sym]
    sym_pred = preds[preds["symbol"] == sym].tail(1)

    shap = {}
    if not sym_feat.empty:
        shap = compute_shap_values(sym_feat, sym)

    analogs = {}
    if not sym_pred.empty:
        analogs = find_historical_analogs(sym_pred.iloc[0])

    return {
        "symbol": sym,
        "latest": sym_pred.to_dict(orient="records"),
        "history": sym_feat.tail(30).to_dict(orient="records"),
        "panel": sym_panel.to_dict(orient="records"),
        "shap": shap,
        "analogs": analogs,
    }


@app.post("/upload")
async def upload(
    accumulation: UploadFile | None = File(None),
    distribution: UploadFile | None = File(None),
    ohlcv: UploadFile | None = File(None),
    report_date: date | None = None,
):
    acc_path = dist_path = ohlcv_path = None
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        if accumulation:
            acc_path = td / accumulation.filename
            acc_path.write_bytes(await accumulation.read())
        if distribution:
            dist_path = td / distribution.filename
            dist_path.write_bytes(await distribution.read())
        if ohlcv:
            ohlcv_path = td / ohlcv.filename
            ohlcv_path.write_bytes(await ohlcv.read())
        result = run_pipeline(report_date=report_date, acc_path=acc_path, dist_path=dist_path, ohlcv_path=ohlcv_path)
    return result


@app.post("/pipeline/run")
def pipeline_run():
    return run_pipeline()


@app.get("/llm/status")
def llm_status_route():
    return llm_status()


@app.get("/llm/test")
def llm_test_route():
    return test_llm_connection()


@app.get("/llm/brief")
def llm_brief(top_n: int = 120):
    store = DataStore()
    df = store.load_predictions()
    panel = store.load_panel()
    if df.empty:
        return {"brief": "No data. Run pipeline first."}
    universe = get_latest_scanner_universe(df, panel=panel, top_n=top_n)
    return {"brief": generate_daily_brief(universe), "universe_size": len(universe)}


@app.get("/llm/symbol/{symbol}")
def llm_symbol(symbol: str):
    store = DataStore()
    df = store.load_predictions()
    row = df[df["symbol"] == symbol.upper()].tail(1)
    if row.empty:
        return {"report": "Symbol not found"}
    return {"report": generate_symbol_report(row.iloc[0])}


@app.post("/llm/chat")
def llm_chat(req: ChatRequest):
    store = DataStore()
    df = store.load_predictions()
    return {"answer": chat_query(req.question, df)}


@app.get("/backtest")
def backtest(entry_tier: str = "Trigger", hold_days: int = 10):
    store = DataStore()
    signals = store.load_predictions()
    ohlcv = store.load_ohlcv()
    return run_backtest(signals, ohlcv, entry_tier=entry_tier, hold_days=hold_days)
