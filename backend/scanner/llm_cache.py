"""LLM scanner score cache (no API calls — safe for Streamlit hot-reload)."""

from __future__ import annotations

import json

import pandas as pd

from backend.config import PROCESSED_DIR, load_yaml_config
from backend.utils.numeric import coerce_numeric

CACHE_PATH = PROCESSED_DIR / "llm_scanner_scores.json"


def apply_cached_llm_scores(universe: pd.DataFrame) -> pd.DataFrame:
    """Merge cached llm_p_long / llm_tier / llm_note from disk."""
    out = universe.copy()
    for col in ("llm_p_long", "llm_tier", "llm_note"):
        if col not in out.columns:
            out[col] = None

    cache: dict = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}

    rd = (
        str(out["report_date"].iloc[0].date())
        if "report_date" in out.columns and len(out)
        else "latest"
    )
    cached_rows = cache.get(f"scores_{rd}", {})
    max_n = load_yaml_config("settings.yaml").get("scanner", {}).get("llm_score_max", 60)
    work_symbols = set(out["symbol"].head(max_n).tolist())

    for idx, row in out.iterrows():
        sym = row["symbol"]
        if sym not in cached_rows:
            continue
        c = cached_rows[sym]
        if c.get("llm_p_long") is not None:
            out.at[idx, "llm_p_long"] = float(c["llm_p_long"])
        if c.get("llm_tier"):
            out.at[idx, "llm_tier"] = str(c["llm_tier"])
        if c.get("llm_note"):
            out.at[idx, "llm_note"] = str(c["llm_note"])
        if sym in work_symbols and c.get("llm_p_long") is not None:
            out.at[idx, "p_long_momentum"] = float(c["llm_p_long"])

    return coerce_numeric(out)
