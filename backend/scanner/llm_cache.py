"""LLM scanner score cache (no API calls — safe for Streamlit hot-reload)."""

from __future__ import annotations

import json

import pandas as pd

from backend.config import PROCESSED_DIR, load_yaml_config

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
        out.at[idx, "llm_p_long"] = c.get("llm_p_long")
        out.at[idx, "llm_tier"] = c.get("llm_tier")
        out.at[idx, "llm_note"] = c.get("llm_note")
        if sym in work_symbols:
            if c.get("llm_p_long") is not None:
                out.at[idx, "p_long_momentum"] = c.get("llm_p_long")
            if c.get("llm_tier"):
                out.at[idx, "signal_tier"] = c.get("llm_tier")
    return out
