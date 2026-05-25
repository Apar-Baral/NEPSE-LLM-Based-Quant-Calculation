from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from backend.config import PROCESSED_DIR, load_yaml_config
from backend.llm.analyst import llm_status
from backend.scanner.llm_cache import apply_cached_llm_scores
from backend.utils.numeric import coerce_numeric

CACHE_PATH = PROCESSED_DIR / "llm_scanner_scores.json"
SHORT_HORIZONS = ("1D", "2D", "3D", "4D", "1W")


def _symbol_brief(row: pd.Series, panel: pd.DataFrame) -> str:
    sym = row["symbol"]
    sub = panel[(panel["symbol"] == sym) & (panel["side"] == "distribution")] if not panel.empty else pd.DataFrame()
    if "horizon" in sub.columns:
        sub = sub[sub["horizon"].isin(SHORT_HORIZONS)]
    parts = [
        f"{sym} LTP={row.get('ltp', 'n/a')}",
        f"1D_turnover_lac={row.get('daily_turnover_lac', 0):.1f}",
        f"broker_pressure={row.get('broker_pressure', 0):.0f}",
        f"EMS={row.get('early_momentum_score', 0):.0f}",
    ]
    return " | ".join(parts)


def _parse_llm_json(text: str) -> list[dict]:
    m = re.search(r"\[[\s\S]*\]", text.strip())
    if not m:
        return []
    try:
        data = json.loads(m.group())
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _call_llm_batch(batch_lines: list[str]) -> list[dict]:
    from backend.llm.analyst import _call_llm

    prompt = (
        "NEPSE floorsheet quant. Score EARLY LONG momentum only.\n"
        "Return ONLY JSON array: "
        '[{"symbol":"X","p_long":0.55,"tier":"Setup","note":"short reason"}]\n\n'
        + "\n".join(batch_lines)
    )
    try:
        return _parse_llm_json(_call_llm(prompt))
    except Exception:
        return []


def score_universe_with_llm(
    universe: pd.DataFrame,
    panel: pd.DataFrame,
    batch_size: int = 6,
    max_symbols: int | None = None,
    max_fetch: int | None = None,
    progress_fn: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """Fetch LLM scores for uncached symbols (limited batches to avoid UI hang)."""
    out = coerce_numeric(apply_cached_llm_scores(universe))

    if not llm_status().get("ready"):
        return out

    cfg = load_yaml_config("settings.yaml").get("scanner", {})
    max_symbols = max_symbols or cfg.get("llm_score_max", 60)
    max_fetch = max_fetch if max_fetch is not None else cfg.get("llm_refresh_max_fetch", 12)
    work = out.head(max_symbols).copy()

    cache: dict = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}

    rd = str(work["report_date"].iloc[0].date()) if "report_date" in work.columns and len(work) else "latest"
    cache_key = f"scores_{rd}"
    cached_rows = cache.get(cache_key, {})

    pending: list[tuple[str, str]] = []
    for _, row in work.iterrows():
        sym = row["symbol"]
        if sym not in cached_rows:
            pending.append((sym, _symbol_brief(row, panel)))
        if len(pending) >= max_fetch:
            break

    if not pending:
        if progress_fn:
            progress_fn(1.0, "All scores already cached.")
        return _apply_cache_to_df(out, cached_rows, set(work["symbol"]))

    total_batches = max(1, (len(pending) + batch_size - 1) // batch_size)
    for bi, i in enumerate(range(0, len(pending), batch_size)):
        batch = [line for _, line in pending[i : i + batch_size]]
        if progress_fn:
            progress_fn(bi / total_batches, f"LLM batch {bi + 1}/{total_batches} ({len(batch)} symbols)...")
        for item in _call_llm_batch(batch):
            sym = str(item.get("symbol", "")).upper()
            if sym:
                cached_rows[sym] = {
                    "llm_p_long": float(item.get("p_long", 0.25)),
                    "llm_tier": str(item.get("tier", "Neutral")),
                    "llm_note": str(item.get("note", ""))[:200],
                }

    cache[cache_key] = cached_rows
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    if progress_fn:
        progress_fn(1.0, f"Done — cached {len(cached_rows)} symbols.")
    return _apply_cache_to_df(out, cached_rows, set(work["symbol"]))


def _apply_cache_to_df(out: pd.DataFrame, cached_rows: dict, work_symbols: set[str]) -> pd.DataFrame:
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
