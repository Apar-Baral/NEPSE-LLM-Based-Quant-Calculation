from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from backend.config import PROCESSED_DIR, load_yaml_config
from backend.llm.analyst import llm_status, test_llm_connection

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
        f"dist_risk={row.get('distribution_risk_score', 0):.0f}",
        f"shakeout={bool(row.get('dist_shakeout_flag', False))}",
    ]
    for h in SHORT_HORIZONS:
        hrow = sub[sub["horizon"] == h] if not sub.empty else pd.DataFrame()
        if not hrow.empty:
            r = hrow.iloc[0]
            parts.append(
                f"{h}: power={r.get('dominant_power','?')} net_lac={float(r.get('net_amount_sum',0) or 0):.1f}"
            )
    return " | ".join(parts)


def _parse_llm_json(text: str) -> list[dict]:
    text = text.strip()
    m = re.search(r"\[[\s\S]*\]", text)
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
        "You are a NEPSE floorsheet quant. Score each symbol for EARLY LONG momentum (not long-term distribution).\n"
        "Return ONLY a JSON array: "
        '[{"symbol":"X","p_long":0.0-1.0,"tier":"Watch|Setup|Trigger|Neutral|Invalidated","note":"one line"}]\n'
        "Favor: light short-term dist + rising broker buy skew + dist shakeout. "
        "Penalize: heavy 3M+ distribution without shakeout.\n\n"
        + "\n".join(batch_lines)
    )
    try:
        raw = _call_llm(prompt)
        return _parse_llm_json(raw)
    except Exception:
        return []


def _load_cache_for_universe(universe: pd.DataFrame) -> tuple[dict, dict, str]:
    cache: dict = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    rd = (
        str(universe["report_date"].iloc[0].date())
        if "report_date" in universe.columns and len(universe)
        else "latest"
    )
    cache_key = f"scores_{rd}"
    return cache, cache.get(cache_key, {}), cache_key


def _merge_cached_into_df(out: pd.DataFrame, cached_rows: dict, work_symbols: set[str]) -> pd.DataFrame:
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


def apply_cached_llm_scores(universe: pd.DataFrame) -> pd.DataFrame:
    """Merge LLM scores from disk cache only (delegates to llm_cache module)."""
    from backend.scanner.llm_cache import apply_cached_llm_scores as _apply

    return _apply(universe)


def score_universe_with_llm(
    universe: pd.DataFrame,
    panel: pd.DataFrame,
    batch_size: int = 12,
    max_symbols: int | None = None,
) -> pd.DataFrame:
    """Fetch new LLM scores for uncached symbols and merge into universe."""
    out = universe.copy()
    for col in ("llm_p_long", "llm_tier", "llm_note"):
        if col not in out.columns:
            out[col] = None

    status = llm_status()
    if not status.get("ready"):
        return apply_cached_llm_scores(out)

    cfg = load_yaml_config("settings.yaml").get("scanner", {})
    max_symbols = max_symbols or cfg.get("llm_score_max", 60)
    work = out.head(max_symbols).copy()

    cache, cached_rows, cache_key = _load_cache_for_universe(work)

    pending = []
    for _, row in work.iterrows():
        sym = row["symbol"]
        if sym in cached_rows:
            continue
        pending.append(_symbol_brief(row, panel))

    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        if not batch:
            continue
        parsed = _call_llm_batch(batch)
        for item in parsed:
            sym = str(item.get("symbol", "")).upper()
            if not sym:
                continue
            cached_rows[sym] = {
                "llm_p_long": float(item.get("p_long", item.get("p_long_momentum", 0.25))),
                "llm_tier": str(item.get("tier", "Neutral")),
                "llm_note": str(item.get("note", ""))[:200],
            }

    cache[cache_key] = cached_rows
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    return _merge_cached_into_df(out, cached_rows, set(work["symbol"].tolist()))
