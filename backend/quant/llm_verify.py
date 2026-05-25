from __future__ import annotations

import json

import pandas as pd

from backend.llm.analyst import llm_status, symbol_to_metrics


def verify_with_llm(sym: str, row: pd.Series, step_summaries: list[dict]) -> dict:
    """LLM cross-check of quant steps (cached note used if API unavailable)."""
    cached_note = str(row.get("llm_note") or "")
    llm_p = row.get("llm_p_long")
    score = 50
    notes = []
    verified = False
    api_used = False

    if pd.notna(llm_p):
        lp = float(llm_p)
        if lp >= 0.55:
            score += 20
            notes.append(f"LLM P(long) {lp:.0%} — bullish")
        elif lp < 0.35:
            score -= 15
            notes.append(f"LLM cautious ({lp:.0%})")
        else:
            notes.append(f"LLM neutral ({lp:.0%})")

    if cached_note and cached_note not in ("None", "—", ""):
        verified = True
        notes.append(f"Cached narrative: {cached_note[:200]}")
        if any(w in cached_note.lower() for w in ("accum", "shakeout", "early", "long")):
            score += 10
        if any(w in cached_note.lower() for w in ("avoid", "distrib", "heavy sell")):
            score -= 12

    if not llm_status().get("ready"):
        return {
            "step": "LLM verification",
            "score": int(score),
            "pass": score >= 50,
            "verified": verified,
            "api_used": False,
            "notes": notes + ["LLM API not configured — using cache/heuristics only"],
        }

    try:
        from backend.llm.analyst import _call_llm

        steps_txt = json.dumps(step_summaries, indent=2)[:3000]
        metrics = symbol_to_metrics(row)
        prompt = (
            f"Verify NEPSE symbol {sym} for early LONG momentum.\n"
            f"Quant step results:\n{steps_txt}\n"
            f"Metrics:\n{json.dumps(metrics, indent=2)}\n"
            "Reply JSON only: "
            '{"verdict":"approve|caution|reject","score":0-100,"checks":["..."]}'
        )
        raw = _call_llm(prompt)
        m = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        score = int(m.get("score", score))
        verified = True
        api_used = True
        notes.append(f"LLM verdict: **{m.get('verdict', 'caution')}**")
        for c in m.get("checks", [])[:5]:
            notes.append(str(c))
    except Exception as exc:
        notes.append(f"LLM verify skipped: {exc}")

    return {
        "step": "LLM verification",
        "score": int(min(100, max(0, score))),
        "pass": score >= 52,
        "verified": verified,
        "api_used": api_used,
        "notes": notes,
    }
