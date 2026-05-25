from __future__ import annotations

import json
from datetime import datetime

import httpx
import pandas as pd

from backend.config import get_settings, load_yaml_config
from backend.features.pattern_library import find_historical_analogs


SYSTEM_PROMPT = """You are a NEPSE quantitative analyst. You ONLY use the JSON metrics provided.
Never invent prices or probabilities. If data is missing, say "insufficient data".
Output concise actionable analysis for early long momentum detection."""


def _llm_config() -> dict:
    return load_yaml_config("settings.yaml")["llm"]


def _parse_chat_response(data: dict) -> str:
    """Extract final answer; include reasoning trace for DeepSeek reasoner models when present."""
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    if reasoning and content:
        return f"**Reasoning**\n{reasoning}\n\n**Analysis**\n{content}"
    return content or reasoning or "(empty LLM response)"


def _openai_compatible_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    with httpx.Client(timeout=120) as client:
        resp = client.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json().get("error", {}).get("message", detail)
            except Exception:
                pass
            raise RuntimeError(f"LLM API error {resp.status_code}: {detail}")
        return _parse_chat_response(resp.json())


def llm_status() -> dict:
    """Return active LLM provider config (no secrets)."""
    settings = get_settings()
    cfg = _llm_config()
    provider = settings.llm_provider or cfg.get("provider", "ollama")
    status = {"provider": provider, "ready": False, "model": None, "hint": ""}

    if provider == "deepseek":
        status["model"] = settings.deepseek_model or cfg.get("deepseek_model", "deepseek-v4-pro")
        status["ready"] = bool(settings.deepseek_api_key)
        status["hint"] = "Set DEEPSEEK_API_KEY in .env" if not status["ready"] else ""
    elif provider == "openai":
        status["model"] = cfg.get("openai_model", "gpt-4o-mini")
        status["ready"] = bool(settings.openai_api_key)
        status["hint"] = "Set OPENAI_API_KEY in .env" if not status["ready"] else ""
    else:
        status["model"] = cfg.get("model", "llama3.1")
        status["ready"] = True
        status["hint"] = f"Ollama at {settings.ollama_base_url}"

    return status


def _call_llm(prompt: str) -> str:
    settings = get_settings()
    cfg = _llm_config()
    provider = (settings.llm_provider or cfg.get("provider", "ollama")).lower()
    temperature = cfg.get("temperature", 0.2)
    max_tokens = cfg.get("max_tokens", 4096)

    if provider == "deepseek" and settings.deepseek_api_key:
        try:
            return _openai_compatible_chat(
                base_url=settings.deepseek_base_url or cfg.get("deepseek_base_url", "https://api.deepseek.com"),
                api_key=settings.deepseek_api_key,
                model=settings.deepseek_model or cfg.get("deepseek_model", "deepseek-v4-pro"),
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            return _fallback_report(prompt, str(exc))

    if provider == "openai" and settings.openai_api_key:
        try:
            return _openai_compatible_chat(
                base_url="https://api.openai.com",
                api_key=settings.openai_api_key,
                model=cfg.get("openai_model", "gpt-4o-mini"),
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            return _fallback_report(prompt, str(exc))

    # Ollama fallback
    base = settings.ollama_base_url or cfg.get("base_url", "http://localhost:11434")
    try:
        with httpx.Client(timeout=90) as client:
            resp = client.post(
                f"{base.rstrip('/')}/api/chat",
                json={
                    "model": cfg.get("model", "llama3.1"),
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
    except Exception as exc:
        return _fallback_report(prompt, str(exc))


def test_llm_connection() -> dict:
    """Quick connectivity test for configured provider."""
    status = llm_status()
    if not status["ready"] and status["provider"] in ("deepseek", "openai"):
        return {**status, "ok": False, "response": status["hint"]}

    try:
        reply = _call_llm(
            'Reply with exactly: "NEPSE Quant LLM connected." and name your model.'
        )
        return {**status, "ok": "NEPSE Quant LLM connected" in reply or len(reply) > 10, "response": reply}
    except Exception as exc:
        return {**status, "ok": False, "response": str(exc)}


def _fallback_report(prompt: str, error: str = "") -> str:
    """Rule-based report when LLM unavailable."""
    try:
        blob = prompt.split("DATA:\n", 1)[-1] if "DATA:\n" in prompt else prompt.split("METRICS:\n", 1)[-1]
        data = json.loads(blob)
        if "metrics" in data:
            data = data["metrics"]
    except Exception:
        return f"LLM unavailable ({error}). Review scanner table for quant scores."

    sym = data.get("symbol", "N/A")
    p = data.get("p_long_momentum", 0)
    ems = data.get("early_momentum_score", 0)
    tier = data.get("signal_tier", "Neutral")
    return (
        f"**{sym}** — Tier: {tier}\n"
        f"- P(long momentum): {p:.0%}\n"
        f"- Early momentum score: {ems:.0f}/100\n"
        f"- Smart money: {data.get('smart_money_score', 0):.0f}\n"
        f"- Distribution risk: {data.get('distribution_risk_score', 0):.0f}\n"
        f"_LLM error: {error}. Set DEEPSEEK_API_KEY and LLM_PROVIDER=deepseek in .env_"
    )


def symbol_to_metrics(row: pd.Series) -> dict:
    keys = [
        "symbol", "ltp", "p_long_momentum", "expected_return_10d", "confidence",
        "signal_tier", "smart_money_score", "early_momentum_score", "early_rank_score",
        "distribution_risk_score", "mtf_convergence", "acc_dist_ratio",
        "demand_zone_distance_pct", "supply_zone_distance_pct", "ofi",
        "daily_volume", "daily_turnover_lac", "float_turnover_1d_abs", "volume_rank",
        "turnover_rank", "early_pick_rank", "broker_pressure", "top_broker_ids",
        "circular_risk", "circular_flag", "llm_p_long", "llm_note",
        "pattern_horizon_ladder", "pattern_dist_shakeout", "pattern_float_spike",
    ]
    return {k: _serialize(row.get(k)) for k in keys if k in row.index}


def _serialize(v):
    if pd.isna(v):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return str(v)
    if hasattr(v, "item"):
        return v.item()
    return v


def generate_symbol_report(row: pd.Series) -> str:
    metrics = symbol_to_metrics(row)
    analogs = find_historical_analogs(row)
    payload = {"metrics": metrics, "historical_analogs": analogs}
    prompt = f"Analyze this NEPSE symbol for early long momentum.\nDATA:\n{json.dumps(payload, indent=2)}"
    return _call_llm(prompt)


def _prepare_scanner_df(scanner_df: pd.DataFrame) -> pd.DataFrame:
    from backend.utils.numeric import coerce_numeric

    df = coerce_numeric(scanner_df.copy())
    if "p_long_momentum" in df.columns:
        df["p_long_momentum"] = pd.to_numeric(df["p_long_momentum"], errors="coerce").fillna(0)
    if "early_rank_score" in df.columns:
        df["early_rank_score"] = pd.to_numeric(df["early_rank_score"], errors="coerce").fillna(0)
    return df


def generate_daily_brief(scanner_df: pd.DataFrame, top_n: int | None = None) -> str:
    if scanner_df.empty:
        return "No scanner data available for today."

    cfg = load_yaml_config("settings.yaml").get("scanner", {})
    detail_n = top_n or cfg.get("brief_detail_n", 25)

    ranked = _prepare_scanner_df(scanner_df)
    if "early_rank_score" in ranked.columns:
        top = ranked.nlargest(detail_n, "early_rank_score")
    else:
        top = ranked.head(detail_n)
    summaries = [symbol_to_metrics(row) for _, row in top.iterrows()]

    vol_note = ""
    if "daily_volume" in scanner_df.columns:
        vol_note = f"Universe: top {len(scanner_df)} NEPSE symbols by 1D traded volume. "

    prompt = (
        f"{vol_note}Analyze early LONG momentum candidates among high-volume NEPSE stocks.\n"
        f"Focus on symbols with highest early_rank_score, accumulation patterns, and volume spikes.\n"
        f"Output a markdown table: Symbol | LTP | Volume | Early Rank | Verdict | Key Drivers | Risks | Action\n"
        f"Then summarize top 3 actionable long setups and names to avoid.\n"
        f"METRICS (top {len(summaries)} by early prediction rank within high-volume universe):\n"
        f"{json.dumps(summaries, indent=2)}"
    )
    return _call_llm(prompt)


def _extract_symbols_from_question(question: str, known_symbols: list[str]) -> list[str]:
    import re

    q = question.upper()
    known = {s.upper() for s in known_symbols}
    found = []
    for sym in sorted(known, key=len, reverse=True):
        if re.search(rf"\b{re.escape(sym)}\b", q):
            found.append(sym)
    tokens = re.findall(r"\b[A-Z]{2,12}\b", q)
    for t in tokens:
        if t in known and t not in found:
            found.append(t)
    return found


def chat_query(
    question: str,
    context_df: pd.DataFrame,
    extra_rows: pd.DataFrame | None = None,
) -> str:
    ctx = _prepare_scanner_df(context_df)
    if extra_rows is not None and not extra_rows.empty:
        extra = _prepare_scanner_df(extra_rows)
        ctx = pd.concat([extra, ctx], ignore_index=True).drop_duplicates(subset=["symbol"], keep="first")

    known = ctx["symbol"].astype(str).str.upper().tolist() if "symbol" in ctx.columns else []
    asked = _extract_symbols_from_question(question, known)

    if asked:
        focus = ctx[ctx["symbol"].astype(str).str.upper().isin(asked)]
        rest = ctx[~ctx["symbol"].astype(str).str.upper().isin(asked)]
        if "early_rank_score" in rest.columns:
            rest = rest.nlargest(min(15, len(rest)), "early_rank_score")
        top = pd.concat([focus, rest], ignore_index=True).head(45)
    elif "early_rank_score" in ctx.columns:
        top = ctx.nlargest(min(40, len(ctx)), "early_rank_score")
    elif "daily_turnover_lac" in ctx.columns:
        top = ctx.nlargest(min(40, len(ctx)), "daily_turnover_lac")
    else:
        top = ctx.head(30)

    metrics = [symbol_to_metrics(r) for _, r in top.iterrows()]
    focus_note = f"User asked about: {', '.join(asked)}. Prioritize these symbols.\n" if asked else ""
    prompt = (
        f"User question: {question}\n\n"
        f"{focus_note}"
        f"Context: NEPSE high-volume early long momentum scanner.\n"
        f"Symbol metrics (JSON):\n{json.dumps(metrics, indent=2)}"
    )
    return _call_llm(prompt)
