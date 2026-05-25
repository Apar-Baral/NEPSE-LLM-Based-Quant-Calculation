"""Per-agent execution logic (quant · financial · broker · LLM rule-analysts)."""

from __future__ import annotations

import pandas as pd

from backend.agents.base import AgentContext, AgentResult, AgentSignal
from backend.config_signals import get_signal_config
from backend.quant.broker_quant import analyze_brokers
from backend.quant.momentum_quant import analyze_momentum
from backend.quant.price_action import detect_fair_value_gaps, detect_order_blocks
from backend.quant.volumetric import analyze_volume
from backend.scanner.broker_desk import analyze_symbol_brokers
from backend.scanner.broker_top10 import analyze_broker_row, discover_top_brokers, symbol_top_brokers_table
from backend.backtest.engine import build_price_series_from_features
from backend.quant.financial_techniques import (
    analyze_price_series,
    herfindahl_index,
    z_score_reversion,
)
from backend.signals.effective_scores import effective_scores


def _sig(score: float, bull_at: float = 58, bear_at: float = 42) -> AgentSignal:
    if score >= bull_at:
        return "bullish"
    if score <= bear_at:
        return "bearish"
    return "neutral"


def _ok(agent_id: str, domain: str, name: str, score: float, summary: str, signal: AgentSignal | None = None, **metrics) -> AgentResult:
    return AgentResult(
        agent_id=agent_id,
        domain=domain,  # type: ignore[arg-type]
        name=name,
        status="ok",
        score=round(max(0, min(100, score)), 1),
        signal=signal or _sig(score),
        summary=summary[:200],
        metrics=metrics,
    )


def _skip(agent_id: str, domain: str, name: str, reason: str) -> AgentResult:
    return AgentResult(
        agent_id=agent_id,
        domain=domain,  # type: ignore[arg-type]
        name=name,
        status="skip",
        score=50.0,
        signal="skip",
        summary=reason,
    )


def _num(row: pd.Series, key: str, default: float = 0.0) -> float:
    v = row.get(key)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --- Quant executors ---


def run_quant_agent(agent_id: str, ctx: AgentContext) -> AgentResult:
    row = ctx.row
    universe = ctx.universe

    handlers = {
        "q_turnover_abs": lambda: _q_turnover_abs(row),
        "q_turnover_pct": lambda: _q_turnover_pct(row, universe),
        "q_daily_qty": lambda: _q_daily_qty(row),
        "q_float_z": lambda: _q_float_z(row),
        "q_float_turn_1d": lambda: _q_float_turn(row),
        "q_p_long_eff": lambda: _q_p_long(row),
        "q_p_long_raw": lambda: _q_p_long_raw(row),
        "q_ems": lambda: _q_ems(row),
        "q_early_rank": lambda: _q_rank(row),
        "q_mtf": lambda: _q_mtf(row),
        "q_dist_risk": lambda: _q_drs(row),
        "q_shakeout": lambda: _q_shakeout(row),
        "q_acc_dist": lambda: _q_acc_dist(row),
        "q_ofi": lambda: _q_ofi(row),
        "q_ob_demand": lambda: _q_ob_demand(row),
        "q_ob_supply": lambda: _q_ob_supply(row),
        "q_fvg_bull": lambda: _q_fvg_bull(row, ctx.panel_sym),
        "q_fvg_bear": lambda: _q_fvg_bear(row, ctx.panel_sym),
        "q_dist_1d": lambda: _q_horizon_power(row, "dist_1D"),
        "q_dist_1w": lambda: _q_horizon_power(row, "dist_1W"),
        "q_acc_1d": lambda: _q_horizon_power(row, "acc_1D"),
        "q_acc_1w": lambda: _q_horizon_power(row, "acc_1W"),
        "q_smart_money": lambda: _q_smart(row),
        "q_analog": lambda: _q_analog(row),
        "q_anomaly": lambda: _q_anomaly(row),
        "q_exp_return": lambda: _q_exp_ret(row),
        "q_ml_conf": lambda: _q_conf(row),
        "q_vol_pipeline": lambda: _q_vol_pipe(row, universe),
        "q_broker_pipeline": lambda: _q_broker_pipe(ctx.symbol, row, ctx.broker_panel),
        "q_pa_pipeline": lambda: _q_pa_pipe(row, ctx.panel_sym),
        "q_momentum_pipeline": lambda: _q_mom_pipe(row),
        "q_rsi": lambda: _q_rsi(ctx),
        "q_roc_momentum": lambda: _q_roc(ctx),
        "q_bollinger": lambda: _q_bb(ctx),
        "q_turnover_zscore": lambda: _q_turn_z(row, ctx),
    }
    fn = handlers.get(agent_id)
    if fn is None:
        return _skip(agent_id, "quant", agent_id, "unknown quant agent")
    return fn()


def _q_turnover_abs(row: pd.Series) -> AgentResult:
    t = _num(row, "daily_turnover_lac")
    s = 35 + min(45, t / 8)
    return _ok("q_turnover_abs", "quant", "1D turnover magnitude", s, f"Turnover {t:,.0f} Lac", turnover=t)


def _q_turnover_pct(row: pd.Series, universe: pd.DataFrame | None) -> AgentResult:
    t = _num(row, "daily_turnover_lac")
    pct = 50.0
    if universe is not None and not universe.empty and "daily_turnover_lac" in universe.columns:
        u = pd.to_numeric(universe["daily_turnover_lac"], errors="coerce").fillna(0)
        if u.max() > 0:
            pct = float((u < t).mean() * 100)
    s = 40 + pct * 0.55
    return _ok("q_turnover_pct", "quant", "Turnover vs universe", s, f"Top {100-pct:.0f}% by 1D turnover", percentile=round(pct, 1))


def _q_daily_qty(row: pd.Series) -> AgentResult:
    q = _num(row, "daily_volume")
    s = 40 + min(40, q / 2500)
    return _ok("q_daily_qty", "quant", "Daily quantity", s, f"Qty {q:,.0f}", qty=q)


def _q_float_z(row: pd.Series) -> AgentResult:
    z = _num(row, "float_turnover_zscore") or _num(row, "float_turnover_zscore_hv")
    s = 45 + min(40, z * 18)
    return _ok("q_float_z", "quant", "Float turnover z-score", s, f"Z={z:.2f}", z=z)


def _q_float_turn(row: pd.Series) -> AgentResult:
    ft = _num(row, "float_turnover_1d_abs")
    s = 42 + min(45, ft * 12)
    return _ok("q_float_turn_1d", "quant", "Float turnover 1D", s, f"Float turn {ft:.2f}", ft=ft)


def _q_p_long(row: pd.Series) -> AgentResult:
    cfg = get_signal_config()
    p, _, _ = effective_scores(row, cfg)
    s = p * 100
    return _ok("q_p_long_eff", "quant", "Effective P(long)", s, f"P(long) {p:.0%}", p=p)


def _q_p_long_raw(row: pd.Series) -> AgentResult:
    raw = _num(row, "p_long_momentum_raw") or _num(row, "p_long_momentum")
    s = raw * 100 if raw <= 1 else min(100, raw)
    return _ok("q_p_long_raw", "quant", "Raw ML P(long)", s, f"Raw {raw:.0%}" if raw <= 1 else f"Raw {raw:.1f}%", raw=raw)


def _q_ems(row: pd.Series) -> AgentResult:
    cfg = get_signal_config()
    _, ems, _ = effective_scores(row, cfg)
    s = min(100, ems * 1.4)
    return _ok("q_ems", "quant", "Early momentum score", s, f"EMS {ems:.0f}", ems=ems)


def _q_rank(row: pd.Series) -> AgentResult:
    r = _num(row, "early_rank_score")
    s = 35 + r * 120
    return _ok("q_early_rank", "quant", "Early rank score", s, f"Rank {r:.2f}", rank=r)


def _q_mtf(row: pd.Series) -> AgentResult:
    m = _num(row, "mtf_convergence")
    s = 40 + m * 55
    return _ok("q_mtf", "quant", "MTF convergence", s, f"MTF {m:.0%}", mtf=m)


def _q_drs(row: pd.Series) -> AgentResult:
    d = _num(row, "distribution_risk_score", 50)
    s = max(10, 100 - d)
    sig: AgentSignal = "bearish" if d >= 75 else ("bullish" if d <= 35 else "neutral")
    return _ok("q_dist_risk", "quant", "Distribution risk (inverse)", s, f"DRS {d:.0f}", drs=d, signal=sig)


def _q_shakeout(row: pd.Series) -> AgentResult:
    sh = row.get("pattern_dist_shakeout") or row.get("dist_shakeout_flag")
    active = sh in (True, 1, "True", "true")
    s = 72 if active else 48
    return _ok("q_shakeout", "quant", "Dist shakeout pattern", s, "Shakeout active" if active else "No shakeout", shakeout=active)


def _q_acc_dist(row: pd.Series) -> AgentResult:
    a = _num(row, "acc_dist_ratio")
    s = 45 + a * 35
    return _ok("q_acc_dist", "quant", "Acc/Dist ratio", s, f"Ratio {a:.2f}", ratio=a)


def _q_ofi(row: pd.Series) -> AgentResult:
    o = _num(row, "ofi")
    s = 50 + o * 80
    return _ok("q_ofi", "quant", "Order flow imbalance", s, f"OFI {o:+.2f}", ofi=o)


def _q_ob_demand(row: pd.Series) -> AgentResult:
    ob = detect_order_blocks(row)
    s = ob["order_block_score"]
    bias = ob.get("order_block_bias", "neutral")
    sig = "bullish" if bias == "at_demand_ob" else "neutral"
    return _ok("q_ob_demand", "quant", "Demand order block", s, "; ".join(ob.get("notes", [])[:2]) or "OB scan", signal=sig)


def _q_ob_supply(row: pd.Series) -> AgentResult:
    ob = detect_order_blocks(row)
    s = 100 - min(40, 8 if ob.get("supply_zone") else 0)
    sig = "bearish" if ob.get("order_block_bias") == "under_supply_ob" else "neutral"
    return _ok("q_ob_supply", "quant", "Supply overhead", s, "Supply pressure" if ob.get("supply_zone") else "Clear overhead", signal=sig)


def _q_fvg_bull(row: pd.Series, panel: pd.DataFrame) -> AgentResult:
    pa = detect_fair_value_gaps(row, panel)
    s = pa["score"] if pa.get("bullish_fvg") else 48
    sig = "bullish" if pa.get("bullish_fvg") else "neutral"
    return _ok("q_fvg_bull", "quant", "Bullish FVG proxy", s, "Bull FVG hint" if pa.get("bullish_fvg") else "No bull FVG", signal=sig)


def _q_fvg_bear(row: pd.Series, panel: pd.DataFrame) -> AgentResult:
    pa = detect_fair_value_gaps(row, panel)
    s = 35 if pa.get("bearish_fvg") else 55
    sig = "bearish" if pa.get("bearish_fvg") else "neutral"
    return _ok("q_fvg_bear", "quant", "Bearish FVG proxy", s, "Bear FVG hint" if pa.get("bearish_fvg") else "No bear FVG", signal=sig)


def _q_horizon_power(row: pd.Series, prefix: str) -> AgentResult:
    p = _num(row, f"{prefix}_power_score", 3)
    s = 40 + (p - 2) * 22
    sig = "bullish" if p >= 3 else ("bearish" if p <= 1 else "neutral")
    return _ok(f"q_{prefix.lower()}", "quant", f"{prefix} power", s, f"{prefix} power {p:.0f}", power=p, signal=sig)


def _q_smart(row: pd.Series) -> AgentResult:
    sm = _num(row, "smart_money_score")
    s = min(100, sm * 1.1) if sm else 50
    return _ok("q_smart_money", "quant", "Smart money score", s, f"Smart money {sm:.0f}", sm=sm)


def _q_analog(row: pd.Series) -> AgentResult:
    h = _num(row, "analog_hit_rate")
    s = 40 + h * 60
    return _ok("q_analog", "quant", "Analog hit rate", s, f"Analog hit {h:.0%}", hit=h)


def _q_anomaly(row: pd.Series) -> AgentResult:
    flag = row.get("anomaly_flag") in (True, 1, "True")
    sc = _num(row, "anomaly_score")
    s = 35 if flag else 55 + min(20, sc * 10)
    sig: AgentSignal = "bearish" if flag else "neutral"
    return _ok("q_anomaly", "quant", "Anomaly detector", s, "Anomaly flagged" if flag else "Normal regime", signal=sig)


def _q_exp_ret(row: pd.Series) -> AgentResult:
    er = _num(row, "expected_return_10d")
    s = 50 + min(40, er * 4)
    return _ok("q_exp_return", "quant", "Expected 10D return", s, f"Exp return {er:+.1f}%", er=er)


def _q_conf(row: pd.Series) -> AgentResult:
    c = _num(row, "confidence", 0.5)
    s = 40 + c * 55
    return _ok("q_ml_conf", "quant", "ML confidence", s, f"Confidence {c:.0%}", conf=c)


def _q_vol_pipe(row: pd.Series, universe: pd.DataFrame | None) -> AgentResult:
    r = analyze_volume(row, universe)
    return _ok("q_vol_pipeline", "quant", "Volumetric pipeline", r["score"], r["notes"][0] if r.get("notes") else "Volume")


def _q_broker_pipe(sym: str, row: pd.Series, bp: pd.DataFrame) -> AgentResult:
    r = analyze_brokers(sym, row, bp)
    return _ok("q_broker_pipeline", "quant", "Broker quant step", r["score"], r["notes"][0] if r.get("notes") else "Brokers")


def _q_pa_pipe(row: pd.Series, panel: pd.DataFrame) -> AgentResult:
    r = detect_fair_value_gaps(row, panel)
    return _ok("q_pa_pipeline", "quant", "Price action pipeline", r["score"], r["notes"][0] if r.get("notes") else "PA")


def _q_mom_pipe(row: pd.Series) -> AgentResult:
    r = analyze_momentum(row)
    return _ok("q_momentum_pipeline", "quant", "Momentum pipeline", r["score"], r["notes"][0] if r.get("notes") else "Momentum")


def _close_series(ctx: AgentContext) -> pd.Series:
    sym = ctx.symbol
    if ctx.features is not None and not ctx.features.empty:
        px = build_price_series_from_features(ctx.features)
        sub = px[px["symbol"] == sym].sort_values("date")
        if len(sub) >= 2:
            return sub["close"]
    if not ctx.panel_sym.empty and "ltp" in ctx.panel_sym.columns:
        return pd.to_numeric(ctx.panel_sym["ltp"], errors="coerce").dropna()
    return pd.Series(dtype=float)


def _q_rsi(ctx: AgentContext) -> AgentResult:
    ta = analyze_price_series(_close_series(ctx))
    rsi_v = ta["rsi"]
    s = 55 + (rsi_v - 50) * 0.8
    sig: AgentSignal = "bullish" if 40 <= rsi_v <= 65 else ("bearish" if rsi_v >= 75 or rsi_v <= 25 else "neutral")
    return _ok("q_rsi", "quant", "RSI (14)", s, f"RSI {rsi_v:.1f} on {ta['bars']} bars", signal=sig, rsi=rsi_v)


def _q_roc(ctx: AgentContext) -> AgentResult:
    ta = analyze_price_series(_close_series(ctx))
    roc = ta["roc"]
    s = 50 + max(-25, min(35, roc * 2))
    sig: AgentSignal = "bullish" if roc > 3 else ("bearish" if roc < -3 else "neutral")
    return _ok("q_roc_momentum", "quant", "ROC momentum", s, f"ROC {roc:+.1f}%", signal=sig)


def _q_bb(ctx: AgentContext) -> AgentResult:
    ta = analyze_price_series(_close_series(ctx))
    bb = ta["bb_pct_b"]
    s = 45 + bb * 35
    sig: AgentSignal = "bullish" if 0.2 <= bb <= 0.55 else ("bearish" if bb > 0.95 else "neutral")
    return _ok("q_bollinger", "quant", "Bollinger %B", s, f"%B {bb:.2f}", signal=sig)


def _q_turn_z(row: pd.Series, ctx: AgentContext) -> AgentResult:
    turn = _num(row, "daily_turnover_lac")
    if ctx.features is not None and "daily_turnover_lac" in ctx.features.columns:
        hist = ctx.features[ctx.features["symbol"].astype(str).str.upper() == ctx.symbol]["daily_turnover_lac"]
    else:
        hist = pd.Series([turn])
    z = z_score_reversion(hist, window=min(20, max(3, len(hist))))
    s = 52 - abs(z) * 8 + (15 if z < -1 else 0)
    sig: AgentSignal = "bullish" if z < -1.2 else ("bearish" if z > 2 else "neutral")
    return _ok("q_turnover_zscore", "quant", "Turnover z-score", s, f"Z {z:+.2f} vs history", signal=sig)


def _f_sharpe(ctx: AgentContext) -> AgentResult:
    ta = analyze_price_series(_close_series(ctx))
    sh = ta["sharpe"]
    s = 50 + max(-20, min(40, sh * 15))
    sig: AgentSignal = "bullish" if sh > 0.5 else ("bearish" if sh < -0.5 else "neutral")
    return _ok("f_sharpe_proxy", "financial", "Sharpe proxy", s, f"Sharpe {sh:.2f}", signal=sig)


def _f_max_dd(ctx: AgentContext) -> AgentResult:
    ta = analyze_price_series(_close_series(ctx))
    dd = ta["max_dd"]
    s = max(20, 100 + dd * 1.2)
    sig: AgentSignal = "bearish" if dd < -12 else "neutral"
    return _ok("f_max_drawdown", "financial", "Max drawdown", s, f"Max DD {dd:.1f}%", signal=sig)


def _f_hhi(ctx: AgentContext) -> AgentResult:
    sym, bp = ctx.symbol, ctx.broker_panel
    if bp.empty:
        return _skip("f_herfindahl", "financial", "Broker HHI", "No broker panel")
    sub = bp[bp["symbol"].astype(str).str.upper() == sym]
    if sub.empty:
        return _skip("f_herfindahl", "financial", "Broker HHI", "No rows")
    act = pd.to_numeric(sub.get("buy_qty", 0), errors="coerce").fillna(0) + pd.to_numeric(sub.get("sell_qty", 0), errors="coerce").fillna(0)
    hhi = herfindahl_index(act)
    s = max(30, 85 - hhi * 0.5)
    return _ok("f_herfindahl", "financial", "Broker concentration HHI", s, f"HHI {hhi:.0f} (lower = broader desk)", hhi=hhi)


# --- Financial executors ---


def run_financial_agent(agent_id: str, ctx: AgentContext) -> AgentResult:
    row = ctx.row
    sym = ctx.symbol
    handlers = {
        "f_ltp": lambda: _f_ltp(row),
        "f_return_10d": lambda: _f_ret(row),
        "f_broker_pressure": lambda: _f_bp(row),
        "f_liquidity_turn": lambda: _f_liq_turn(row),
        "f_liquidity_qty": lambda: _f_liq_qty(row),
        "f_risk_drs": lambda: _f_risk_drs(row),
        "f_risk_invalidate": lambda: _f_inv(row),
        "f_capital_eff": lambda: _f_cap_eff(row),
        "f_dist_3y": lambda: _f_horizon_net(row, "dist_3Y"),
        "f_dist_1m": lambda: _f_horizon_net(row, "dist_1M"),
        "f_dist_3m": lambda: _f_horizon_net(row, "dist_3M"),
        "f_acc_1w": lambda: _f_horizon_net(row, "acc_1W"),
        "f_acc_1m": lambda: _f_horizon_net(row, "acc_1M"),
        "f_float_z_fin": lambda: _f_float_z(row),
        "f_turnover_rank": lambda: _f_turn_rank(row),
        "f_early_pick": lambda: _f_pick(row),
        "f_tier_bull": lambda: _f_tier(row),
        "f_llm_p_long": lambda: _f_llm_p(row),
        "f_price_demand": lambda: _f_px_demand(row),
        "f_price_supply": lambda: _f_px_supply(row),
        "f_anomaly_fin": lambda: _f_anom_fin(row),
        "f_drawdown_guard": lambda: _f_dd(row),
        "f_turnover_eff": lambda: _f_turn_eff(row),
        "f_net_1d": lambda: _f_net_1d(row),
        "f_hold_score": lambda: _f_hold(row),
        "f_entry_timing": lambda: _f_entry(row),
        "f_exit_risk": lambda: _f_exit(row),
        "f_portfolio_fit": lambda: _f_portfolio(row, ctx.universe),
        "f_margin_safety": lambda: _f_margin(row),
        "f_stress_pass": lambda: _f_stress(row),
        "f_circular_fin": lambda: _f_circ(row, ctx.broker_panel, sym),
        "f_wash_fin": lambda: _f_wash(row, ctx.broker_panel, sym),
        "f_sharpe_proxy": lambda: _f_sharpe(ctx),
        "f_max_drawdown": lambda: _f_max_dd(ctx),
        "f_herfindahl": lambda: _f_hhi(ctx),
    }
    fn = handlers.get(agent_id)
    if fn is None:
        return _skip(agent_id, "financial", agent_id, "unknown financial agent")
    return fn()


def _f_ltp(row: pd.Series) -> AgentResult:
    ltp = _num(row, "ltp")
    s = 55 if ltp > 0 else 30
    return _ok("f_ltp", "financial", "Price availability", s, f"LTP Rs {ltp:.2f}" if ltp else "LTP missing")


def _f_ret(row: pd.Series) -> AgentResult:
    er = _num(row, "expected_return_10d")
    s = 52 + min(35, er * 3)
    return _ok("f_return_10d", "financial", "Forward return outlook", s, f"E[R10d] {er:+.1f}%")


def _f_bp(row: pd.Series) -> AgentResult:
    bp = _num(row, "broker_pressure")
    s = 45 + min(45, bp * 0.9)
    return _ok("f_broker_pressure", "financial", "Broker pressure (fin)", s, f"Pressure {bp:.0f}")


def _f_liq_turn(row: pd.Series) -> AgentResult:
    t = _num(row, "daily_turnover_lac")
    s = 38 + min(50, t / 6)
    return _ok("f_liquidity_turn", "financial", "Liquidity (turnover)", s, f"{t:,.0f} Lac liquidity")


def _f_liq_qty(row: pd.Series) -> AgentResult:
    q = _num(row, "daily_volume")
    s = 40 + min(45, q / 3000)
    return _ok("f_liquidity_qty", "financial", "Liquidity (volume)", s, f"Qty liquidity {q:,.0f}")


def _f_risk_drs(row: pd.Series) -> AgentResult:
    d = _num(row, "distribution_risk_score", 50)
    s = max(15, 95 - d)
    sig: AgentSignal = "bearish" if d >= 80 else "neutral"
    return _ok("f_risk_drs", "financial", "Distribution risk", s, f"DRS {d:.0f}", signal=sig)


def _f_inv(row: pd.Series) -> AgentResult:
    tier = str(row.get("signal_tier", ""))
    s = 25 if tier == "Invalidated" else 70
    sig: AgentSignal = "bearish" if tier == "Invalidated" else "bullish"
    return _ok("f_risk_invalidate", "financial", "Invalidation guard", s, f"Tier {tier}", signal=sig)


def _f_cap_eff(row: pd.Series) -> AgentResult:
    t, ltp = _num(row, "daily_turnover_lac"), _num(row, "ltp")
    eff = t / (ltp + 1) * 100 if ltp else t
    s = 42 + min(45, eff / 2)
    return _ok("f_capital_eff", "financial", "Capital turnover efficiency", s, f"Eff {eff:.1f}")


def _f_horizon_net(row: pd.Series, prefix: str) -> AgentResult:
    n = _num(row, f"{prefix}_net_amount")
    s = 52 + max(-30, min(35, n / 50))
    sig: AgentSignal = "bullish" if n > 0 else ("bearish" if n < -20 else "neutral")
    return _ok(f"f_{prefix.lower()}", "financial", f"{prefix} net flow", s, f"Net {n:+.1f}", signal=sig)


def _f_float_z(row: pd.Series) -> AgentResult:
    z = _num(row, "float_turnover_zscore") or _num(row, "float_turnover_zscore_hv")
    s = 48 + min(35, abs(z) * 12)
    return _ok("f_float_z_fin", "financial", "Float activity (fin)", s, f"Z {z:.2f}")


def _f_turn_rank(row: pd.Series) -> AgentResult:
    r = _num(row, "turnover_rank") or _num(row, "volume_rank")
    s = max(30, 100 - r * 0.8) if r else 50
    return _ok("f_turnover_rank", "financial", "Turnover rank", s, f"Rank #{r:.0f}")


def _f_pick(row: pd.Series) -> AgentResult:
    p = _num(row, "early_pick_rank")
    s = max(35, 100 - p * 0.5) if p else 50
    return _ok("f_early_pick", "financial", "Early pick rank", s, f"Pick #{p:.0f}")


def _f_tier(row: pd.Series) -> AgentResult:
    tier = str(row.get("signal_tier", "Neutral"))
    m = {"Confirmed": 85, "Trigger": 75, "Setup": 65, "Watch": 55, "Neutral": 48, "Invalidated": 20}
    s = m.get(tier, 50)
    sig = "bullish" if tier in ("Confirmed", "Trigger", "Setup") else ("bearish" if tier == "Invalidated" else "neutral")
    return _ok("f_tier_bull", "financial", "Signal tier (fin)", s, tier, signal=sig)


def _f_llm_p(row: pd.Series) -> AgentResult:
    lp = row.get("llm_p_long")
    if lp is None or (isinstance(lp, float) and pd.isna(lp)):
        return _skip("f_llm_p_long", "financial", "Cached LLM P(long)", "No cached LLM score")
    p = float(lp)
    s = p * 100 if p <= 1 else min(100, p)
    return _ok("f_llm_p_long", "financial", "Cached LLM P(long)", s, f"LLM P {p:.0%}" if p <= 1 else f"LLM {p:.0f}%")


def _f_px_demand(row: pd.Series) -> AgentResult:
    ltp, d = _num(row, "ltp"), row.get("tech_demand_zone")
    if not ltp or pd.isna(d):
        return _skip("f_price_demand", "financial", "Price vs demand", "No demand zone")
    dist = (ltp - float(d)) / ltp * 100
    s = 70 if -3 <= dist <= 5 else 45
    return _ok("f_price_demand", "financial", "Price vs demand zone", s, f"{dist:+.1f}% from demand")


def _f_px_supply(row: pd.Series) -> AgentResult:
    ltp, szone = _num(row, "ltp"), row.get("tech_supply_zone")
    if not ltp or pd.isna(szone):
        return _skip("f_price_supply", "financial", "Price vs supply", "No supply zone")
    dist = (float(szone) - ltp) / ltp * 100
    s = 35 if dist < 5 else 55
    sig: AgentSignal = "bearish" if dist < 4 else "neutral"
    return _ok("f_price_supply", "financial", "Price vs supply zone", s, f"Supply {dist:.1f}% away", signal=sig)


def _f_anom_fin(row: pd.Series) -> AgentResult:
    r = _q_anomaly(row)
    return _ok("f_anomaly_fin", "financial", "Anomaly (financial view)", r.score, r.summary, signal=r.signal)


def _f_dd(row: pd.Series) -> AgentResult:
    drs = _num(row, "distribution_risk_score", 50)
    s = 80 if drs < 50 else (50 if drs < 70 else 30)
    return _ok("f_drawdown_guard", "financial", "Drawdown guard", s, "Low dist risk" if drs < 50 else "Elevated dist risk")


def _f_turn_eff(row: pd.Series) -> AgentResult:
    r = _f_cap_eff(row)
    return _ok("f_turnover_eff", "financial", "Turnover efficiency", r.score, r.summary, signal=r.signal)


def _f_net_1d(row: pd.Series) -> AgentResult:
    n = _num(row, "dist_1D_net_amount") or _num(row, "acc_1D_net_amount")
    s = 50 + max(-25, min(30, n / 30))
    return _ok("f_net_1d", "financial", "1D net flow", s, f"Net {n:+.1f} Lac")


def _f_hold(row: pd.Series) -> AgentResult:
    cfg = get_signal_config()
    p, ems, _ = effective_scores(row, cfg)
    s = (p * 50 + ems * 0.35)
    return _ok("f_hold_score", "financial", "Hold quality score", s, f"Hold bias {s:.0f}/100")


def _f_entry(row: pd.Series) -> AgentResult:
    sh = row.get("pattern_dist_shakeout") or row.get("dist_shakeout_flag")
    r = _num(row, "early_rank_score")
    s = 68 if sh in (True, 1) else 45 + r * 40
    return _ok("f_entry_timing", "financial", "Entry timing", s, "Shakeout entry window" if sh else f"Rank-based timing {r:.2f}")


def _f_exit(row: pd.Series) -> AgentResult:
    d = _num(row, "distribution_risk_score", 50)
    s = max(20, 100 - d * 0.9)
    sig: AgentSignal = "bearish" if d >= 75 else "neutral"
    return _ok("f_exit_risk", "financial", "Exit risk", s, f"Exit risk inverse {s:.0f}", signal=sig)


def _f_portfolio(row: pd.Series, universe: pd.DataFrame | None) -> AgentResult:
    if universe is None or universe.empty:
        return _skip("f_portfolio_fit", "financial", "Portfolio fit", "No universe")
    sym = str(row.get("symbol", "")).upper()
    in_uni = sym in universe["symbol"].astype(str).str.upper().values
    s = 62 if in_uni else 40
    return _ok("f_portfolio_fit", "financial", "High-volume universe fit", s, "In top turnover universe" if in_uni else "Outside scanner universe")


def _f_margin(row: pd.Series) -> AgentResult:
    t = _num(row, "daily_turnover_lac")
    s = 50 + min(35, t / 15)
    return _ok("f_margin_safety", "financial", "Liquidity margin safety", s, "Adequate float liquidity" if t >= 80 else "Thin — size down")


def _f_stress(row: pd.Series) -> AgentResult:
    tier = str(row.get("signal_tier", ""))
    circ = row.get("circular_confirmed") in (True, 1)
    s = 35 if circ or tier == "Invalidated" else 65
    return _ok("f_stress_pass", "financial", "Stress test", s, "Failed stress" if s < 50 else "Passed basic stress")


def _f_circ(row: pd.Series, bp: pd.DataFrame, sym: str) -> AgentResult:
    desk = analyze_symbol_brokers(sym, bp) if not bp.empty else {}
    risk = float(desk.get("circular_risk") or row.get("circular_risk") or 0)
    s = max(20, 100 - risk)
    sig: AgentSignal = "bearish" if desk.get("circular_confirmed") else "neutral"
    return _ok("f_circular_fin", "financial", "Circular risk (fin)", s, f"Risk {risk:.0f}%", signal=sig)


def _f_wash(row: pd.Series, bp: pd.DataFrame, sym: str) -> AgentResult:
    desk = analyze_symbol_brokers(sym, bp) if not bp.empty else {}
    w = float(desk.get("wash_score") or row.get("wash_score") or 0)
    s = max(25, 100 - w * 0.7)
    return _ok("f_wash_fin", "financial", "Wash score (fin)", s, f"Wash {w:.1f}")


# --- Broker executors ---


def run_broker_agent(agent_id: str, ctx: AgentContext) -> AgentResult:
    sym, bp = ctx.symbol, ctx.broker_panel
    if bp.empty and not agent_id.startswith("b_meta"):
        return _skip(agent_id, "broker", agent_id, "Empty broker panel")

    if agent_id.startswith("b_broker_"):
        bid = agent_id.replace("b_broker_", "")
        return _broker_single(sym, bid, bp)

    handlers = {
        "b_circular_risk": lambda: _b_circ(sym, bp),
        "b_circular_flag": lambda: _b_circ_flag(sym, bp),
        "b_wash": lambda: _b_wash(sym, bp),
        "b_directional": lambda: _b_dir(sym, bp),
        "b_reciprocal": lambda: _b_recip(sym, bp),
        "b_activity": lambda: _b_activity(sym, bp),
        "b_top_net": lambda: _b_top_net(sym, bp),
        "b_pressure": lambda: _b_pressure(ctx.row),
        "b_conviction_max": lambda: _b_conv_max(sym, bp),
        "b_conviction_avg": lambda: _b_conv_avg(sym, bp),
        "b_buy_dom": lambda: _b_buy_dom(sym, bp),
        "b_sell_dom": lambda: _b_sell_dom(sym, bp),
        "b_churn": lambda: _b_churn(sym, bp),
        "b_net_skew": lambda: _b_net_skew(sym, bp),
        "b_dispersion": lambda: _b_dispersion(sym, bp),
        "b_watch_aggregate": lambda: _b_watch_agg(sym, bp),
        "b_market_top10": lambda: _b_top10_presence(sym, bp),
    }
    fn = handlers.get(agent_id)
    if fn is None:
        return _skip(agent_id, "broker", agent_id, "unknown broker agent")
    return fn()


def _broker_slice(sym: str, bp: pd.DataFrame, broker_id: str) -> pd.DataFrame:
    sub = bp[(bp["symbol"].astype(str).str.upper() == sym) & (bp["broker_id"].astype(str) == str(broker_id))]
    if "horizon" in sub.columns:
        sub = sub[sub["horizon"].isin(("1D", "2D", "3D", "4D", "1W"))]
    return sub


def _broker_single(sym: str, broker_id: str, bp: pd.DataFrame) -> AgentResult:
    sub = _broker_slice(sym, bp, broker_id)
    if sub.empty:
        return _skip(f"b_broker_{broker_id}", "broker", f"Broker {broker_id}", "No 1D activity")
    sym_act = pd.to_numeric(bp[bp["symbol"].astype(str).str.upper() == sym].get("buy_qty", 0), errors="coerce").fillna(0).sum()
    sym_act += pd.to_numeric(bp[bp["symbol"].astype(str).str.upper() == sym].get("sell_qty", 0), errors="coerce").fillna(0).sum()
    row = analyze_broker_row(broker_id, sub, sym_act)
    s = row["conviction_score"]
    sig: AgentSignal = row.get("signal", "neutral")
    if sig not in ("bullish", "bearish", "neutral"):
        sig = "bullish" if row.get("bias") in ("acc_buy", "absorption", "buy") else (
            "bearish" if row.get("bias") in ("dist_heavy", "acc_sell", "sell") else "neutral"
        )
    return _ok(
        f"b_broker_{broker_id}",
        "broker",
        f"Broker {broker_id} desk",
        s,
        row.get("flow_label", row.get("bias", ""))[:120],
        signal=sig,
        **row,
    )


def _desk(sym: str, bp: pd.DataFrame) -> dict:
    return analyze_symbol_brokers(sym, bp)


def _b_circ(sym: str, bp: pd.DataFrame) -> AgentResult:
    d = _desk(sym, bp)
    risk = float(d.get("circular_risk") or 0)
    s = max(15, 100 - risk)
    return _ok("b_circular_risk", "broker", "Circular risk", s, f"Risk {risk:.0f}%")


def _b_circ_flag(sym: str, bp: pd.DataFrame) -> AgentResult:
    d = _desk(sym, bp)
    s = 25 if d.get("circular_confirmed") else (40 if d.get("circular_flag") else 70)
    sig: AgentSignal = "bearish" if d.get("circular_confirmed") else "neutral"
    return _ok("b_circular_flag", "broker", "Circular flag", s, "Confirmed circular" if d.get("circular_confirmed") else "No confirmed circular", signal=sig)


def _b_wash(sym: str, bp: pd.DataFrame) -> AgentResult:
    d = _desk(sym, bp)
    w = float(d.get("wash_score") or 0)
    s = max(20, 100 - w * 0.75)
    return _ok("b_wash", "broker", "Wash / churn", s, f"Wash {w:.1f}")


def _b_dir(sym: str, bp: pd.DataFrame) -> AgentResult:
    d = _desk(sym, bp)
    dp = float(d.get("directional_pct") or 0)
    s = 40 + min(50, dp * 0.55)
    return _ok("b_directional", "broker", "Directional flow %", s, f"Directional {dp:.1f}%")


def _b_recip(sym: str, bp: pd.DataFrame) -> AgentResult:
    d = _desk(sym, bp)
    r = int(d.get("reciprocal_brokers") or 0)
    s = max(30, 85 - r * 12)
    sig: AgentSignal = "bearish" if r >= 3 else "neutral"
    return _ok("b_reciprocal", "broker", "Reciprocal brokers", s, f"{r} reciprocal desks", signal=sig)


def _b_activity(sym: str, bp: pd.DataFrame) -> AgentResult:
    d = _desk(sym, bp)
    a = float(d.get("symbol_activity_qty") or 0)
    s = 40 + min(50, a / 2000)
    return _ok("b_activity", "broker", "Symbol broker activity", s, f"Activity qty {a:,.0f}")


def _b_top_net(sym: str, bp: pd.DataFrame) -> AgentResult:
    d = _desk(sym, bp)
    net = float(d.get("top_broker_net_lac") or 0)
    s = 52 + max(-30, min(35, net * 2))
    sig: AgentSignal = "bullish" if net > 5 else ("bearish" if net < -5 else "neutral")
    return _ok("b_top_net", "broker", "Top broker net Lac", s, f"Net {net:+.1f} Lac", signal=sig)


def _b_pressure(row: pd.Series) -> AgentResult:
    bp = _num(row, "broker_pressure")
    s = 45 + min(45, bp * 0.85)
    return _ok("b_pressure", "broker", "Broker pressure index", s, f"Pressure {bp:.0f}")


def _b_conv_max(sym: str, bp: pd.DataFrame) -> AgentResult:
    tbl = symbol_top_brokers_table(sym, bp, top_n=10)
    if tbl.empty:
        return _skip("b_conviction_max", "broker", "Max conviction", "No table")
    s = float(tbl["conviction_score"].max())
    return _ok("b_conviction_max", "broker", "Max broker conviction", s, f"Max {s:.0f}")


def _b_conv_avg(sym: str, bp: pd.DataFrame) -> AgentResult:
    tbl = symbol_top_brokers_table(sym, bp, top_n=10)
    if tbl.empty:
        return _skip("b_conviction_avg", "broker", "Avg conviction", "No table")
    s = float(tbl["conviction_score"].mean())
    return _ok("b_conviction_avg", "broker", "Avg broker conviction", s, f"Avg {s:.0f}")


def _b_buy_dom(sym: str, bp: pd.DataFrame) -> AgentResult:
    tbl = symbol_top_brokers_table(sym, bp, top_n=10)
    if tbl.empty:
        return _skip("b_buy_dom", "broker", "Buy dominance", "No table")
    bull = tbl["bias"].isin(["acc_buy", "absorption", "buy"])
    buys = int(bull.sum())
    s = 45 + buys * 8
    return _ok("b_buy_dom", "broker", "Bullish desk flow", s, f"{buys}/10 desks acc_buy or absorption")


def _b_sell_dom(sym: str, bp: pd.DataFrame) -> AgentResult:
    tbl = symbol_top_brokers_table(sym, bp, top_n=10)
    if tbl.empty:
        return _skip("b_sell_dom", "broker", "Sell dominance", "No table")
    bear = tbl["bias"].isin(["dist_heavy", "acc_sell", "sell"])
    sells = int(bear.sum())
    s = max(25, 90 - sells * 10)
    sig: AgentSignal = "bearish" if sells >= 6 else "neutral"
    return _ok("b_sell_dom", "broker", "Distribution-heavy desks", s, f"{sells}/10 dist-heavy (floorsheet sell cols)", signal=sig)


def _b_churn(sym: str, bp: pd.DataFrame) -> AgentResult:
    d = _desk(sym, bp)
    w = float(d.get("wash_score") or 0)
    s = max(25, 100 - w)
    return _ok("b_churn", "broker", "Two-sided churn", s, f"Churn/wash {w:.1f}")


def _b_net_skew(sym: str, bp: pd.DataFrame) -> AgentResult:
    tbl = symbol_top_brokers_table(sym, bp, top_n=10)
    if tbl.empty:
        return _skip("b_net_skew", "broker", "Net amount skew", "No table")
    net = pd.to_numeric(tbl["net_amount_lac"], errors="coerce").fillna(0)
    skew = float(net.sum())
    s = 52 + max(-35, min(38, skew * 3))
    sig: AgentSignal = "bullish" if skew > 3 else ("bearish" if skew < -3 else "neutral")
    return _ok("b_net_skew", "broker", "Net Lac skew (top10)", s, f"Skew {skew:+.1f}", signal=sig)


def _b_dispersion(sym: str, bp: pd.DataFrame) -> AgentResult:
    tbl = symbol_top_brokers_table(sym, bp, top_n=10)
    if tbl.empty or len(tbl) < 2:
        return _skip("b_dispersion", "broker", "Broker dispersion", "Insufficient brokers")
    conv = pd.to_numeric(tbl["conviction_score"], errors="coerce")
    disp = float(conv.std())
    s = 55 + min(30, disp)
    return _ok("b_dispersion", "broker", "Conviction dispersion", s, f"Std {disp:.1f}")


def _b_watch_agg(sym: str, bp: pd.DataFrame) -> AgentResult:
    from backend.config import load_yaml_config

    watch = [str(int(b)) for b in load_yaml_config("settings.yaml").get("brokers", {}).get("watch_list", [])]
    scores = []
    for bid in watch:
        sub = _broker_slice(sym, bp, bid)
        if not sub.empty:
            sym_act = 1.0
            scores.append(analyze_broker_row(bid, sub, sym_act)["conviction_score"])
    if not scores:
        return _skip("b_watch_aggregate", "broker", "Watch-list aggregate", "No watch broker rows")
    s = sum(scores) / len(scores)
    return _ok("b_watch_aggregate", "broker", "Watch-list brokers (58,49…)", s, f"Avg conviction {s:.0f} over {len(scores)} desks")


def _b_top10_presence(sym: str, bp: pd.DataFrame) -> AgentResult:
    top = discover_top_brokers(bp, top_n=10)
    tbl = symbol_top_brokers_table(sym, bp, top_n=10)
    active = int((tbl["activity_qty"] > 0).sum()) if not tbl.empty else 0
    s = 40 + active * 6
    return _ok("b_market_top10", "broker", "Market top-10 presence", s, f"{active}/10 active · IDs {','.join(top[:5])}…")


# --- LLM rule-analyst agents (no per-agent API; template reasoning) ---


def run_llm_agent(agent_id: str, ctx: AgentContext) -> AgentResult:
    row = ctx.row
    templates = {
        "llm_trend": ("Trend narrative", _llm_trend),
        "llm_risk": ("Risk narrative", _llm_risk),
        "llm_broker": ("Broker narrative", _llm_broker),
        "llm_volume": ("Volume narrative", _llm_volume),
        "llm_distribution": ("Distribution narrative", _llm_dist),
        "llm_accumulation": ("Accumulation narrative", _llm_acc),
        "llm_shakeout": ("Shakeout narrative", _llm_shake),
        "llm_circular": ("Circular narrative", _llm_circ_n),
        "llm_entry": ("Entry narrative", _llm_entry),
        "llm_exit": ("Exit narrative", _llm_exit),
        "llm_tier": ("Tier narrative", _llm_tier),
        "llm_p_long": ("P(long) narrative", _llm_p),
        "llm_ems": ("EMS narrative", _llm_ems),
        "llm_analog": ("Analog narrative", _llm_analog),
        "llm_fvg": ("FVG narrative", _llm_fvg),
        "llm_orderblock": ("Order block narrative", _llm_ob),
        "llm_turnover": ("Turnover narrative", _llm_turn),
        "llm_watchlist": ("Watchlist fit", _llm_watch),
        "llm_long_bias": ("Long bias consensus", _llm_long),
        "llm_short_bias": ("Short/avoid bias", _llm_short),
        "llm_hold": ("Hold guidance", _llm_hold_n),
        "llm_verdict": ("Composite LLM verdict", _llm_verdict),
        "llm_momentum": ("Momentum story", _llm_mom_story),
        "llm_financial": ("Financial health story", _llm_fin_story),
        "llm_quant_sync": ("Quant sync check", _llm_quant_sync),
    }
    if agent_id not in templates:
        return _skip(agent_id, "llm", agent_id, "unknown LLM agent")
    name, fn = templates[agent_id]
    text, s, sig = fn(row, ctx)
    return _ok(agent_id, "llm", name, s, text, signal=sig)


def _llm_trend(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    cfg = get_signal_config()
    p, ems, _ = effective_scores(row, cfg)
    s = p * 55 + ems * 0.3
    sig = _sig(s)
    return f"Trend: effective P(long) {p:.0%}, EMS {ems:.0f} → {'up bias' if sig == 'bullish' else 'mixed/weak'}.", s, sig


def _llm_risk(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    d = _num(row, "distribution_risk_score", 50)
    s = max(20, 100 - d)
    sig: AgentSignal = "bearish" if d >= 75 else "neutral"
    return f"Risk: distribution score {d:.0f} — {'elevated exit risk' if d >= 75 else 'manageable'}.", s, sig


def _llm_broker(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    bp = _num(row, "broker_pressure")
    s = 45 + min(45, bp * 0.8)
    return f"Brokers: pressure {bp:.0f}/100 — {'accumulation skew' if bp >= 20 else 'neutral desk'}.", s, _sig(s)


def _llm_volume(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    t = _num(row, "daily_turnover_lac")
    s = 40 + min(50, t / 7)
    return f"Volume: {t:,.0f} Lac 1D — {'institutional interest' if t >= 150 else 'needs more liquidity'}.", s, _sig(s)


def _llm_dist(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    p3 = _num(row, "dist_3Y_power_score", 2)
    s = max(25, 90 - p3 * 15)
    sig: AgentSignal = "bearish" if p3 >= 3 else "neutral"
    return f"Distribution ladder 3Y power {p3:.0f} — {'heavy exit pressure' if p3 >= 3 else 'light'}.", s, sig


def _llm_acc(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    a1 = _num(row, "acc_1D_power_score", 2)
    s = 42 + a1 * 18
    return f"Accumulation 1D power {a1:.0f} — {'building' if a1 >= 3 else 'weak acc'}.", s, _sig(s)


def _llm_shake(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    sh = row.get("pattern_dist_shakeout") or row.get("dist_shakeout_flag")
    active = sh in (True, 1)
    s = 75 if active else 45
    return ("Shakeout: washout-of-weak-hands pattern — watch for reversal long." if active else "No shakeout — distribution may continue."), s, "bullish" if active else "neutral"


def _llm_circ_n(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    if ctx.broker_panel.empty:
        return "Circular: no broker data.", 50, "skip"
    d = analyze_symbol_brokers(ctx.symbol, ctx.broker_panel)
    s = max(20, 100 - float(d.get("circular_risk") or 0))
    sig: AgentSignal = "bearish" if d.get("circular_confirmed") else "neutral"
    return f"Circular: {d.get('verdict', 'n/a')} wash {d.get('wash_score', 0):.0f}.", s, sig


def _llm_entry(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    return _llm_shake(row, ctx)


def _llm_exit(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    return _llm_risk(row, ctx)


def _llm_tier(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    tier = str(row.get("signal_tier", "Neutral"))
    m = {"Confirmed": 82, "Trigger": 74, "Setup": 66, "Watch": 56, "Neutral": 48, "Invalidated": 22}
    s = m.get(tier, 50)
    sig = "bullish" if tier in ("Confirmed", "Trigger") else ("bearish" if tier == "Invalidated" else "neutral")
    return f"Signal tier **{tier}** — align size with tier.", s, sig


def _llm_p(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    return _llm_trend(row, ctx)


def _llm_ems(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    cfg = get_signal_config()
    _, ems, _ = effective_scores(row, cfg)
    s = min(100, ems * 1.3)
    return f"EMS {ems:.0f}: {'strong early momentum' if ems >= 35 else 'still building'}.", s, _sig(s)


def _llm_analog(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    h = _num(row, "analog_hit_rate")
    s = 42 + h * 50
    return f"Historical analog hit rate {h:.0%}.", s, _sig(s)


def _llm_fvg(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    pa = detect_fair_value_gaps(row, ctx.panel_sym)
    s = pa["score"]
    return f"FVG: {'bull gap' if pa.get('bullish_fvg') else 'bear gap' if pa.get('bearish_fvg') else 'none'}.", s, _sig(s)


def _llm_ob(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    ob = detect_order_blocks(row)
    return f"Order blocks: {ob.get('order_block_bias', 'neutral')}.", ob["order_block_score"], _sig(ob["order_block_score"])


def _llm_turn(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    return _llm_volume(row, ctx)


def _llm_watch(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    note = str(row.get("llm_note") or "")
    s = 58 if note and note != "—" else 50
    return (f"Cached note: {note[:120]}" if note else "No cached LLM note — refresh scanner LLM."), s, "neutral"


def _llm_long(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    cfg = get_signal_config()
    p, _, _ = effective_scores(row, cfg)
    s = p * 100
    sig: AgentSignal = "bullish" if p >= 0.52 else "neutral"
    return f"Long bias agent: P(long) {p:.0%}.", s, sig


def _llm_short(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    d = _num(row, "distribution_risk_score", 50)
    s = min(100, d * 0.95)
    sig: AgentSignal = "bearish" if d >= 65 else "neutral"
    return f"Avoid-long agent: dist risk {d:.0f}.", s, sig


def _llm_hold_n(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    tier = str(row.get("signal_tier", ""))
    s = 60 if tier in ("Setup", "Trigger", "Watch") else 45
    return f"Hold: tier {tier} — {'monitor add-on levels' if tier == 'Watch' else 'review triggers'}.", s, "neutral"


def _llm_verdict(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    t, r = _llm_trend(row, ctx)
    rk, _ = _llm_risk(row, ctx)
    s = (t[1] + r[1]) / 2
    sig = _sig(s)
    return f"Verdict blend: {sig} composite {s:.0f}/100.", s, sig


def _llm_mom_story(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    r = analyze_momentum(row)
    return f"Momentum: {r['notes'][0] if r.get('notes') else 'n/a'}.", r["score"], _sig(r["score"])


def _llm_fin_story(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    er = _num(row, "expected_return_10d")
    s = 50 + min(35, er * 3)
    return f"Financial: expected 10D {er:+.1f}%.", s, _sig(s)


def _llm_quant_sync(row: pd.Series, ctx: AgentContext) -> tuple[str, float, AgentSignal]:
    v = analyze_volume(row, ctx.universe)
    m = analyze_momentum(row)
    s = (v["score"] + m["score"]) / 2
    return f"Quant sync: volume {v['score']} + momentum {m['score']}.", s, _sig(s)
