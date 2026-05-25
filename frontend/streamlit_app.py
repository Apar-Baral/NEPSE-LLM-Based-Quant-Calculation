"""NEPSE Quant Platform — Streamlit Dashboard"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Streamlit hot-reload can keep stale backend modules — drop them on each run
_STALE_PREFIXES = (
    "backend.scanner.",
    "backend.db.",
    "backend.backtest",
    "backend.llm",
    "backend.quant",
    "backend.signals",
    "backend.config",
    "backend.config_signals",
    "backend.agents",
    "backend.knowledge",
)
for _mod in list(sys.modules):
    if _mod in ("backend.scanner", "backend.config") or any(_mod.startswith(p) for p in _STALE_PREFIXES):
        del sys.modules[_mod]


def _load_scanner_helpers():
    """Load scanner functions without relying on package __init__ (Streamlit-safe)."""
    import importlib.util

    vu_path = ROOT / "backend" / "scanner" / "volume_universe.py"
    try:
        from backend.scanner.volume_universe import (
            get_latest_scanner_universe as _g,
            symbol_horizon_snapshot as _s,
        )

        return _g, _s
    except ImportError:
        spec = importlib.util.spec_from_file_location("nepse_volume_universe", vu_path)
        mod = importlib.util.module_from_spec(spec)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load scanner module from {vu_path}") from None
        spec.loader.exec_module(mod)
        return mod.get_latest_scanner_universe, mod.symbol_horizon_snapshot


get_latest_scanner_universe, symbol_horizon_snapshot = _load_scanner_helpers()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from backend.backtest.engine import (
    build_price_series_from_features,
    merge_ohlcv_sources,
    run_backtest,
)
from backend.config import ensure_dirs
from backend.config_signals import clear_config_cache

clear_config_cache()
import importlib

_analyst = importlib.import_module("backend.llm.analyst")
chat_query = _analyst.chat_query
generate_daily_brief = _analyst.generate_daily_brief
generate_symbol_report = _analyst.generate_symbol_report
llm_status = _analyst.llm_status
test_llm_connection = _analyst.test_llm_connection
from backend.models.trainer import compute_shap_values
from backend.alerts.webhooks import check_trigger_alerts
from backend.ingest.panel_utils import snapshot_panel_all_horizons
from backend.pipeline import run_pipeline
from backend.scanner.broker_insights import horizon_net_flow
from frontend.background_jobs import brief_status_sidebar, poll_brief_job, start_brief_generation
from frontend.data_access import _fresh_datastore, load_broker_panel_safe, save_broker_panel_safe
from frontend.display_config import TIER_COLORS, TIER_HELP, TIER_ORDER, format_scanner_table
from frontend.symbol_deep_dive import render_symbol_deep_dive
from frontend.ui_theme import hero, inject_global_css
from backend.scanner.symbol_lookup import all_tracked_symbols, enrich_symbol_row, filter_universe_by_symbol

st.set_page_config(page_title="NEPSE Quant", page_icon="📈", layout="wide")
ensure_dirs()
inject_global_css()

PAGES = [
    "Momentum Scanner",
    "Symbol Deep Dive",
    "Knowledge Graph",
    "Model Lab",
    "Daily Upload",
    "Backtest",
    "Data & Storage",
    "LLM Briefing",
    "Chat",
]

with st.sidebar:
    st.markdown("### 📈 NEPSE Quant")
    st.caption("Floorsheet · ML · LLM fusion")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    if st.button("Run Pipeline", use_container_width=True, type="primary"):
        with st.spinner("Running pipeline..."):
            result = run_pipeline()
            st.session_state["last_pipeline"] = result
            st.session_state.pop("scanner_df", None)
            st.success(f"Done: {result.get('symbols', 0)} symbols")
    if st.session_state.get("last_pipeline"):
        lp = st.session_state["last_pipeline"]
        st.caption(f"Last run: {lp.get('trigger_count', 0)} triggers · FS avg {lp.get('floorsheet_score_avg', '—')}")
        ps = lp.get("panel_sides") or {}
        st.caption(f"Panel: acc {ps.get('accumulation_rows', 0)} · dist {ps.get('distribution_rows', 0)} rows")
        inv = lp.get("data_inventory") or {}
        if inv.get("message"):
            st.caption(inv["message"][:120])
        if lp.get("multimodal_meta"):
            mm = lp["multimodal_meta"]
            st.caption(f"Multimodal: {mm.get('status', mm.get('reason', '—'))}")
    brief_status_sidebar()

def _safe_float(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _zone_from_panel(sym_panel: pd.DataFrame, col: str) -> float | None:
    if sym_panel.empty or col not in sym_panel.columns:
        return None
    vals = sym_panel[col].dropna()
    vals = vals[vals.apply(lambda x: _safe_float(x) is not None)]
    if vals.empty:
        return None
    return _safe_float(vals.iloc[0])


def _col(df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
    """Resolve column, including merge suffixes (_x/_y)."""
    if name in df.columns:
        return df[name]
    for suffix in ("_x", "_y"):
        if f"{name}{suffix}" in df.columns:
            return df[f"{name}{suffix}"]
    return pd.Series(default, index=df.index)


store = _fresh_datastore()
preds = store.load_predictions()
features = store.load_features()
panel = snapshot_panel_all_horizons(store.load_panel())
broker_panel = load_broker_panel_safe(store)
if broker_panel.empty:
    try:
        from backend.ingest.broker_loader import backfill_broker_panel_from_data

        broker_panel = backfill_broker_panel_from_data()
        if not broker_panel.empty:
            save_broker_panel_safe(store, broker_panel)
    except Exception:
        broker_panel = pd.DataFrame()
ohlcv = store.load_ohlcv()

poll_brief_job()


def _tier_color_scale():
    return {t: TIER_COLORS.get(t, "#888") for t in TIER_ORDER}

if page == "Momentum Scanner":
    hero(
        "Momentum Scanner",
        "Top 120 by 1D turnover · multimodal early-long score · broker desk 58/49",
    )
    if preds.empty:
        st.info("No predictions yet. Click **Run Pipeline** in the sidebar.")
    else:
        try:
            if st.session_state.get("scanner_df") is not None:
                df = st.session_state["scanner_df"]
            else:
                df = get_latest_scanner_universe(preds, panel=panel, broker_panel=broker_panel, top_n=120)
        except ImportError as exc:
            st.error(f"Scanner import error: {exc}")
            st.code(
                'cd "E:\\Major Project - Nepse Data LLM"\n'
                "Get-ChildItem -Recurse backend -Filter __pycache__ | Remove-Item -Recurse -Force\n"
                "python scripts\\test_scanner.py\n"
                "streamlit run frontend/streamlit_app.py",
                language="powershell",
            )
            st.stop()
        except Exception as exc:
            st.error(f"Scanner error: {exc}")
            st.stop()
        if df.empty:
            st.warning("No symbols in high-volume universe. Run **Run Pipeline** from the sidebar.")
            st.stop()
        latest = df["report_date"].max() if not df.empty else preds["report_date"].max()
        st.caption(
            f"Report date: **{latest.date()}** | Universe: **{len(df)}** symbols | "
            f"Ranked by **1D turnover (Lac)** then LLM + quant early-momentum score"
        )
        if st.button("Refresh LLM scores (max 12 new)", help="Calls DeepSeek in small batches — won't hang"):
            from backend.scanner.llm_scorer import score_universe_with_llm
            from backend.signals.universe_tiers import assign_universe_tiers

            prog = st.progress(0, text="Starting LLM refresh...")
            status = st.empty()

            def _progress(pct: float, msg: str) -> None:
                prog.progress(min(max(pct, 0.0), 1.0), text=msg)
                status.caption(msg)

            df = score_universe_with_llm(df, panel, progress_fn=_progress)
            df["signal_tier"] = assign_universe_tiers(df)
            st.session_state["scanner_df"] = df
            st.success("LLM scores updated (cached).")
            st.rerun()

        tiers_present = [t for t in TIER_ORDER if t in df["signal_tier"].unique()]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trigger+", int(df[df["signal_tier"].isin(["Trigger", "Confirmed"])].shape[0]))
        c2.metric("Setup", int(df[df["signal_tier"] == "Setup"].shape[0]))
        c3.metric("Watch", int(df[df["signal_tier"] == "Watch"].shape[0]))
        top_turn = df.sort_values("daily_turnover_lac", ascending=False).iloc[0] if "daily_turnover_lac" in df.columns and len(df) else df.iloc[0] if len(df) else None
        c4.metric("Top 1D Turnover", top_turn["symbol"] if top_turn is not None else "—")
        if len(df) and "early_pick_rank" in df.columns:
            _best = df.loc[df["early_pick_rank"].idxmin()]
            c5.metric("Top Early Pick", f"#{int(_best['early_pick_rank'])} {_best['symbol']}")
        else:
            c5.metric("Top Early Pick", df.iloc[0]["symbol"] if len(df) else "—")

        with st.expander("Signal tier guide", expanded=False):
            for t in tiers_present:
                st.markdown(f"**{t}** — {TIER_HELP.get(t, '')}")

        default_tiers = [t for t in ("Trigger", "Confirmed", "Setup", "Watch") if t in tiers_present]
        if not default_tiers:
            default_tiers = tiers_present
        c_search, c_tier = st.columns([1, 2])
        with c_search:
            search_q = st.text_input(
                "Search symbol",
                placeholder="e.g. NGPL, CITY — any tracked stock",
                key="scanner_symbol_search",
            )
        with c_tier:
            tier_filter = st.multiselect(
                "Filter by signal",
                tiers_present,
                default=default_tiers,
                help="Neutral = no edge; Invalidated = heavy distribution risk",
            )
        df = df[df["signal_tier"].isin(tier_filter)]
        if search_q:
            narrowed = filter_universe_by_symbol(df, search_q)
            if not narrowed.empty:
                df = narrowed
            else:
                extra = enrich_symbol_row(search_q, preds, panel, broker_panel, features=features)
                if not extra.empty:
                    df = pd.concat([extra, df], ignore_index=True).drop_duplicates(
                        subset=["symbol"], keep="first"
                    )
                    st.success(f"Showing **{extra.iloc[0]['symbol']}** (outside top 120 universe)")
                else:
                    st.warning(f"**{search_q.strip().upper()}** not in predictions — run pipeline after ingest.")

        tier_counts = df["signal_tier"].value_counts().to_dict() if "signal_tier" in df.columns else {}
        st.caption(f"Tier mix: {', '.join(f'{k}={v}' for k, v in sorted(tier_counts.items(), key=lambda x: -x[1]))}")

        display_cols = [
            "turnover_rank", "early_pick_rank", "symbol", "ltp", "daily_turnover_lac",
            "signal_tier", "early_rank_score", "p_long_momentum", "early_momentum_score",
            "floorsheet_momentum_score", "broker_pressure",
            "top_broker_ids", "circular_risk", "circular_flag", "circular_confirmed",
            "wash_score", "directional_pct", "reciprocal_brokers",
            "distribution_risk_score", "llm_p_long", "llm_note",
        ]
        if "turnover_rank" not in df.columns and "volume_rank" in df.columns:
            df["turnover_rank"] = df["volume_rank"]
        for canonical in ("early_momentum_score", "distribution_risk_score", "ltp", "early_rank_score", "broker_pressure"):
            if canonical not in df.columns:
                df[canonical] = _col(df, canonical)
        display_cols = [c for c in display_cols if c in df.columns]
        show = format_scanner_table(df[display_cols].sort_values("early_pick_rank", ascending=True))
        st.dataframe(show, use_container_width=True, hide_index=True, height=420)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Setup map", "Broker desk (top 10)", "Circular risk", "Tier breakdown"]
        )
        plot_df = df.copy()
        plot_df["early_rank_pct"] = (_col(plot_df, "early_rank_score") * 100).round(1)
        plot_df["broker_pressure"] = pd.to_numeric(_col(plot_df, "broker_pressure"), errors="coerce").fillna(0).round(1)
        if "circular_risk" in plot_df.columns:
            plot_df["circular_risk"] = pd.to_numeric(plot_df["circular_risk"], errors="coerce").fillna(0)
        p_long = _col(plot_df, "p_long_momentum")
        if "llm_p_long" in plot_df.columns:
            p_long = plot_df["llm_p_long"].fillna(p_long)
        plot_df["p_long_pct"] = (pd.to_numeric(p_long, errors="coerce").fillna(0) * 100).round(1)

        with tab1:
            st.markdown("**X** = LLM/quant P(long) · **Y** = composite early rank · size = 1D turnover")
            fig_map = px.scatter(
                plot_df,
                x="p_long_pct",
                y="early_rank_pct",
                color="signal_tier",
                color_discrete_map=_tier_color_scale(),
                hover_name="symbol",
                size="daily_turnover_lac" if "daily_turnover_lac" in plot_df.columns else None,
                labels={
                    "p_long_pct": "P(long 10D) %",
                    "early_rank_pct": "Early rank %",
                    "signal_tier": "Signal",
                    "daily_turnover_lac": "1D turnover (Lac)",
                },
                title="Actionable setups — probability vs early rank",
            )
            fig_map.update_layout(legend_title_text="Signal")
            st.plotly_chart(fig_map, use_container_width=True)

        with tab2:
            from backend.scanner.broker_top10 import discover_top_brokers
            from backend.scanner.broker_desk import top_broker_market_view

            top10 = discover_top_brokers(broker_panel, top_n=10)
            st.markdown(f"**Top 10 brokers (mathematical rank):** {', '.join(top10)} — **1D** net flow")
            if not broker_panel.empty:
                mkt = top_broker_market_view(broker_panel, horizon="1D")
                if not mkt.empty:
                    st.dataframe(mkt, use_container_width=True, hide_index=True)
                sym_pick = st.selectbox("Symbol broker breakdown", plot_df["symbol"].tolist())
                if sym_pick:
                    sub = broker_panel[
                        (broker_panel["symbol"] == sym_pick)
                        & (broker_panel["horizon"] == "1D")
                        & (broker_panel["broker_id"].astype(str).isin(top10))
                    ]
                    if not sub.empty:
                        fig_br = px.bar(
                            sub,
                            x="broker_id",
                            y="net_amount",
                            color="net_amount",
                            color_continuous_scale="RdYlGn",
                            title=f"{sym_pick} — top broker net (1D)",
                        )
                        st.plotly_chart(fig_br, use_container_width=True)
            else:
                st.info("Run **Run Pipeline** once to build broker-level data from CSV/Excel.")
            top_b = plot_df[plot_df["broker_pressure"] > 0].nlargest(20, "broker_pressure")
            if not top_b.empty:
                fig_b = px.bar(
                    top_b.sort_values("broker_pressure"),
                    x="broker_pressure",
                    y="symbol",
                    orientation="h",
                    color="signal_tier",
                    color_discrete_map=_tier_color_scale(),
                    title="Broker pressure score (1D–1W skew)",
                )
                fig_b.update_xaxes(range=[0, 100])
                st.plotly_chart(fig_b, use_container_width=True)

        with tab3:
            if "circular_confirmed" in plot_df.columns:
                circ = plot_df[plot_df["circular_confirmed"] == True]  # noqa: E712
                suspect = plot_df[(plot_df["circular_flag"] == True) & (~plot_df["circular_confirmed"])]  # noqa: E712
            else:
                circ = plot_df[plot_df.get("circular_flag", False) == True]  # noqa: E712
                suspect = pd.DataFrame()
            st.caption(
                "Flags only **top wash %ile** with high two-sided broker activity — not all high-volume names."
            )
            c_cf, c_cs = st.columns(2)
            c_cf.metric("Confirmed circular", int(circ.shape[0]) if not circ.empty else 0)
            c_cs.metric("Suspect only", int(suspect.shape[0]) if not suspect.empty else 0)
            show_circ = pd.concat([circ, suspect], ignore_index=True) if not circ.empty or not suspect.empty else pd.DataFrame()
            if show_circ.empty:
                st.info("No circular flags in current filter (strict calibration).")
            else:
                fig_c = px.scatter(
                    show_circ,
                    x="wash_score",
                    y="early_rank_pct",
                    color="circular_confirmed" if "circular_confirmed" in show_circ.columns else "signal_tier",
                    hover_name="symbol",
                    size="reciprocal_brokers" if "reciprocal_brokers" in show_circ.columns else None,
                    title="Wash score vs early rank (confirmed = strict)",
                )
                st.plotly_chart(fig_c, use_container_width=True)
                cols = [c for c in ("symbol", "wash_score", "directional_pct", "reciprocal_brokers", "circular_confirmed", "signal_tier") if c in show_circ.columns]
                st.dataframe(show_circ[cols].sort_values("wash_score", ascending=False), hide_index=True)

        with tab4:
            tier_counts = plot_df["signal_tier"].value_counts().reindex(tiers_present).dropna()
            fig_t = px.bar(
                x=tier_counts.index,
                y=tier_counts.values,
                color=tier_counts.index,
                color_discrete_map=_tier_color_scale(),
                labels={"x": "Signal", "y": "Count"},
                title="Signals in filtered universe",
            )
            fig_t.update_layout(showlegend=False)
            st.plotly_chart(fig_t, use_container_width=True)

        if st.button("Send Trigger Alerts"):
            sent = check_trigger_alerts(df)
            st.write(f"Alerts sent for: {sent or 'none (configure webhook)'}")

elif page == "Knowledge Graph":
    from frontend.knowledge_viz import render_knowledge_graph_page

    hero("Knowledge Graph", "Vector DB + logic relationships (MiroFish-style graph)")
    dive_sym = st.session_state.get("dive_select", "")
    render_knowledge_graph_page(symbol=dive_sym if dive_sym else None)

elif page == "Data & Storage":
    from frontend.knowledge_viz import storage_status

    hero("Data & Storage", "Everything persisted under data/ — SQLite + parquet + graph + vectors")
    stat = storage_status()
    st.dataframe(pd.DataFrame(stat["rows"]), hide_index=True, use_container_width=True)
    st.markdown(
        """
        **What gets saved when you run pipeline / upload:**
        - **SQLite** `data/nepse_quant.db` — mirror of tables
        - **Parquet** `data/processed/*.parquet` — source of truth (features, predictions, panels)
        - **Logic graph** `logic_graph.json` — symbol ↔ brokers ↔ quant steps
        - **Vectors** — ChromaDB folder `data/processed/chroma` + JSON fallback
        - **Models** `data/models/` — LightGBM + multimodal weights
        """
    )
    if st.button("Verify write test"):
        n = store.save_panel(panel.tail(1), "symbol_panel") if not panel.empty else 0
        st.success(f"Panel write OK ({n} rows sample). DB: {store.db_path}")

elif page == "Model Lab":
    hero("Model Lab", "Self-learning multimodal stack trained on your floorsheet data")
    from backend.models.multimodal.architecture import torch_available
    from backend.models.multimodal.interpret import attention_dataframe
    from backend.models.multimodal.train import train_multimodal
    from backend.config import MODELS_DIR
    import json

    m1, m2, m3 = st.columns(3)
    m1.metric("PyTorch", "Ready" if torch_available() else "Install torch")
    meta_path = MODELS_DIR / "multimodal_meta.json"
    mm_meta = {}
    if meta_path.exists():
        try:
            mm_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    m2.metric("Multimodal model", "Trained" if (MODELS_DIR / "multimodal.pt").exists() else "Not trained")
    m3.metric("Samples (last train)", mm_meta.get("samples", "—"))

    with st.expander("Architecture — key innovations", expanded=True):
        st.markdown(
            """
            | Component | What it does |
            |-----------|----------------|
            | **Multi-scale temporal CNN** | Dilated convolutions on 1D→1W acc/dist horizons |
            | **Learnable temporal decay** | Recent horizons weighted more (λ learned) |
            | **LLM semantic vector** | Cached narratives + pattern flags fused with quant |
            | **Cross-modal attention** | Quant ↔ semantic bidirectional MultiheadAttention |
            | **Graph propagation** | Broker overlap edges between symbols (58, 49, …) |
            | **Phase-aware loss** | Higher weight on accumulation / early phase samples |
            | **Interpretable attention** | Per-horizon weights exported after each predict |
            """
        )

    if st.button("Train multimodal model on stored data", type="primary"):
        labels = __import__("backend.models.labels", fromlist=["build_labels"]).build_labels(features, ohlcv)
        with st.spinner("Training CNN + graph + cross-modal fusion…"):
            result = train_multimodal(features, labels, broker_panel)
        st.json(result)
        if result.get("status") == "ok":
            st.success("Training complete. Run **Run Pipeline** to blend predictions.")

    attn_df = attention_dataframe()
    if not attn_df.empty:
        st.subheader("Temporal attention (why this horizon mattered)")
        fig_attn = px.bar(
            attn_df,
            x="horizon",
            y="temporal_weight",
            color="symbol",
            barmode="group",
            title="Learned horizon importance by symbol",
        )
        st.plotly_chart(fig_attn, use_container_width=True)
    else:
        st.info("Run pipeline after training to generate attention weights.")

elif page == "Symbol Deep Dive":
    hero(
        "Symbol Deep Dive",
        "5-step quant: volumetric · broker · order block/FVG · momentum · LLM verify",
    )
    if preds.empty:
        st.info("Run pipeline first.")
    else:
        all_syms = all_tracked_symbols(preds, features)
        sym_search = st.text_input("Search symbol", placeholder="NGPL, API…", key="dive_search").strip().upper()
        if sym_search and sym_search in all_syms:
            st.session_state["dive_select"] = sym_search
        if "dive_select" not in st.session_state or st.session_state["dive_select"] not in all_syms:
            st.session_state["dive_select"] = all_syms[0]
        sym = st.selectbox(
            f"Symbol ({len(all_syms)} tracked)",
            all_syms,
            key="dive_select",
        )
        universe_dive = (
            get_latest_scanner_universe(preds, panel=panel, broker_panel=broker_panel, top_n=120)
            if not preds.empty
            else pd.DataFrame()
        )
        render_symbol_deep_dive(sym, preds, features, panel, broker_panel, universe_df=universe_dive)

elif page == "Daily Upload":
    from backend.ingest.data_inventory import data_folder_inventory, panel_side_summary

    st.header("Daily Upload")
    st.markdown("Upload **Accumulation** and **Distribution** Excel workbooks (multi-sheet). Optional OHLCV CSV.")

    inv = data_folder_inventory()
    ps = panel_side_summary(panel)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acc files in folder", inv["accumulation_file_count"])
    c2.metric("Dist files in folder", inv["distribution_file_count"])
    c3.metric("Panel acc rows", ps.get("accumulation_rows", 0))
    c4.metric("Panel dist rows", ps.get("distribution_rows", 0))
    if inv.get("message"):
        st.info(inv["message"])
    if inv.get("files"):
        st.dataframe(pd.DataFrame(inv["files"]), hide_index=True, use_container_width=True)
    st.markdown(
        """
        **Why Floorsheet / Accumulation can show 0 in UI**
        - **Accumulation Data** must contain CSV/Excel with headers like `Net Buy Amt`, `Accumulation Power`, `Buyer Broker`.
        - Files named `AccumulationDistribution (13).csv` in **Distribution Data** are still **distribution** exports (`Net Sell Amt`, `Distribution Power`) — both sides are not in one file today.
        - After adding accumulation files, click **Run Pipeline** in the sidebar to rebuild features.
        """
    )
    rd = st.date_input("Report date", value=date.today())
    acc_file = st.file_uploader("Accumulation Excel", type=["xlsx", "xls"])
    dist_file = st.file_uploader("Distribution Excel", type=["xlsx", "xls"])
    ohlcv_file = st.file_uploader("OHLCV CSV", type=["csv"])

    if st.button("Ingest & Analyze"):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            acc_path = dist_path = ohlcv_path = None
            if acc_file:
                acc_path = td / acc_file.name
                acc_path.write_bytes(acc_file.read())
            if dist_file:
                dist_path = td / dist_file.name
                dist_path.write_bytes(dist_file.read())
            if ohlcv_file:
                ohlcv_path = td / ohlcv_file.name
                ohlcv_path.write_bytes(ohlcv_file.read())
            with st.spinner("Processing..."):
                result = run_pipeline(report_date=rd, acc_path=acc_path, dist_path=dist_path, ohlcv_path=ohlcv_path)
            st.json(result)

elif page == "Backtest":
    from backend.backtest.engine import prepare_backtest_signals
    from backend.quant.algorithms_catalog import ALGORITHM_SECTIONS

    hero("Backtest", "Simulate entries when signal tier fires · exit after N days using LTP history")
    px_df = merge_ohlcv_sources(ohlcv, features)
    sig_hist, sig_diag = prepare_backtest_signals(preds, features)
    st.caption(
        f"Price history: **{px_df['symbol'].nunique() if not px_df.empty else 0}** symbols · "
        f"**{px_df['date'].nunique() if not px_df.empty else 0}** dates · "
        f"Signal history: **{sig_diag.get('rows', 0)}** rows · **{sig_diag.get('dates', 0)}** dates "
        f"({sig_diag.get('source', '—')})"
    )
    if sig_diag.get("tier_counts"):
        st.markdown("**Tiers in signal history:** " + ", ".join(f"{k}={v}" for k, v in sig_diag["tier_counts"].items()))

    with st.expander("Why ‘no Confirmed / Setup / Watch’?", expanded=True):
        st.markdown(
            """
1. **Tiers are strict** — most names are **Neutral** or **Watch** after calibration (not a bug).  
2. **Backtest needs history** — upload **multiple daily** Distribution/Accumulation files so `features` has **2+ report_date** per symbol.  
3. **Entry filter** — if you pick only **Confirmed** but history has zero Confirmed rows, you get 0 trades. Use **Watch** or **Setup** or multi-select below.  
4. **Prices** — each trade needs LTP on entry day **and** a later day; single-day uploads cannot fill trades.
            """
        )

    with st.expander("Quant algorithms used in signals", expanded=False):
        for sec in ALGORITHM_SECTIONS:
            st.markdown(f"**{sec['title']}**")
            for item in sec["items"]:
                st.markdown(f"- {item}")

    entry_tiers = st.multiselect(
        "Entry tiers (any match)",
        ["Watch", "Setup", "Trigger", "Confirmed"],
        default=["Setup", "Trigger", "Confirmed"],
        help="Include lower tiers if Confirmed count is 0 in your data.",
    )
    hold = st.slider("Hold days", 3, 30, 10)
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Simulating trades…"):
            result = run_backtest(
                sig_hist if not sig_hist.empty else preds,
                ohlcv,
                entry_tier=entry_tiers[0] if entry_tiers else "Trigger",
                hold_days=hold,
                features=features,
                entry_tiers=entry_tiers or None,
            )
        st.session_state["backtest_result"] = result
    result = st.session_state.get("backtest_result")
    if result:
        if result.get("message") and result["message"] != "ok":
            st.warning(result["message"])
        if result.get("tier_counts"):
            st.caption(f"Filtered tiers: {result.get('tier_filter')} · matched entries: {result.get('entries_matched', '—')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", result["trades"])
        c2.metric("Win Rate", f"{result['win_rate']:.0%}")
        c3.metric("Avg Return", f"{result['avg_return']:.2f}%")
        c4.metric("CAGR Proxy", f"{result['cagr_proxy']:.1%}")
        if result.get("skipped"):
            st.caption(f"Skipped breakdown: {result['skipped']}")
        if result.get("details"):
            st.dataframe(pd.DataFrame(result["details"]), use_container_width=True, hide_index=True)
    else:
        st.info("Click **Run Backtest**. Upload multiple daily Excel files so features have several `report_date`s per symbol.")

elif page == "LLM Briefing":
    st.header("LLM Daily Brief")
    status = llm_status()
    st.caption(f"Provider: **{status['provider']}** | Model: **{status.get('model')}** | Ready: **{status['ready']}**")
    if status.get("hint"):
        st.info(status["hint"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Test LLM Connection"):
            with st.spinner("Testing..."):
                test = test_llm_connection()
            st.write("OK:", test.get("ok"))
            st.markdown(test.get("response", ""))
    with c2:
        if st.button("Generate Brief (Top 120 Vol)", disabled=st.session_state.get("brief_generating", False)):
            universe = get_latest_scanner_universe(preds, panel=panel, broker_panel=broker_panel, top_n=120)
            start_brief_generation(universe)
            st.rerun()
    poll_brief_job()
    if st.session_state.get("brief_generating"):
        st.info("Generating brief in a background thread — **you can switch tabs now.** Sidebar updates when ready.")
        try:
            st.autorefresh(interval=4000, key="brief_page_poll")
        except Exception:
            pass
    if st.session_state.get("brief_error"):
        st.error(st.session_state["brief_error"])
    if st.session_state.get("llm_brief"):
        st.markdown(st.session_state["llm_brief"])

elif page == "Chat":
    hero("Quant Chat", "Ask about any symbol — e.g. 'Should I long NGPL?' or 'Compare API vs AKJCL'")
    all_syms = all_tracked_symbols(preds, features) if not preds.empty else []
    c1, c2 = st.columns([1, 2])
    with c1:
        chat_sym = st.selectbox("Focus symbol (optional)", ["—"] + all_syms, key="chat_sym")
    with c2:
        q = st.text_input("Your question", placeholder="e.g. Is SIPD a good early momentum long?")
    if st.button("Send", type="primary") and q:
        universe = (
            get_latest_scanner_universe(preds, panel=panel, broker_panel=broker_panel, top_n=120)
            if not preds.empty
            else preds
        )
        extra = pd.DataFrame()
        if chat_sym and chat_sym != "—":
            extra = enrich_symbol_row(chat_sym, preds, panel, broker_panel, features=features)
        with st.spinner("Thinking…"):
            _cq = importlib.import_module("backend.llm.analyst").chat_query
            kwargs = {"extra_rows": extra} if not extra.empty else {}
            ans = _cq(q, universe, **kwargs)
        st.session_state["chat_last_answer"] = ans
    if st.session_state.get("chat_last_answer"):
        st.markdown(st.session_state["chat_last_answer"])
