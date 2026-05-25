"""Model Lab — explain multimodal temporal attention in plain language."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HORIZON_HELP = {
    "1D": "Today’s floorsheet flow (most weight in early momentum)",
    "2D": "2-day cumulative broker/float activity",
    "3D": "3-day trend",
    "4D": "4-day trend",
    "1W": "Weekly structure — confirms if 1D spike is sustained",
}


def render_model_lab_charts(attn_df: pd.DataFrame, focus_sym: str | None = None) -> None:
    if attn_df.empty:
        st.info(
            "No attention weights yet. Train the model, then **Run Pipeline** so predictions export "
            "per-symbol horizon importance."
        )
        return

    df = attn_df.copy()
    df["temporal_weight"] = pd.to_numeric(df["temporal_weight"], errors="coerce").fillna(0)
    df["horizon"] = df["horizon"].astype(str)
    order = ["1D", "2D", "3D", "4D", "1W"]
    df["horizon"] = pd.Categorical(df["horizon"], categories=order, ordered=True)

    st.markdown(
        """
**What “temporal attention” means**  
The multimodal model reads your floorsheet like a trader scanning horizons: **1D → 1W**.  
It learns which horizons best predict **early long momentum** for each stock.  
Higher bar = the model relied more on that horizon when scoring that symbol.
        """
    )

    mean_attn = df.groupby("horizon", observed=False)["temporal_weight"].mean().reset_index()
    mean_attn = mean_attn.sort_values("horizon")

    c1, c2 = st.columns([1, 1])
    with c1:
        fig_mean = go.Figure(
            data=[
                go.Bar(
                    x=mean_attn["horizon"].astype(str),
                    y=mean_attn["temporal_weight"],
                    marker_color="#58a6ff",
                    text=[f"{v:.2f}" for v in mean_attn["temporal_weight"]],
                    textposition="outside",
                )
            ]
        )
        fig_mean.update_layout(
            title="Market-wide average — which horizons matter most",
            xaxis_title="Floorsheet horizon",
            yaxis_title="Learned weight",
            height=360,
        )
        st.plotly_chart(fig_mean, use_container_width=True)
        st.caption("If **1D** dominates, the model trusts same-day turnover/flow; if **1W** rises, it wants weekly confirmation.")

    with c2:
        for h, tip in HORIZON_HELP.items():
            st.markdown(f"- **{h}** — {tip}")

    syms = sorted(df["symbol"].astype(str).unique())
    pick = focus_sym if focus_sym and focus_sym in syms else syms[0]
    if len(syms) > 1:
        pick = st.selectbox("Symbol attention profile", syms, index=syms.index(pick) if pick in syms else 0)

    one = df[df["symbol"] == pick].sort_values("horizon")
    if not one.empty:
        fig_one = go.Figure(
            data=[
                go.Scatter(
                    x=one["horizon"].astype(str),
                    y=one["temporal_weight"],
                    mode="lines+markers",
                    line=dict(color="#3fb950", width=3),
                    marker=dict(size=12),
                    fill="tozeroy",
                )
            ]
        )
        fig_one.update_layout(
            title=f"{pick} — horizon importance (this stock)",
            xaxis_title="Horizon",
            yaxis_title="Weight",
            height=320,
        )
        st.plotly_chart(fig_one, use_container_width=True)
        top_h = one.loc[one["temporal_weight"].idxmax(), "horizon"]
        st.success(f"For **{pick}**, the model leans most on **{top_h}** when estimating P(long).")

    if df["symbol"].nunique() > 1:
        heat = df.pivot_table(index="symbol", columns="horizon", values="temporal_weight", aggfunc="mean")
        heat = heat.reindex(columns=[c for c in order if c in heat.columns])
        fig_hm = px.imshow(
            heat.head(40),
            labels=dict(x="Horizon", y="Symbol", color="Weight"),
            title="Horizon weights heatmap (top 40 symbols)",
            color_continuous_scale="Blues",
            aspect="auto",
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("Compare stocks: a bright **1D** cell means that name’s move is a one-day story; bright **1W** = weekly trend story.")
