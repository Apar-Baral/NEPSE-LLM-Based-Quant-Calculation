"""Shared Streamlit styling."""

from __future__ import annotations

import streamlit as st

from frontend.display_config import TIER_COLORS

# Plotly: mouse-wheel zoom + toolbar (use on st.plotly_chart(..., config=PLOTLY_ZOOM_CONFIG))
PLOTLY_ZOOM_CONFIG: dict = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def inject_global_css() -> None:
    tier_css = "".join(
        f'.tier-{t.lower()} {{ color: {c}; font-weight: 600; }}' for t, c in TIER_COLORS.items()
    )
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'DM Sans', system-ui, sans-serif; }}
        .main .block-container {{ padding-top: 1.5rem; max-width: 1400px; }}
        h1, h2, h3 {{ letter-spacing: -0.02em; }}
        div[data-testid="stMetric"] {{
            background: linear-gradient(145deg, #12151c 0%, #1e2430 55%, #252b38 100%);
            padding: 14px 18px; border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 4px 24px rgba(0,0,0,0.25);
        }}
        div[data-testid="stMetric"] label {{ color: #8b95a8; font-size: 0.8rem; }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: #f0f4ff; font-weight: 700;
        }}
        .nepse-hero {{
            background: linear-gradient(120deg, #0d1117 0%, #161b22 40%, #1a2332 100%);
            border: 1px solid #30363d; border-radius: 14px;
            padding: 1.25rem 1.5rem; margin-bottom: 1rem;
        }}
        .nepse-hero h2 {{ margin: 0; color: #58a6ff; font-size: 1.35rem; }}
        .nepse-hero p {{ margin: 0.35rem 0 0; color: #8b949e; font-size: 0.9rem; }}
        div[data-testid="stDataFrame"] {{
            border-radius: 10px; border: 1px solid #30363d;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0; font-weight: 500;
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(90deg, #238636, #2ea043);
            border: none; font-weight: 600;
        }}
        {tier_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="nepse-hero"><h2>{title}</h2><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def tier_badge(tier: str) -> str:
    return f'<span class="tier-{tier.lower()}">{tier}</span>'
