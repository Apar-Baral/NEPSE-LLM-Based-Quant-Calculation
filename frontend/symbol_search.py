"""Reusable symbol picker with company-name search."""

from __future__ import annotations

import streamlit as st

from backend.data.symbol_registry import (
    format_symbol_label,
    get_symbol_display_map,
    refresh_symbol_names_from_nepse,
    search_symbols,
)


def symbol_picker(
    label: str,
    all_symbols: list[str],
    *,
    key_prefix: str,
    default: str = "",
    help_text: str = "",
) -> str:
    """
    Type-ahead style picker: search box + dropdown of matches (ticker + full name).
    Returns selected symbol (uppercase).
    """
    names_key = f"{key_prefix}_names_loaded"
    if names_key not in st.session_state:
        names = get_symbol_display_map(all_symbols)
        if len(names) < max(50, len(all_symbols) // 2):
            names = refresh_symbol_names_from_nepse()
        st.session_state[names_key] = names
    names: dict[str, str] = st.session_state[names_key]

    c1, c2 = st.columns([3, 1])
    with c2:
        if st.button("↻ Names", key=f"{key_prefix}_refresh_names", help="Refresh company names from NEPSE"):
            st.session_state[names_key] = refresh_symbol_names_from_nepse()
            st.rerun()

    query = c1.text_input(
        label,
        value=default,
        key=f"{key_prefix}_query",
        placeholder="Type symbol or company name (e.g. NGPL, Nepal…)",
        help=help_text or "Matches ticker and full company name when cached.",
    ).strip()

    matches = search_symbols(query, all_symbols, names, limit=30) if query else all_symbols[:30]
    if not matches:
        st.caption("No matches — check spelling or run pipeline for symbol list.")
        return (default or (all_symbols[0] if all_symbols else "")).strip().upper()

    pick_key = f"{key_prefix}_pick"
    if pick_key not in st.session_state or st.session_state[pick_key] not in all_symbols:
        st.session_state[pick_key] = matches[0]

    if query:
        q_upper = query.upper()
        if q_upper in all_symbols:
            st.session_state[pick_key] = q_upper
        elif matches:
            st.session_state[pick_key] = matches[0]

    sym = st.selectbox(
        f"Select ({len(all_symbols)} listed)",
        matches if query else all_symbols,
        index=0,
        format_func=lambda s: format_symbol_label(s, names),
        key=pick_key,
    )
    return str(sym).strip().upper()
