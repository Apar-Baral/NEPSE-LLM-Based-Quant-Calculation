"""Cross-page background tasks for Streamlit (non-blocking)."""

from __future__ import annotations

import threading
from typing import Any, Callable

import pandas as pd
import streamlit as st

# Module-level job store (threads cannot safely write st.session_state)
_BRIEF_JOBS: dict[str, dict[str, Any]] = {}


def _job_key() -> str:
    if "_brief_job_id" not in st.session_state:
        import uuid

        st.session_state["_brief_job_id"] = str(uuid.uuid4())
    return st.session_state["_brief_job_id"]


def start_brief_generation(universe_snapshot: pd.DataFrame) -> None:
    key = _job_key()
    st.session_state["brief_generating"] = True
    st.session_state["brief_ready"] = False
    st.session_state["brief_error"] = None
    st.session_state["brief_universe_snapshot"] = universe_snapshot.copy()

    uni = universe_snapshot.copy()

    def _worker() -> None:
        try:
            from backend.llm.analyst import generate_daily_brief

            text = generate_daily_brief(uni)
            _BRIEF_JOBS[key] = {"running": False, "brief": text, "error": None}
        except Exception as exc:
            _BRIEF_JOBS[key] = {"running": False, "brief": None, "error": str(exc)}

    _BRIEF_JOBS[key] = {"running": True, "brief": None, "error": None}
    threading.Thread(target=_worker, daemon=True).start()


def poll_brief_job() -> None:
    """Sync thread results into session_state — call once per run, does not block UI."""
    if not st.session_state.get("brief_generating"):
        return

    key = st.session_state.get("_brief_job_id")
    if not key or key not in _BRIEF_JOBS:
        return

    job = _BRIEF_JOBS[key]
    if job.get("running"):
        return

    st.session_state["brief_generating"] = False
    st.session_state["brief_ready"] = True
    if job.get("brief"):
        st.session_state["llm_brief"] = job["brief"]
    if job.get("error"):
        st.session_state["brief_error"] = job["error"]


def brief_status_sidebar() -> None:
    poll_brief_job()
    if st.session_state.get("brief_generating"):
        st.sidebar.warning("LLM brief generating… you can switch tabs now.")
        if hasattr(st, "autorefresh"):
            st.autorefresh(interval=4000, key="brief_poll")
    elif st.session_state.get("brief_ready") and st.session_state.get("llm_brief"):
        st.sidebar.success("LLM brief ready — open **LLM Briefing**")
