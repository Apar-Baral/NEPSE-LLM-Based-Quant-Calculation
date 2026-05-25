"""Signal thresholds with safe defaults (fixes stale YAML cache KeyErrors)."""

from __future__ import annotations

from typing import Any

from backend.config import load_yaml_config

DEFAULT_SIGNALS: dict[str, Any] = {
    "watch_zscore": 1.5,
    "trigger_probability": 0.60,
    "confirmed_probability": 0.65,
    "early_momentum_score": 70,
    "distribution_mode": True,
    "dist_trigger_probability": 0.30,
    "dist_confirmed_probability": 0.40,
    "dist_early_momentum_score": 20,
    "dist_broker_pressure_trigger": 18,
    "dist_invalidate_drs": 85,
}


def get_signal_config() -> dict[str, Any]:
    raw = load_yaml_config("settings.yaml")
    signals = raw.get("signals") if isinstance(raw, dict) else None
    if not isinstance(signals, dict):
        signals = {}
    return {**DEFAULT_SIGNALS, **signals}


def clear_config_cache() -> None:
    """Clear cached YAML reads (Streamlit-safe; call at app startup)."""
    try:
        load_yaml_config.cache_clear()
    except Exception:
        pass
