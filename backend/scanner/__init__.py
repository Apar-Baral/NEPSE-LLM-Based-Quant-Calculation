"""Scanner package — re-exports from volume_universe."""

from __future__ import annotations

from typing import Any

_EXPORTS = {
    "attach_volume_from_panel",
    "compute_early_rank_score",
    "get_latest_scanner_universe",
    "select_high_volume_universe",
    "symbol_horizon_snapshot",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        from . import volume_universe

        return getattr(volume_universe, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORTS)
