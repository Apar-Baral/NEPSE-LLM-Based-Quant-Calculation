"""Agent fleet — import submodules directly to avoid circular imports with knowledge graph."""

from backend.agents.catalog import build_agent_catalog, catalog_summary
from backend.agents.fleet import deploy_agent_fleet, fleet_status

__all__ = [
    "deploy_agent_fleet",
    "fleet_status",
    "build_agent_catalog",
    "catalog_summary",
    "run_analysis_swarm",
]


def __getattr__(name: str):
    if name == "run_analysis_swarm":
        from backend.agents.orchestrator import run_analysis_swarm

        return run_analysis_swarm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
