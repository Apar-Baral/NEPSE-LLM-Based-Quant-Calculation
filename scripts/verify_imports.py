"""Verify scanner imports work (run from project root)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("Python:", sys.executable)
print("ROOT:", ROOT)

try:
    from backend.scanner.volume_universe import get_latest_scanner_universe, symbol_horizon_snapshot
    print("volume_universe import: OK")
    print("  get_latest_scanner_universe:", get_latest_scanner_universe)
    print("  symbol_horizon_snapshot:", symbol_horizon_snapshot)
except Exception as e:
    print("volume_universe import: FAILED", e)

try:
    from backend.scanner import get_latest_scanner_universe as g2
    print("backend.scanner import: OK")
except Exception as e:
    print("backend.scanner import: FAILED", e)

try:
    from backend.signals import assign_universe_tiers
    print("assign_universe_tiers import: OK")
except Exception as e:
    print("assign_universe_tiers import: FAILED", e)

try:
    import importlib
    m = importlib.import_module("backend.scanner.volume_universe")
    print("importlib: OK", hasattr(m, "get_latest_scanner_universe"))
except Exception as e:
    print("importlib: FAILED", e)
