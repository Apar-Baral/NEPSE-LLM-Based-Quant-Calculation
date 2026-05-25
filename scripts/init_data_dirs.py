"""Create data directory structure per plan."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = [
    ROOT / "data" / "raw" / "accumulation",
    ROOT / "data" / "raw" / "distribution",
    ROOT / "data" / "raw" / "ohlcv",
    ROOT / "data" / "processed",
    ROOT / "data" / "models",
    ROOT / "data" / "inbox",
]

for d in DIRS:
    d.mkdir(parents=True, exist_ok=True)
    (d / ".gitkeep").write_text("", encoding="utf-8")

print("Created:", ", ".join(str(d.relative_to(ROOT)) for d in DIRS))
