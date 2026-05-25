from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.config import PROCESSED_DIR


RAG_INDEX_PATH = PROCESSED_DIR / "rag_index.json"


class SimpleRAG:
    """Lightweight JSON RAG without external vector DB dependency."""

    def __init__(self):
        self.entries: list[dict] = []
        if RAG_INDEX_PATH.exists():
            self.entries = json.loads(RAG_INDEX_PATH.read_text(encoding="utf-8"))

    def index_outcomes(self, features: pd.DataFrame, labels: pd.DataFrame) -> int:
        if features.empty:
            return 0
        merged = features.merge(
            labels,
            left_on=["symbol", "report_date"],
            right_on=["symbol", "as_of_date"],
            how="inner",
        )
        for _, row in merged.iterrows():
            self.entries.append(
                {
                    "symbol": row["symbol"],
                    "date": str(row["report_date"]),
                    "early_momentum_score": float(row.get("early_momentum_score", 0) or 0),
                    "outcome_10d": float(row.get("forward_return_10d", 0) or 0),
                    "label": int(row.get("long_momentum_10d", 0) or 0),
                }
            )
        RAG_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAG_INDEX_PATH.write_text(json.dumps(self.entries[-5000:], indent=2), encoding="utf-8")
        return len(merged)

    def find_analogs(self, symbol: str, score: float, k: int = 5) -> list[dict]:
        if not self.entries:
            return []
        ranked = sorted(self.entries, key=lambda e: abs(e["early_momentum_score"] - score))
        return ranked[:k]
