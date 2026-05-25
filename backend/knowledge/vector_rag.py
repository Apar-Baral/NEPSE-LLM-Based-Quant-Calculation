"""Vector store for logic chains (ChromaDB when available, else JSON fallback)."""

from __future__ import annotations

import json
from pathlib import Path

from backend.config import PROCESSED_DIR

VECTOR_PATH = PROCESSED_DIR / "logic_vectors.json"


class VectorLogicRAG:
    def __init__(self, collection: str = "nepse_logic"):
        self.collection_name = collection
        self._chroma = None
        self._collection = None
        self._fallback: list[dict] = []
        self._init_store()
        if VECTOR_PATH.exists():
            try:
                self._fallback = json.loads(VECTOR_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._fallback = []

    def _init_store(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            client = chromadb.PersistentClient(
                path=str(PROCESSED_DIR / "chroma"),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._chroma = client
            self._collection = client.get_or_create_collection(self.collection_name)
        except Exception:
            self._chroma = None
            self._collection = None

    def index_logic_chain(self, doc_id: str, text: str, metadata: dict) -> None:
        entry = {"id": doc_id, "text": text, "metadata": metadata}
        self._fallback.append(entry)
        if self._collection is not None:
            try:
                self._collection.upsert(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[{k: str(v) for k, v in metadata.items()}],
                )
            except Exception:
                pass

    def query(self, question: str, n: int = 5) -> list[dict]:
        if self._collection is not None:
            try:
                res = self._collection.query(query_texts=[question], n_results=n)
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]
            except Exception:
                pass
        # Fallback: keyword overlap
        q = question.lower()
        scored = []
        for e in self._fallback:
            t = e["text"].lower()
            score = sum(1 for w in q.split() if w in t)
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:n]]

    def save_fallback(self) -> None:
        VECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        VECTOR_PATH.write_text(json.dumps(self._fallback[-5000:], indent=2), encoding="utf-8")
