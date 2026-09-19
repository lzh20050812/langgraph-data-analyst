"""Persistent vector memory for reusable, completed business-analysis reports."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List

from storage.chromadb.embedder import get_embedder


COLLECTION = "analysis_memory"


class AnalysisMemory:
    def __init__(self):
        self.embedder = get_embedder()

    @property
    def collection(self):
        return self.embedder.client.get_or_create_collection(
            COLLECTION, metadata={"description": "Reusable historical analysis reports"}
        )

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        try:
            self.embedder.client.delete_collection(COLLECTION)
        except Exception:
            pass

    def remember(
        self, *, owner_id: str, query: str, report: str,
        category: str = "", source_id: str = "",
    ) -> str:
        if not owner_id.strip():
            raise ValueError("owner_id is required for analysis memory")
        if not query.strip() or len(report.strip()) < 50:
            raise ValueError("query/report is too short for durable analysis memory")
        memory_id = source_id or hashlib.sha256(
            (owner_id + "\n" + query + "\n" + report).encode("utf-8")
        ).hexdigest()[:24]
        document = f"业务问题：{query}\n历史分析报告：{report[:6000]}"
        embedding = self.embedder.model.encode([document]).tolist()
        metadata = {"owner_id": owner_id[:128], "query": query[:1000], "category": category[:200],
                    "created_utc": datetime.now(timezone.utc).isoformat(), "source_id": source_id[:100]}
        self.collection.upsert(ids=[memory_id], embeddings=embedding, documents=[document], metadatas=[metadata])
        return memory_id

    def recall(
        self, query: str, *, owner_id: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        if not owner_id.strip():
            return []
        if self.count() == 0:
            return []
        n = min(max(1, int(top_k)), self.count())
        embedding = self.embedder.model.encode([query]).tolist()
        result = self.collection.query(query_embeddings=embedding, n_results=n,
            where={"owner_id": owner_id},
            include=["metadatas", "documents", "distances"])
        return [{"memory_id": mid, "query": meta.get("query", ""),
                 "category": meta.get("category", ""), "document": doc,
                 "score": round(1-float(distance), 6), "rank": rank}
                for rank, (mid, meta, doc, distance) in enumerate(zip(result["ids"][0],
                    result["metadatas"][0], result["documents"][0], result["distances"][0]), 1)]


_memory: AnalysisMemory | None = None


def get_analysis_memory() -> AnalysisMemory:
    global _memory
    if _memory is None:
        _memory = AnalysisMemory()
    return _memory
