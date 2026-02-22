from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from .models import IndexedPaper


class PaperStore:
    def __init__(self, chroma_path: str) -> None:
        client = chromadb.PersistentClient(path=chroma_path)
        self.collection: Collection = client.get_or_create_collection(
            name="papers", metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def build_paper_id(file_path: str) -> str:
        return hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:24]

    def exists(self, paper_id: str) -> bool:
        found = self.collection.get(ids=[paper_id])
        return bool(found.get("ids"))

    def upsert(self, item: IndexedPaper, embedding: list[float]) -> None:
        metadata = {
            "paper_id": item.paper_id,
            "title": item.title,
            "file_path": item.parsed.file_path,
            "method_type": item.insight.method_type,
            "added_at": item.added_at.isoformat(),
            "summary": item.insight.summary,
            "innovations": " || ".join(item.insight.innovations),
            "contributions": " || ".join(item.insight.contributions),
            "training_info": " || ".join(item.insight.training_info),
            "architecture": item.insight.architecture,
            "pros": " || ".join(item.insight.pros),
            "cons": " || ".join(item.insight.cons),
            "next_steps": " || ".join(item.insight.next_steps),
            "research_ideas": " || ".join(item.insight.research_ideas),
            "equations": " || ".join(item.parsed.equation_candidates[:20]),
        }
        document = (
            f"Title: {item.title}\n"
            f"Method type: {item.insight.method_type}\n"
            f"Summary: {item.insight.summary}\n"
            f"Innovations: {'; '.join(item.insight.innovations)}\n"
            f"Contributions: {'; '.join(item.insight.contributions)}\n"
            f"Training: {'; '.join(item.insight.training_info)}\n"
            f"Architecture: {item.insight.architecture}\n"
            f"Pros: {'; '.join(item.insight.pros)}\n"
            f"Cons: {'; '.join(item.insight.cons)}\n"
            f"Next steps: {'; '.join(item.insight.next_steps)}\n"
            f"Research ideas: {'; '.join(item.insight.research_ideas)}"
        )
        self.collection.upsert(
            ids=[item.paper_id],
            documents=[document],
            metadatas=[metadata],
            embeddings=[embedding],
        )

    def query(self, query_text: str, query_embedding: list[float], limit: int = 10) -> list[dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query_text],
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        merged: list[dict[str, Any]] = []
        for doc, meta, distance in zip(docs, metadatas, distances):
            merged.append(
                {
                    "document": doc,
                    "metadata": meta,
                    "score": round(1 - float(distance), 4),
                }
            )
        return merged

    def papers_since(self, since: datetime) -> list[dict[str, Any]]:
        all_items = self.collection.get(include=["metadatas", "documents"])
        rows: list[dict[str, Any]] = []
        for meta, doc in zip(all_items.get("metadatas", []), all_items.get("documents", [])):
            added_at = datetime.fromisoformat(meta["added_at"])
            if added_at >= since:
                rows.append({"metadata": meta, "document": doc})
        return rows

    def all_papers(self) -> list[dict[str, Any]]:
        all_items = self.collection.get(include=["metadatas", "documents"])
        rows: list[dict[str, Any]] = []
        for meta, doc in zip(all_items.get("metadatas", []), all_items.get("documents", [])):
            rows.append({"metadata": meta, "document": doc})
        return rows
