from __future__ import annotations

from dataclasses import dataclass

from .embeddings import Embedder
from .highlights import HighlightedParagraph
from .llm_client import LocalLLMClient
from .vector_store import PaperStore


@dataclass
class CompanionResponse:
    highlight: HighlightedParagraph
    related_concepts: list[str]
    expert_explanation: str
    simplified_explanation: str
    retrieved_papers: list[str]


class ReadingCompanion:
    def __init__(self, store: PaperStore, embedder: Embedder, llm_client: LocalLLMClient) -> None:
        self.store = store
        self.embedder = embedder
        self.llm_client = llm_client

    def explain(
        self,
        highlight: HighlightedParagraph,
        expertise_level: str = "ML researcher",
        include_simplified: bool = False,
        limit: int = 5,
    ) -> CompanionResponse:
        query_embedding = self.embedder.embed([highlight.text])[0]
        results = self.store.query(highlight.text, query_embedding, limit=limit)

        related_concepts: list[str] = []
        retrieved_papers: list[str] = []
        for row in results:
            meta = row.get("metadata", {})
            title = str(meta.get("title", "Untitled"))
            method = str(meta.get("method_type", "other"))
            summary = str(meta.get("summary", "")).strip()
            innovations = str(meta.get("innovations", "")).replace(" || ", "; ").strip()
            snippet = f"{title} ({method})"
            if summary:
                snippet += f": {summary[:220]}"
            if innovations:
                snippet += f" | innovations: {innovations[:220]}"
            related_concepts.append(snippet)
            retrieved_papers.append(title)

        llm_output = self.llm_client.explain_highlight(
            highlight_text=highlight.text,
            related_concepts=related_concepts,
            expertise_level=expertise_level,
            include_simplified=include_simplified,
        )

        return CompanionResponse(
            highlight=highlight,
            related_concepts=llm_output.get("related_links") or related_concepts,
            expert_explanation=llm_output.get("expert_explanation", ""),
            simplified_explanation=llm_output.get("simplified_explanation", ""),
            retrieved_papers=retrieved_papers,
        )
