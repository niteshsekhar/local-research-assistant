from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .embeddings import Embedder
from .llm_client import LocalLLMClient
from .models import IndexedPaper
from .parser import parse_pdf
from .report import generate_paper_report
from .vector_store import PaperStore


class IngestionPipeline:
    def __init__(
        self,
        store: PaperStore,
        embedder: Embedder,
        llm_client: LocalLLMClient,
        reports_dir: Path | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm_client = llm_client
        self.reports_dir = reports_dir

    def ingest_pdf(self, pdf_path: Path, force: bool = False) -> str:
        paper_id = self.store.build_paper_id(str(pdf_path.resolve()))
        if self.store.exists(paper_id) and not force:
            return f"Skipped {pdf_path.name} (already indexed)."

        parsed = parse_pdf(pdf_path)
        insight = self.llm_client.analyze_paper(parsed)

        title = parsed.full_text.splitlines()[0][:180] if parsed.full_text else pdf_path.stem
        indexed = IndexedPaper(
            paper_id=paper_id,
            title=title or pdf_path.stem,
            added_at=datetime.utcnow(),
            parsed=parsed,
            insight=insight,
        )

        embedding_source = (
            f"{indexed.title}\n"
            f"{indexed.insight.summary}\n"
            f"{' '.join(indexed.insight.innovations)}\n"
            f"{' '.join(indexed.insight.contributions)}\n"
            f"{' '.join(indexed.insight.training_info)}\n"
            f"{indexed.insight.architecture}\n"
            f"{' '.join(indexed.insight.pros)}\n"
            f"{' '.join(indexed.insight.cons)}\n"
            f"{' '.join(indexed.insight.next_steps)}\n"
            f"{' '.join(indexed.insight.research_ideas)}"
        )
        embedding = self.embedder.embed([embedding_source])[0]
        self.store.upsert(indexed, embedding)
        if self.reports_dir is not None:
            generate_paper_report(indexed, self.reports_dir)
        action = "Re-indexed" if force else "Indexed"
        return f"{action} {pdf_path.name}"

    def query(self, text: str, limit: int = 10) -> list[dict]:
        query_embedding = self.embedder.embed([text])[0]
        return self.store.query(text, query_embedding, limit=limit)
