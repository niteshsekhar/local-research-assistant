from __future__ import annotations

import argparse
from pathlib import Path

from research_assistant.config import get_settings
from research_assistant.embeddings import Embedder
from research_assistant.llm_client import LocalLLMClient
from research_assistant.pipeline import IngestionPipeline
from research_assistant.vector_store import PaperStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-index papers to refresh richer metadata.")
    parser.add_argument("--file", type=str, default="", help="Single PDF path to re-index.")
    args = parser.parse_args()

    settings = get_settings()
    store = PaperStore(str(settings.chroma_dir))
    embedder = Embedder(settings.embedding_model)
    llm_client = LocalLLMClient(settings)
    pipeline = IngestionPipeline(
        store=store,
        embedder=embedder,
        llm_client=llm_client,
        reports_dir=settings.reports_dir,
    )

    if args.file:
        target = Path(args.file).expanduser().resolve()
        if not target.exists() or target.suffix.lower() != ".pdf":
            raise SystemExit(f"Invalid PDF path: {target}")
        print(pipeline.ingest_pdf(target, force=True))
        return

    for pdf_path in sorted(settings.watch_dir.glob("*.pdf")):
        print(pipeline.ingest_pdf(pdf_path, force=True))


if __name__ == "__main__":
    main()
