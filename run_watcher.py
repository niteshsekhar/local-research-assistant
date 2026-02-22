from research_assistant.config import get_settings
from research_assistant.embeddings import Embedder
from research_assistant.llm_client import LocalLLMClient
from research_assistant.pipeline import IngestionPipeline
from research_assistant.vector_store import PaperStore
from research_assistant.watcher import FolderWatcher



def main() -> None:
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

    watcher = FolderWatcher(
        watch_dir=settings.watch_dir,
        pipeline=pipeline,
        interval_seconds=settings.watch_interval,
    )
    watcher.run_forever()


if __name__ == "__main__":
    main()
