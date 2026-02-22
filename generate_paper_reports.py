from __future__ import annotations

from research_assistant.config import get_settings
from research_assistant.report import generate_paper_report_from_metadata
from research_assistant.vector_store import PaperStore


def main() -> None:
    settings = get_settings()
    store = PaperStore(str(settings.chroma_dir))
    rows = store.all_papers()
    if not rows:
        print("No indexed papers found.")
        return

    for row in rows:
        path = generate_paper_report_from_metadata(row["metadata"], settings.reports_dir)
        print(path)


if __name__ == "__main__":
    main()
