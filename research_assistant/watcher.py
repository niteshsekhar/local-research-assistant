from __future__ import annotations

import time
from pathlib import Path

from .pipeline import IngestionPipeline


class FolderWatcher:
    def __init__(self, watch_dir: Path, pipeline: IngestionPipeline, interval_seconds: int = 10) -> None:
        self.watch_dir = watch_dir
        self.pipeline = pipeline
        self.interval_seconds = interval_seconds
        self._seen: set[str] = set()

    def _list_pdfs(self) -> list[Path]:
        return sorted(self.watch_dir.glob("*.pdf"))

    def run_forever(self) -> None:
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        print(f"Watching {self.watch_dir} for new PDFs...")
        while True:
            for pdf_path in self._list_pdfs():
                if str(pdf_path.resolve()) in self._seen:
                    continue
                try:
                    message = self.pipeline.ingest_pdf(pdf_path)
                    print(message)
                    self._seen.add(str(pdf_path.resolve()))
                except Exception as exc:
                    print(f"Failed to process {pdf_path.name}: {exc}")
            time.sleep(self.interval_seconds)
