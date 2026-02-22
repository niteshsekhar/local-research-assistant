from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    watch_dir: Path
    chroma_dir: Path
    reports_dir: Path
    embedding_model: str
    llm_api_base: str
    llm_api_key: str
    llm_model: str
    watch_interval: int



def _resolve_watch_dir(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path
    if path.is_absolute() and not path.exists():
        fallback = Path.cwd() / "papers"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    resolved = (Path.cwd() / path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved



def get_settings() -> Settings:
    watch_dir = _resolve_watch_dir(os.getenv("WATCH_DIR", "/papers"))
    chroma_dir = Path(os.getenv("CHROMA_DIR", "./data/chroma")).resolve()
    reports_dir = Path(os.getenv("REPORTS_DIR", "./reports")).resolve()

    chroma_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        watch_dir=watch_dir,
        chroma_dir=chroma_dir,
        reports_dir=reports_dir,
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        llm_api_base=os.getenv("LLM_API_BASE", "http://localhost:11434/v1"),
        llm_api_key=os.getenv("LLM_API_KEY", "local-key"),
        llm_model=os.getenv("LLM_MODEL", "llama3.1"),
        watch_interval=int(os.getenv("WATCH_INTERVAL", "10")),
    )
