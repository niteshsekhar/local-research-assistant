from __future__ import annotations

import re
from pathlib import Path
from typing import List

import fitz

from .models import ParsedPaper

EQUATION_PATTERN = re.compile(
    r"(?:\\[a-zA-Z]+|\$[^\$]{2,}\$|[A-Za-z]\s*=\s*[^\n]{1,80}|[∑∫√≈≠≤≥→λθμσπ])"
)



def _extract_equation_candidates(lines: List[str]) -> List[str]:
    candidates: list[str] = []
    seen = set()
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if EQUATION_PATTERN.search(text):
            normalized = re.sub(r"\s+", " ", text)
            if normalized not in seen:
                seen.add(normalized)
                candidates.append(normalized)
    return candidates[:80]



def parse_pdf(pdf_path: Path) -> ParsedPaper:
    with fitz.open(pdf_path) as doc:
        page_texts = [page.get_text("text") for page in doc]

    full_text = "\n".join(page_texts)
    lines = full_text.splitlines()
    equation_candidates = _extract_equation_candidates(lines)

    return ParsedPaper(
        file_path=str(pdf_path.resolve()),
        file_name=pdf_path.name,
        full_text=full_text,
        equation_candidates=equation_candidates,
    )
