from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class HighlightedParagraph:
    page: int
    text: str
    context: str


def _rect_overlap(a: fitz.Rect, b: fitz.Rect) -> float:
    intersection = a & b
    if intersection.is_empty:
        return 0.0
    area_a = max(float(a.width) * float(a.height), 0.0)
    area_b = max(float(b.width) * float(b.height), 0.0)
    area_intersection = max(float(intersection.width) * float(intersection.height), 0.0)
    min_area = max(min(area_a, area_b), 1.0)
    return area_intersection / min_area


def _paragraph_from_blocks(page: fitz.Page, rect: fitz.Rect) -> str:
    blocks = page.get_text("blocks")
    snippets: list[str] = []
    for block in blocks:
        block_rect = fitz.Rect(block[:4])
        text = (block[4] or "").strip()
        if not text:
            continue
        if _rect_overlap(block_rect, rect) > 0.2:
            snippets.append(text)
    if snippets:
        return "\n".join(snippets)
    return page.get_textbox(rect).strip()


def extract_highlighted_paragraphs(pdf_path: Path) -> list[HighlightedParagraph]:
    highlights: list[HighlightedParagraph] = []
    seen: set[str] = set()

    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            annotation = page.first_annot
            while annotation:
                if annotation.type[1] == "Highlight":
                    paragraph = _paragraph_from_blocks(page, annotation.rect)
                    paragraph = " ".join(paragraph.split())
                    if paragraph and paragraph not in seen:
                        seen.add(paragraph)
                        page_context = " ".join(page.get_text("text").split())
                        highlights.append(
                            HighlightedParagraph(
                                page=page_index + 1,
                                text=paragraph,
                                context=page_context[:1800],
                            )
                        )
                annotation = annotation.next
    return highlights
