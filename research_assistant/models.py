from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class ParsedPaper:
    file_path: str
    file_name: str
    full_text: str
    equation_candidates: List[str]


@dataclass
class PaperInsight:
    summary: str
    innovations: List[str]
    contributions: List[str]
    method_type: str
    training_info: List[str]
    architecture: str
    pros: List[str]
    cons: List[str]
    next_steps: List[str]
    research_ideas: List[str]


@dataclass
class IndexedPaper:
    paper_id: str
    title: str
    added_at: datetime
    parsed: ParsedPaper
    insight: PaperInsight
