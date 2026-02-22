from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re

from .models import IndexedPaper
from .vector_store import PaperStore


def _split_field(value: str, fallback: str) -> list[str]:
    items = [item.strip() for item in str(value or "").split(" || ") if item.strip()]
    return items if items else [fallback]


def _safe_slug(text: str, default: str = "paper") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return cleaned[:80] if cleaned else default


def _render_paper_sections(meta: dict, include_header: bool = True) -> list[str]:
    innovations = _split_field(meta.get("innovations", ""), "Not captured (re-index for richer analysis).")
    training_info = _split_field(meta.get("training_info", ""), "Not captured (re-index for richer analysis).")
    pros = _split_field(meta.get("pros", ""), "Not captured.")
    cons = _split_field(meta.get("cons", ""), "Not captured.")
    next_steps = _split_field(meta.get("next_steps", ""), "Review paper manually and design follow-up experiments.")
    contributions = _split_field(meta.get("contributions", ""), "Not captured.")
    research_ideas = _split_field(meta.get("research_ideas", ""), "No generated ideas found.")
    equations = _split_field(meta.get("equations", ""), "No equation candidates captured.")

    lines: list[str] = []
    if include_header:
        lines.extend(
            [
                f"### {meta.get('title', 'Untitled')}",
                f"- Method: {meta.get('method_type', 'other')}",
                "",
            ]
        )

    lines.extend(["#### 1) Summary & Innovations", f"- Summary: {meta.get('summary', 'Not captured.')}", "- Important innovations:"])
    for item in innovations:
        lines.append(f"- {item}")

    lines.extend(["", "#### 2) Training Details", "- Training setup / hyperparameters / losses:"])
    for item in training_info:
        lines.append(f"- {item}")

    lines.extend(["", "#### 3) Architecture", f"- {meta.get('architecture', 'Not captured (re-index for richer analysis).')}"])

    lines.extend(["", "#### 4) Contributions", "- Key contributions:"])
    for item in contributions:
        lines.append(f"- {item}")

    lines.extend(["", "#### 5) Pros & Cons", "- Pros:"])
    for item in pros:
        lines.append(f"- {item}")
    lines.append("- Cons:")
    for item in cons:
        lines.append(f"- {item}")

    lines.extend(["", "#### 6) Next Steps", "- Suggested next steps:"])
    for item in next_steps:
        lines.append(f"- {item}")

    lines.extend(["", "#### 7) Research Ideas", "- Top idea seeds:"])
    for idea in research_ideas[:5]:
        lines.append(f"- {idea}")

    lines.extend(["", "#### 8) Equation Candidates", "- Parsed equation-like lines:"])
    for eq in equations[:20]:
        lines.append(f"- {eq}")
    lines.append("")
    return lines


def generate_paper_report(indexed: IndexedPaper, reports_dir: Path) -> Path:
    paper_reports_dir = reports_dir / "papers"
    paper_reports_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "title": indexed.title,
        "file_path": indexed.parsed.file_path,
        "method_type": indexed.insight.method_type,
        "added_at": indexed.added_at.isoformat(),
        "summary": indexed.insight.summary,
        "innovations": " || ".join(indexed.insight.innovations),
        "contributions": " || ".join(indexed.insight.contributions),
        "training_info": " || ".join(indexed.insight.training_info),
        "architecture": indexed.insight.architecture,
        "pros": " || ".join(indexed.insight.pros),
        "cons": " || ".join(indexed.insight.cons),
        "next_steps": " || ".join(indexed.insight.next_steps),
        "research_ideas": " || ".join(indexed.insight.research_ideas),
        "equations": " || ".join(indexed.parsed.equation_candidates[:20]),
    }

    lines = [
        f"# Paper Report: {indexed.title}",
        "",
        f"- File: {indexed.parsed.file_path}",
        f"- Indexed at (UTC): {indexed.added_at.isoformat()}",
        f"- Method type: {indexed.insight.method_type}",
        "",
    ]
    lines.extend(_render_paper_sections(meta, include_header=False))

    filename = f"{indexed.added_at.date().isoformat()}_{_safe_slug(indexed.title, indexed.paper_id)}.md"
    report_path = paper_reports_dir / filename
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def generate_paper_report_from_metadata(meta: dict, reports_dir: Path) -> Path:
    paper_reports_dir = reports_dir / "papers"
    paper_reports_dir.mkdir(parents=True, exist_ok=True)

    title = str(meta.get("title", "Untitled"))
    added_raw = str(meta.get("added_at", ""))
    try:
        added_at = datetime.fromisoformat(added_raw)
    except ValueError:
        added_at = datetime.utcnow()

    lines = [
        f"# Paper Report: {title}",
        "",
        f"- File: {meta.get('file_path', '')}",
        f"- Indexed at (UTC): {added_at.isoformat()}",
        f"- Method type: {meta.get('method_type', 'other')}",
        "",
    ]
    lines.extend(_render_paper_sections(meta, include_header=False))

    file_stem = Path(str(meta.get("file_path", "") or "paper")).stem
    filename = f"{added_at.date().isoformat()}_{_safe_slug(f'{title}-{file_stem}')}.md"
    report_path = paper_reports_dir / filename
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def generate_weekly_report(store: PaperStore, reports_dir: Path) -> Path:
    now = datetime.utcnow()
    since = now - timedelta(days=7)
    recent = store.papers_since(since)

    lines = [
        f"# Weekly Research Insights ({now.date().isoformat()})",
        "",
        f"Window: {since.date().isoformat()} to {now.date().isoformat()}",
        f"New papers indexed: {len(recent)}",
        "",
        "## Highlights",
    ]

    if not recent:
        lines.append("- No new papers indexed this week.")
    else:
        for row in recent:
            lines.extend(_render_paper_sections(row["metadata"], include_header=True))

    report_path = reports_dir / f"weekly_{now.date().isoformat()}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
