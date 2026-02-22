from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import requests


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    summary: str
    authors: List[str]
    published: str
    pdf_url: str
    primary_category: str


class ArxivClient:
    def search(
        self,
        query: str = "",
        category: str = "cs.LG",
        max_results: int = 20,
        start: int = 0,
    ) -> list[ArxivPaper]:
        query_parts: list[str] = []
        if query.strip():
            query_parts.append(f"all:{query.strip()}")
        if category.strip():
            query_parts.append(f"cat:{category.strip()}")
        search_query = " AND ".join(query_parts) if query_parts else "all:machine learning"

        params = {
            "search_query": search_query,
            "start": str(start),
            "max_results": str(max_results),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API_URL}?{urlencode(params)}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return self._parse_feed(response.text)

    def download_pdf(self, pdf_url: str, destination_dir: Path) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        paper_id = pdf_url.rstrip("/").split("/")[-1]
        file_name = f"{paper_id}.pdf" if not paper_id.endswith(".pdf") else paper_id
        file_path = destination_dir / file_name

        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        file_path.write_bytes(response.content)
        return file_path

    @staticmethod
    def _parse_feed(xml_text: str) -> list[ArxivPaper]:
        root = ET.fromstring(xml_text)
        entries = root.findall("atom:entry", ATOM_NS)
        papers: list[ArxivPaper] = []

        for entry in entries:
            title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=ATOM_NS) or "").strip()
            entry_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
            arxiv_id = entry_id.rstrip("/").split("/")[-1]

            authors = [
                (author.findtext("atom:name", default="", namespaces=ATOM_NS) or "").strip()
                for author in entry.findall("atom:author", ATOM_NS)
            ]
            authors = [author for author in authors if author]

            primary_category_node = entry.find("atom:category", ATOM_NS)
            primary_category = primary_category_node.attrib.get("term", "") if primary_category_node is not None else ""

            pdf_url = ""
            for link in entry.findall("atom:link", ATOM_NS):
                href = link.attrib.get("href", "")
                title_attr = link.attrib.get("title", "")
                link_type = link.attrib.get("type", "")
                if title_attr == "pdf" or link_type == "application/pdf":
                    pdf_url = href
                    break

            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            papers.append(
                ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    summary=summary,
                    authors=authors,
                    published=published,
                    pdf_url=pdf_url,
                    primary_category=primary_category,
                )
            )
        return papers
