# Local Research Assistant (Python)

A local-first paper assistant that watches a `/papers` folder, ingests new PDFs, extracts text/equation-like lines, summarizes contributions, classifies method type, generates 5 research ideas, stores everything in ChromaDB, supports semantic querying, creates weekly + per-paper markdown reports, and includes a reading companion for highlighted PDF passages.

## Stack
- `PyMuPDF` for PDF parsing
- `sentence-transformers` for embeddings
- `ChromaDB` for local vector storage
- local LLM API interface (OpenAI-compatible `/chat/completions`)
- `Streamlit` for UI

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables (optional):

```bash
cp .env.example .env
```

Set values as needed (notably `LLM_API_BASE`, `LLM_MODEL`).

4. Run watcher (auto-ingest from watch folder):

```bash
python run_watcher.py
```

5. Run UI:

```bash
streamlit run streamlit_app.py
```

6. Test local LLM server connectivity:

```bash
python check_llm_server.py
```

7. Re-index papers to backfill richer report fields:

```bash
python reindex_papers.py
```

8. Generate per-paper reports for already indexed papers:

```bash
python generate_paper_reports.py
```

## Behavior
- Default watch path is `/papers`; if unavailable locally, it falls back to `./papers`.
- New PDFs are parsed with PyMuPDF.
- Equation extraction is heuristic (math symbols, LaTeX-ish fragments, assignment-style lines).
- Paper analysis uses multi-hop LLM querying:
  - hop 1: global paper understanding
  - hop 2A: summary/innovations/contributions
  - hop 2B: architecture/training details
  - hop 2C: pros/cons/next steps/research ideas
- LLM output is expected as strict JSON with:
  - `summary`
  - `contributions`
  - `method_type` (`scaling law`, `optimization`, `RL`, `architecture`, `systems`, `data`, `theory`, `other`)
  - `research_ideas` (5 items)
- Each LLM call enforces an input budget under ~4096 tokens (approximation-based guard).
- Query example: `show me all papers related to token routing`
- Discover tab supports ArXiv API search + one-click `Download + Index`.
- Weekly reports write to `./reports/weekly_YYYY-MM-DD.md` with sections for:
  - summary + innovations
  - training details (hyperparameters/losses if available)
  - architecture
  - pros and cons
  - next steps
  - research ideas
- Per-paper reports are auto-generated on ingest at `./reports/papers/*.md`
- Reading companion flow:
  - Open a paper in your PDF viewer and save highlight annotations
  - In Streamlit, choose the same PDF and click `Load Highlights From PDF`
  - Select a highlighted paragraph to retrieve related concepts from indexed papers
  - Generate an explanation for an `ML researcher`, with optional simplified mode

Note: papers indexed before this schema upgrade may miss some fields; re-index those PDFs to backfill richer report sections.

## Main Files
- `run_watcher.py` — folder watcher process
- `streamlit_app.py` — Streamlit app
- `research_assistant/parser.py` — PDF + equation candidate extraction
- `research_assistant/highlights.py` — PDF highlight paragraph extraction
- `research_assistant/reading_companion.py` — highlight retrieval + explanation workflow
- `research_assistant/arxiv_client.py` — ArXiv discovery + PDF download connector
- `research_assistant/llm_client.py` — local LLM API wrapper
- `research_assistant/vector_store.py` — Chroma persistence/query
- `research_assistant/report.py` — weekly markdown report generator
