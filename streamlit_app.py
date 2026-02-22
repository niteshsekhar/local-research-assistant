from __future__ import annotations

import base64
import html
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from research_assistant.config import get_settings
from research_assistant.embeddings import Embedder
from research_assistant.highlights import extract_highlighted_paragraphs
from research_assistant.llm_client import LocalLLMClient
from research_assistant.pipeline import IngestionPipeline
from research_assistant.reading_companion import ReadingCompanion
from research_assistant.report import generate_weekly_report
from research_assistant.vector_store import PaperStore


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        :root {
          --bg: #f6f7fb;
          --surface: #ffffff;
          --surface-2: #f1f4ff;
          --text: #1c1b1f;
          --muted: #66616f;
          --primary: #6750a4;
          --primary-2: #7f67be;
          --outline: #e2e3eb;
          --shadow-1: 0 1px 3px rgba(27,31,35,0.08), 0 2px 6px rgba(27,31,35,0.06);
          --shadow-2: 0 8px 22px rgba(27,31,35,0.12), 0 2px 8px rgba(27,31,35,0.08);
          --radius: 16px;
        }
        html, body, [class*="css"] { font-family: 'Roboto', sans-serif; color: var(--text); }
        .stApp { background: var(--bg); }
        .block-container { max-width: 1240px; padding-top: 0.55rem; padding-bottom: 2rem; }
        p, span, label, li, div { color: var(--text); }
        header[data-testid="stHeader"] { display: none; }
        div[data-testid="stToolbar"] { display: none; }
        #MainMenu { visibility: hidden; }
        .app-bar {
          background: var(--surface);
          border: 1px solid var(--outline);
          box-shadow: var(--shadow-1);
          border-radius: var(--radius);
          padding: 0.95rem 1.1rem;
          margin-bottom: 0.9rem;
        }
        .app-title { margin: 0; font-size: 1.45rem; font-weight: 700; letter-spacing: 0.1px; }
        .app-subtitle { margin: 0.2rem 0 0 0; color: var(--muted); font-size: 0.92rem; }
        .mat-surface {
          background: var(--surface);
          border: 1px solid var(--outline);
          border-radius: var(--radius);
          box-shadow: var(--shadow-1);
          padding: 0.8rem 1rem;
          margin-bottom: 0.95rem;
        }
        .mat-stat {
          background: var(--surface-2);
          border-radius: 12px;
          border: 1px solid #e5e6f1;
          padding: 0.55rem 0.7rem;
        }
        .mat-stat .k { font-size: 0.75rem; color: var(--muted); margin-bottom: 0.08rem; }
        .mat-stat .v { font-size: 1.05rem; font-weight: 600; }
        .result-card {
          background: var(--surface);
          border: 1px solid var(--outline);
          border-radius: 14px;
          box-shadow: var(--shadow-1);
          padding: 0.8rem 0.9rem;
          margin: 0.45rem 0 0.75rem 0;
        }
        .result-title { margin: 0; font-size: 1.03rem; font-weight: 600; }
        .chip {
          display:inline-flex; align-items:center;
          font-size:0.72rem; line-height:1;
          padding:0.28rem 0.52rem; border-radius:999px;
          background:#efeafd; color:#4c3a7e; border:1px solid #ddd1fb;
          margin-right:0.35rem;
        }
        .chip.muted { background:#f1f2f8; color:#565266; border-color:#e1e2ec; }
        .section-note { color: var(--muted); font-size: 0.9rem; margin-top: -0.15rem; }
        .report-preview {
          background: #ffffff;
          border: 1px solid var(--outline);
          border-radius: 12px;
          padding: 0.65rem 0.8rem;
          box-shadow: var(--shadow-1);
          margin-bottom: 0.9rem;
        }
        .report-preview pre {
          margin: 0;
          white-space: pre-wrap;
          font-size: 0.84rem;
          line-height: 1.45;
          color: #111827;
          background: transparent;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        }
        .companion-card {
          background: #ffffff;
          border: 1px solid var(--outline);
          border-radius: 12px;
          box-shadow: var(--shadow-1);
          padding: 0.75rem 0.9rem;
          margin: 0.45rem 0 0.75rem 0;
        }
        .companion-card h4 {
          margin: 0 0 0.35rem 0;
          color: #111827;
          font-size: 0.95rem;
          font-weight: 600;
        }
        .companion-card p, .companion-card li {
          color: #1f2937;
          margin: 0.18rem 0;
        }
        [data-testid="stSelectbox"] label,
        [data-testid="stCheckbox"] label,
        [data-testid="stToggle"] label {
          color: #1f2937 !important;
          font-weight: 500;
        }
        [data-testid="stSelectbox"] > div,
        [data-testid="stTextInput"] > div,
        [data-testid="stNumberInput"] > div {
          background: #ffffff;
          border-radius: 10px;
        }
        [data-baseweb="select"] > div,
        [data-baseweb="select"] input {
          background: #ffffff !important;
          color: #111827 !important;
        }
        [data-baseweb="select"] svg {
          fill: #374151 !important;
        }
        [data-testid="stSelectbox"] div[role="combobox"] {
          color: #111827 !important;
          background: #ffffff !important;
        }
        [data-baseweb="popover"],
        [data-baseweb="popover"] * {
          background: #ffffff !important;
          color: #111827 !important;
        }
        div[role="listbox"] {
          background: #ffffff !important;
          border: 1px solid #d7dbe7 !important;
          box-shadow: 0 8px 22px rgba(15,23,42,0.12) !important;
        }
        div[role="option"] {
          background: #ffffff !important;
          color: #111827 !important;
        }
        div[role="option"]:hover {
          background: #eef2ff !important;
          color: #111827 !important;
        }
        div[role="option"][aria-selected="true"] {
          background: #e0e7ff !important;
          color: #111827 !important;
        }
        [data-testid="stButton"] > button {
          border-radius: 999px;
          border: 1px solid #d7d2e8;
          background: var(--primary);
          color: #fff;
          font-weight: 500;
          box-shadow: var(--shadow-1);
        }
        [data-testid="stButton"] > button:hover {
          background: var(--primary-2);
          border-color: #c3b9e1;
        }
        div[data-baseweb="tab-list"] {
          gap: 0.28rem;
          display: flex;
          width: 100%;
        }
        button[data-baseweb="tab"] {
          border-radius: 12px;
          background: var(--surface);
          border: 1px solid var(--outline);
          color: #3f3a4a;
          font-weight: 500;
          flex: 1 1 0;
          justify-content: center;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
          border-color: #d7d2e8;
          color: var(--primary);
          background: #f4f0ff;
        }
        [data-testid="stFileUploader"] {
          border:1px dashed #d0c9e8;
          border-radius:14px;
          background:#faf8ff;
        }
        [data-testid="stExpander"] {
          background: var(--surface);
          border: 1px solid var(--outline);
          border-radius: 12px;
        }
        [data-testid="stExpander"] summary, [data-testid="stExpander"] p, [data-testid="stExpander"] div {
          color: var(--text);
        }
        .stCodeBlock, .stCode {
          color: #0f172a;
        }
        [data-testid="stMetric"] { background: transparent !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def build_pipeline() -> tuple[IngestionPipeline, PaperStore, ReadingCompanion, Path, Path]:
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
    companion = ReadingCompanion(store=store, embedder=embedder, llm_client=llm_client)
    return pipeline, store, companion, settings.reports_dir, settings.watch_dir


def render_pdf_viewer(pdf_path: Path) -> None:
    if not pdf_path.exists():
        st.warning(f"PDF not found: {pdf_path}")
        return
    payload = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
    iframe = (
        f'<iframe src="data:application/pdf;base64,{payload}" '
        'width="100%" height="700" type="application/pdf"></iframe>'
    )
    components.html(iframe, height=720, scrolling=True)


def render_report_preview(report_file: Path) -> None:
    content = html.escape(report_file.read_text(encoding="utf-8")[:2200])
    st.markdown(f"**{report_file.name}**")
    st.markdown(
        f"""
        <div class="report-preview">
          <pre>{content}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Local Research Assistant", layout="wide")
apply_styles()

pipeline, store, companion, reports_dir, watch_dir = build_pipeline()
pdf_files = sorted(watch_dir.glob("*.pdf"))
weekly_reports = sorted(reports_dir.glob("weekly_*.md"), reverse=True)
paper_reports = sorted((reports_dir / "papers").glob("*.md"), reverse=True)

st.markdown(
    """
    <div class="app-bar">
      <p class="app-title">📚 Local Reading Companion</p>
      <p class="app-subtitle">Personal research assistant</p>
    </div>
    """,
    unsafe_allow_html=True,
)

bar_a, bar_b, bar_c, bar_d = st.columns([1.3, 1, 1, 1.1])
with bar_a:
    query_limit = st.slider("Search results", min_value=3, max_value=25, value=10)
with bar_b:
    st.markdown(f"<div class='mat-stat'><div class='k'>PDFs</div><div class='v'>{len(pdf_files)}</div></div>", unsafe_allow_html=True)
with bar_c:
    st.markdown(
        f"<div class='mat-stat'><div class='k'>Paper Reports</div><div class='v'>{len(paper_reports)}</div></div>",
        unsafe_allow_html=True,
    )
with bar_d:
    st.caption("")
    if st.button("Generate Weekly Report"):
        report_path = generate_weekly_report(store, reports_dir)
        st.success(f"Weekly report created: {report_path.name}")

tab_ingest, tab_search, tab_reports, tab_companion = st.tabs(
    ["📥 Ingest", "🔎 Search", "📝 Reports", "🧠 Reading Companion"]
)

with tab_ingest:
    st.subheader("Ingest a PDF")
    st.markdown("<p class='section-note'>Upload a PDF to parse, analyze, embed, and index it locally.</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop a PDF", type=["pdf"])
    if uploaded is not None:
        temp_path = Path("papers") / uploaded.name
        temp_path.write_bytes(uploaded.read())
        try:
            with st.spinner("Analyzing and indexing..."):
                msg = pipeline.ingest_pdf(temp_path)
            st.success(msg)
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")

with tab_search:
    st.subheader("Semantic Paper Search")
    st.markdown("<p class='section-note'>Example: show me all papers related to token routing</p>", unsafe_allow_html=True)
    query = st.text_input("Search query", placeholder="e.g. token routing in sparse MoE")
    if st.button("Search papers"):
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            try:
                with st.spinner("Searching..."):
                    results = pipeline.query(query.strip(), limit=query_limit)
            except Exception as exc:
                st.error(f"Search failed: {exc}")
                results = []

            if not results:
                st.info("No results found.")
            for idx, row in enumerate(results, start=1):
                meta = row["metadata"]
                score = row["score"]
                method = meta.get("method_type", "other")
                st.markdown(
                    f"""
                    <div class="result-card">
                      <p class="result-title">{idx}. {meta.get('title', 'Untitled')}</p>
                      <span class="chip">score {score}</span>
                      <span class="chip muted">{method}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("Open details", expanded=(idx == 1)):
                    st.write(f"**Summary:** {meta.get('summary', '')}")
                    st.write(f"**Innovations:** {meta.get('innovations', '').replace(' || ', '; ')}")
                    st.write(f"**Training:** {meta.get('training_info', '').replace(' || ', '; ')}")
                    st.write(f"**Architecture:** {meta.get('architecture', '')}")
                    st.write(f"**Contributions:** {meta.get('contributions', '').replace(' || ', '; ')}")
                    st.write(f"**Pros:** {meta.get('pros', '').replace(' || ', '; ')}")
                    st.write(f"**Cons:** {meta.get('cons', '').replace(' || ', '; ')}")
                    st.write(f"**Next Steps:** {meta.get('next_steps', '').replace(' || ', '; ')}")
                    st.write(f"**Ideas:** {meta.get('research_ideas', '').replace(' || ', '; ')}")
                    st.caption("Source: Indexed paper")

with tab_reports:
    left, right = st.columns(2)
    with left:
        st.subheader("Weekly Reports")
        if not weekly_reports:
            st.info("No weekly reports yet.")
        for report_file in weekly_reports[:10]:
            render_report_preview(report_file)
    with right:
        st.subheader("Per-Paper Reports")
        if not paper_reports:
            st.info("No per-paper reports yet.")
        for report_file in paper_reports[:10]:
            render_report_preview(report_file)

with tab_companion:
    st.subheader("Reading Companion (Highlights)")
    if not pdf_files:
        st.info("No PDFs found yet.")
    else:
        ctl1, ctl2, ctl3 = st.columns([2.6, 1.2, 1.2])
        with ctl1:
            selected_pdf = st.selectbox("Select a PDF", options=pdf_files, format_func=lambda path: path.name)
        with ctl2:
            st.caption("")
            st.caption("")
            load_clicked = st.button("Load Highlights")
        with ctl3:
            st.caption("")
            show_simplified = st.toggle("Simplified mode", value=False)
        show_pdf_preview = st.toggle("Show PDF preview", value=False, help="Enable to view the selected paper inline.")

        if load_clicked:
            try:
                st.session_state["highlights"] = extract_highlighted_paragraphs(selected_pdf)
                st.success(f"Loaded {len(st.session_state['highlights'])} highlights.")
            except Exception as exc:
                st.error(f"Failed to read highlights: {exc}")

        if show_pdf_preview:
            render_pdf_viewer(selected_pdf)

        highlights = st.session_state.get("highlights", [])
        if highlights:
            chosen = st.selectbox(
                "Choose highlighted paragraph",
                options=list(range(len(highlights))),
                format_func=lambda idx: f"Page {highlights[idx].page}: {highlights[idx].text[:140]}...",
            )
            if st.button("Explain Highlight"):
                with st.spinner("Retrieving related concepts and generating explanation..."):
                    try:
                        response = companion.explain(
                            highlight=highlights[chosen],
                            expertise_level="ML researcher",
                            include_simplified=show_simplified,
                        )
                    except Exception as exc:
                        st.error(f"Reading companion failed: {exc}")
                        response = None

                if response is not None:
                    st.markdown(
                        f"""
                        <div class="companion-card">
                          <h4>Highlighted Paragraph</h4>
                          <p>{html.escape(response.highlight.text)}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                        <div class="companion-card">
                          <h4>Expert Explanation (ML Researcher)</h4>
                          <p>{html.escape(response.expert_explanation or "No explanation generated.")}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if show_simplified:
                        st.markdown(
                            f"""
                            <div class="companion-card">
                              <h4>Simplified Explanation</h4>
                              <p>{html.escape(response.simplified_explanation or "No simplified explanation generated.")}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    related_list = "".join(f"<li>{html.escape(item)}</li>" for item in response.related_concepts)
                    papers_list = "".join(f"<li>{html.escape(title)}</li>" for title in response.retrieved_papers)
                    st.markdown(
                        f"""
                        <div class="companion-card">
                          <h4>Related Concepts</h4>
                          <ul>{related_list}</ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                        <div class="companion-card">
                          <h4>Retrieved Papers</h4>
                          <ul>{papers_list}</ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
