from __future__ import annotations

from pathlib import Path

import streamlit as st

from research_assistant.config import get_settings
from research_assistant.embeddings import Embedder
from research_assistant.llm_client import LocalLLMClient
from research_assistant.pipeline import IngestionPipeline
from research_assistant.report import generate_weekly_report
from research_assistant.vector_store import PaperStore


@st.cache_resource
def build_pipeline() -> tuple[IngestionPipeline, PaperStore, Path]:
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
    return pipeline, store, settings.reports_dir


st.set_page_config(page_title="Local Research Assistant", layout="wide")
st.title("📚 Local Research Assistant")

pipeline, store, reports_dir = build_pipeline()

with st.sidebar:
    st.header("Actions")
    query_limit = st.slider("Results", min_value=3, max_value=25, value=10)
    if st.button("Generate Weekly Report"):
        report_path = generate_weekly_report(store, reports_dir)
        st.success(f"Weekly report written: {report_path}")

st.subheader("Ingest a PDF manually")
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

st.subheader("Query your paper memory")
query = st.text_input("Example: show me all papers related to token routing")
if st.button("Search") and query.strip():
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
        st.markdown(f"### {idx}. {meta.get('title', 'Untitled')}")
        st.write(f"**Score:** {row['score']}")
        st.write(f"**Method:** {meta.get('method_type', 'other')}")
        st.write(f"**Summary:** {meta.get('summary', '')}")
        st.write(f"**Innovations:** {meta.get('innovations', '').replace(' || ', '; ')}")
        st.write(f"**Training:** {meta.get('training_info', '').replace(' || ', '; ')}")
        st.write(f"**Architecture:** {meta.get('architecture', '')}")
        st.write(f"**Contributions:** {meta.get('contributions', '').replace(' || ', '; ')}")
        st.write(f"**Pros:** {meta.get('pros', '').replace(' || ', '; ')}")
        st.write(f"**Cons:** {meta.get('cons', '').replace(' || ', '; ')}")
        st.write(f"**Next Steps:** {meta.get('next_steps', '').replace(' || ', '; ')}")
        st.write(f"**Ideas:** {meta.get('research_ideas', '').replace(' || ', '; ')}")
        st.caption(meta.get("file_path", ""))

st.subheader("Recent weekly reports")
for report_file in sorted(reports_dir.glob("weekly_*.md"), reverse=True)[:10]:
    st.markdown(f"- `{report_file}`")
    st.code(report_file.read_text(encoding="utf-8")[:2000], language="markdown")

st.subheader("Recent per-paper reports")
for report_file in sorted((reports_dir / "papers").glob("*.md"), reverse=True)[:10]:
    st.markdown(f"- `{report_file}`")
    st.code(report_file.read_text(encoding="utf-8")[:2000], language="markdown")
