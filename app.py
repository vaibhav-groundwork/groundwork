"""
app.py — Groundwork Streamlit application.

Slice 2 scope: research mode + document mode (single question each).
  - Research mode: web research loop via run_research (unchanged from Slice 1)
  - Document mode: PDF upload, ingestion, and single-question RAG via rag_node
  - Mode toggle in sidebar; each mode maintains independent session state

Deferred to later passes:
  - Multi-turn question accumulation within a session
  - Synthesis agent (brief generation)
  - Judge agent (quality scoring)
  - Export (DOCX / PPTX generation and download)
"""

import os
import tempfile
import uuid

import streamlit as st

# st.set_page_config must be the absolute first Streamlit call in the file.
# Any st.* usage before this — including inside functions called before this
# line — will raise a StreamlitAPIException.
st.set_page_config(
    page_title="Groundwork",
    page_icon="📋",
    layout="wide",
)

from src.agents.rag_agent import rag_node  # noqa: E402
from src.config import MAX_FILE_SIZE_MB  # noqa: E402
from src.graph import run_research  # noqa: E402
from src.ingestion import get_chroma_collection, ingest_document  # noqa: E402
from src.tracing import setup_tracing  # noqa: E402

# setup_tracing() prints to the server terminal, not the browser — expected.
setup_tracing()


# ── Shared helpers ────────────────────────────────────────────────────────────

def render_sources(sources: list[dict]) -> None:
    """
    Renders an expander containing a clickable list of sources.
    Handles both research-mode dicts (title + link) and RAG-mode dicts
    (source filename + text) from a single shared call site.
    """
    with st.expander(f"📎 Sources ({len(sources)})"):
        for s in sources:
            title = s.get("title") or s.get("source") or "Unknown source"
            link = s.get("link", "")
            if link:
                st.markdown(f"- [{title}]({link})")
            else:
                st.markdown(f"- {title}")


# ── Session state initialisation ──────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]

if "mode" not in st.session_state:
    st.session_state.mode = "Research"

# Research mode state
if "research_result" not in st.session_state:
    st.session_state.research_result = None

if "current_topic" not in st.session_state:
    st.session_state.current_topic = None

# Document mode state
if "collection" not in st.session_state:
    st.session_state.collection = get_chroma_collection(st.session_state.session_id)

if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []

if "document_result" not in st.session_state:
    st.session_state.document_result = None

if "document_question" not in st.session_state:
    st.session_state.document_question = None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Groundwork")
    st.caption(
        "Ask as many questions as you need, then generate a report that's "
        "audited for accuracy before you export it."
    )
    st.radio("Mode", options=["Research", "Document"], key="mode")
    st.divider()
    st.caption("🔒 Private session")


# ── Main panel ────────────────────────────────────────────────────────────────

# ── Research mode ─────────────────────────────────────────────────────────────

if st.session_state.mode == "Research":

    if st.session_state.research_result is None:

        st.markdown("## What would you like to research?")
        st.markdown(
            "Enter a topic below and Groundwork will search the web, identify gaps, "
            "and compile structured research notes."
        )

        form_placeholder = st.empty()

        with form_placeholder.form(key="research_form", clear_on_submit=False, border=False):
            topic_input = st.text_input(
                label="Research topic",
                placeholder="e.g. The impact of AI on the legal profession",
                label_visibility="collapsed",
            )
            submit = st.form_submit_button("Research", type="primary")

        if submit:
            topic = topic_input.strip()
            if not topic:
                st.warning("Enter a topic to research.")
            else:
                form_placeholder.empty()
                with st.spinner("Researching — this may take a minute…"):
                    result = run_research(topic)

                st.session_state.research_result = result
                st.session_state.current_topic = topic
                st.rerun()

    else:

        topic = st.session_state.current_topic
        result = st.session_state.research_result

        st.markdown(f"## {topic}")

        status = result.get("status_message", "")
        if status.startswith("⚠️"):
            st.warning(status)
        elif status:
            st.success(status)

        with st.container():
            notes = result.get("research_notes", "")
            st.markdown(notes if notes else "*No research notes were produced.*")

        sources = result.get("sources", [])
        if sources:
            render_sources(sources)

        st.divider()

        if st.button("Ask another question"):
            st.session_state.research_result = None
            st.session_state.current_topic = None
            st.rerun()


# ── Document mode ─────────────────────────────────────────────────────────────

elif st.session_state.mode == "Document":

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # Process any newly uploaded files
    if uploaded_files:
        for file in uploaded_files:
            if file.name in st.session_state.uploaded_filenames:
                continue

            file_bytes = file.getvalue()
            size_limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

            if len(file_bytes) > size_limit_bytes:
                st.error(
                    f"'{file.name}' exceeds the {MAX_FILE_SIZE_MB} MB size limit — skipped."
                )
                continue

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                ingest_result = ingest_document(
                    tmp_path, st.session_state.collection, display_name=file.name
                )
                
                if ingest_result["status"] == "indexed":
                    st.success(f"'{file.name}' indexed.")
                else:
                    st.info(f"'{file.name}' was already indexed.")

                st.session_state.uploaded_filenames.append(file.name)

            except Exception as exc:
                st.error(f"Failed to ingest '{file.name}' — {exc}")

            # Clean up temp file in its own try/except — a deletion failure
            # must never crash the upload flow or block the next file.
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ── Post-upload flow ──────────────────────────────────────────────────────

    if st.session_state.uploaded_filenames:

        st.caption(
            "Indexed: " + ", ".join(st.session_state.uploaded_filenames)
        )

        if st.session_state.document_result is None:

            form_placeholder = st.empty()

            with form_placeholder.form(key="document_form", clear_on_submit=False, border=False):
                question_input = st.text_input(
                    label="Question",
                    placeholder="e.g. What are the main conclusions?",
                    label_visibility="collapsed",
                )
                ask = st.form_submit_button("Ask", type="primary")

            if ask:
                question = question_input.strip()
                if not question:
                    st.warning("Enter a question to ask.")
                else:
                    form_placeholder.empty()
                    with st.spinner("Searching your documents…"):
                        doc_result = rag_node({
                            "question": question,
                            "collection": st.session_state.collection,
                        })

                    st.session_state.document_result = doc_result
                    st.session_state.document_question = question
                    st.rerun()

        else:

            question = st.session_state.document_question
            doc_result = st.session_state.document_result

            st.markdown(f"## {question}")

            status = doc_result.get("status_message", "")
            if status.startswith("⚠️"):
                st.warning(status)
            elif status:
                st.success(status)

            with st.container():
                answer = doc_result.get("answer", "")
                st.markdown(answer if answer else "*No answer was produced.*")

            sources = doc_result.get("sources", [])
            if sources:
                render_sources(sources)

            st.divider()

            if st.button("Ask another question"):
                st.session_state.document_result = None
                st.session_state.document_question = None
                st.rerun()

    else:

        st.markdown("## Upload a document to get started")
        st.markdown(
            "Upload one or more PDF files above and Groundwork will index them "
            "so you can ask questions grounded in their content."
        )
