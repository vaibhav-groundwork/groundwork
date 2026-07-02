"""
app.py — Groundwork Streamlit application.

Slice 2 scope: research mode + document mode (single question each).
  - Research mode: web research loop via run_research (unchanged from Slice 1)
  - Document mode: PDF upload, ingestion, and single-question RAG via rag_node
  - Mode toggle in sidebar; each mode maintains independent session state

Slice 3 scope: multi-turn accumulation across both modes.
  - Research mode and Document mode maintain fully independent accumulation,
    chat-style history, and report-triggering — research_material/
    research_sources/research_report_flag and document_material/
    document_sources/document_report_flag are never combined, since document
    mode's groundedness guarantee (answers strictly from uploaded files) must
    never blend with research mode's web-sourced material.
  - Each mode renders its own scrolling history and 'Generate Full Report'
    button inside its own section; the in-progress current result renders below.
  - generate_report flags are set on click; report generation logic deferred
    to Slice 4.
  - Follow-up research questions are routed intelligently — answered directly
    from accumulated context when possible, or triggering genuine new web
    search when the question covers new ground, via route_followup_node and
    answer_from_notes_node in the research graph.

Slice 4 scope: Generate Full Report wiring. Each mode independently calls
synthesis_node then judge_node exactly once per report (guarded against
re-running on every Streamlit rerun via research_report/document_report being
None-checked before generation). Full score breakdown (all four dimensions with
reasoning) shown as a table by default. Once generated, a report is a snapshot
— asking further questions afterward does not auto-regenerate it. Regeneration
(using previous_report and new context, or judge feedback) and export are
deferred to later passes.

Slice 5 scope: Regenerate Report button per mode, independently. Each mode
exposes a 'Regenerate Report' button below the judge score table, capped at 1
regeneration per mode (research_regen_count / document_regen_count). On click,
the full current accumulated history at the time of click is used to rebuild
source_material. synthesis_node is called with the existing brief as
previous_report and the weakest-scoring dimension name and its reasoning as
weakest_dimension and weakest_dimension_feedback (weakest dimension determined
by min() over scores dict). judge_node is then called on the new brief.
The original report (research_report/research_judge_result or
document_report/document_judge_result) is never overwritten — the regenerated
result is stored in research_report_regen/research_judge_regen or
document_report_regen/document_judge_regen and rendered below the original,
with a one-liner showing which dimension was targeted and the before/after
overall score. The other mode's state is never touched. After the cap is
reached, a caption replaces the button.

Slice 6 scope: Export. Below each mode's report, users select version
  (Original or Regenerated, if available) and format (DOCX or PPTX), then
  click Download Report. The exporter is called once and the resulting bytes
  cached in session state (research_export_bytes / document_export_bytes)
  keyed by a version+format composite string — switching selection correctly
  invalidates the cache and re-exports. The two modes are fully independent.
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

st.markdown("""
<style>
:root {
    --primary-color: #6366F1 !important;
}
/* ── Page and background ─────────────────────────────── */
.stApp {
    background-color: #0F0F1A;
}

section[data-testid="stSidebar"] {
    background-color: #16162A;
    border-right: 1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem;
}

/* ── Main content area ───────────────────────────────── */
.block-container {
    padding: 2rem 2.5rem;
    max-width: 860px;
}

/* ── Typography ──────────────────────────────────────── */
h1, h2, h3, h4 {
    color: #E8E8F0 !important;
    letter-spacing: -0.3px;
}

h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}

p, li, .stMarkdown {
    color: #B0B0C0;
    line-height: 1.65;
    font-size: 0.95rem;
}

/* ── Buttons ─────────────────────────────────────────── */
button[kind="primaryFormSubmit"],
button[kind="secondary"],
.stButton button,
.stButton > button,
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {
    background-color: #6366F1 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1.25rem !important;
    transition: background-color 0.15s ease !important;
}

[data-testid="baseButton-secondary"]:hover,
[data-testid="baseButton-primary"]:hover,
.stButton button:hover {
    background-color: #4F52D6 !important;
}

/* ── Download button ─────────────────────────────────── */
.stDownloadButton > button,
[data-testid="baseButton-secondary"].stDownloadButton {
    background-color: transparent !important;
    color: #6366F1 !important;
    border: 1px solid #6366F1 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    transition: background-color 0.15s ease !important;
}

.stDownloadButton > button:hover {
    background-color: rgba(99,102,241,0.1) !important;
}

/* ── Text inputs ─────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-baseweb="input"] input,
input[type="text"],
input[aria-label] {
    background-color: #1E1E32 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #E8E8F0 !important;
    font-size: 0.9rem !important;
}

[data-testid="stTextInput"] input:focus,
[data-baseweb="input"] input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* ── Radio buttons (mode toggle) ─────────────────────── */
.stRadio > div {
    gap: 0.5rem;
}

.stRadio > div > label {
    background-color: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.9rem !important;
    color: #888 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
    width: 100%;
}

.stRadio > div > label:hover {
    border-color: rgba(99,102,241,0.4) !important;
    color: #E8E8F0 !important;
}

.stRadio [aria-checked="true"] > div,
div[data-baseweb="radio"] input:checked ~ div {
    background-color: #6366F1 !important;
    color: #ffffff !important;
    border-color: #6366F1 !important;
}

/* ── Selectbox ───────────────────────────────────────── */
.stSelectbox > div > div {
    background-color: #1E1E32 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #E8E8F0 !important;
}

/* ── Expander (sources) ──────────────────────────────── */
.streamlit-expanderHeader {
    background-color: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 8px !important;
    color: #6366F1 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}

.streamlit-expanderContent {
    background-color: #16162A !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Dividers ────────────────────────────────────────── */
hr {
    border-color: rgba(255,255,255,0.06) !important;
    margin: 1.25rem 0 !important;
}

/* ── Captions ────────────────────────────────────────── */
.stCaption, small {
    color: #666680 !important;
    font-size: 0.8rem !important;
}

/* ── Success / warning / error messages ──────────────── */
.stSuccess {
    background-color: rgba(45,212,191,0.1) !important;
    border: 1px solid rgba(45,212,191,0.3) !important;
    border-radius: 8px !important;
    color: #2DD4BF !important;
}

.stWarning {
    background-color: rgba(251,191,36,0.08) !important;
    border: 1px solid rgba(251,191,36,0.25) !important;
    border-radius: 8px !important;
}

.stError {
    background-color: rgba(239,68,68,0.08) !important;
    border: 1px solid rgba(239,68,68,0.25) !important;
    border-radius: 8px !important;
}

/* ── Spinner ─────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: #6366F1 !important;
}

/* ── Table (judge scores) ────────────────────────────── */
.stTable table {
    background-color: #16162A !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    font-size: 0.875rem !important;
}

.stTable th {
    background-color: rgba(99,102,241,0.12) !important;
    color: #6366F1 !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1rem !important;
}

.stTable td {
    color: #B0B0C0 !important;
    padding: 0.6rem 1rem !important;
    border-color: rgba(255,255,255,0.04) !important;
}

.stTable tr:hover td {
    background-color: rgba(99,102,241,0.05) !important;
}

/* ── File uploader ───────────────────────────────────── */
.stFileUploader {
    background-color: #1E1E32 !important;
    border: 1px dashed rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
}

/* ── Hide Streamlit default chrome ───────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

from src.agents.rag_agent import rag_node  # noqa: E402
from src.agents.synthesis_agent import synthesis_node  # noqa: E402
from src.agents.judge_agent import judge_node  # noqa: E402
from src.exporters.docx_export import export_to_docx  # noqa: E402
from src.exporters.pptx_export import export_to_pptx  # noqa: E402
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


def render_history_entry(entry: dict) -> None:
    """Renders one accumulated Q&A entry in chat-style format."""
    if entry["mode"] == "research":
        st.markdown("""
<span style="background:rgba(99,102,241,0.15); color:#6366F1; font-size:0.75rem; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid rgba(99,102,241,0.3); display:inline-block; margin-bottom:8px;">🔍 Research</span>
""", unsafe_allow_html=True)
        st.markdown(f"### {entry['topic']}")
        st.markdown(entry["notes"])
    else:
        st.markdown("""
<span style="background:rgba(255,75,75,0.15); color:#FF4B4B; font-size:0.75rem; font-weight:600; padding:3px 10px; border-radius:20px; border:1px solid rgba(255,75,75,0.3); display:inline-block; margin-bottom:8px;">📄 Document</span>
""", unsafe_allow_html=True)
        st.markdown(f"### {entry['question']}")
        st.markdown(entry["answer"])
    if entry.get("sources"):
        render_sources(entry["sources"])
    st.divider()


def get_research_context() -> str:
    """
    Concatenates all accumulated research notes from this session into one
    context string, used to let follow-up questions route intelligently
    instead of always triggering a fresh, contextless web search.
    Returns an empty string if no research questions have been asked yet.
    """
    if not st.session_state.research_material:
        return ""
    return "\n\n---\n\n".join(
        f"Topic: {e['topic']}\n{e['notes']}" for e in st.session_state.research_material
    )


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

# Multi-turn accumulation state — fully independent per mode, never combined
if "research_material" not in st.session_state:
    st.session_state.research_material = []

if "research_sources" not in st.session_state:
    st.session_state.research_sources = []

if "research_report_flag" not in st.session_state:
    st.session_state.research_report_flag = False

if "research_report" not in st.session_state:
    st.session_state.research_report = None

if "research_judge_result" not in st.session_state:
    st.session_state.research_judge_result = None

if "research_regen_count" not in st.session_state:
    st.session_state.research_regen_count = 0

if "research_report_regen" not in st.session_state:
    st.session_state.research_report_regen = None

if "research_judge_regen" not in st.session_state:
    st.session_state.research_judge_regen = None

if "research_export_bytes" not in st.session_state:
    st.session_state.research_export_bytes = None

if "research_export_format" not in st.session_state:
    st.session_state.research_export_format = None

if "document_material" not in st.session_state:
    st.session_state.document_material = []

if "document_sources" not in st.session_state:
    st.session_state.document_sources = []

if "document_report_flag" not in st.session_state:
    st.session_state.document_report_flag = False

if "document_report" not in st.session_state:
    st.session_state.document_report = None

if "document_judge_result" not in st.session_state:
    st.session_state.document_judge_result = None

if "document_regen_count" not in st.session_state:
    st.session_state.document_regen_count = 0

if "document_report_regen" not in st.session_state:
    st.session_state.document_report_regen = None

if "document_judge_regen" not in st.session_state:
    st.session_state.document_judge_regen = None

if "document_export_bytes" not in st.session_state:
    st.session_state.document_export_bytes = None

if "document_export_format" not in st.session_state:
    st.session_state.document_export_format = None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
    <div style="display:flex; flex-direction:column; align-items:center; gap:3px; flex-shrink:0;">
        <div style="width:12px; height:3px; background:#6366F1; border-radius:2px; opacity:0.4;"></div>
        <div style="width:18px; height:3px; background:#6366F1; border-radius:2px; opacity:0.7;"></div>
        <div style="width:24px; height:3px; background:#6366F1; border-radius:2px;"></div>
    </div>
    <span style="font-size:1.5rem; font-weight:700; color:#E8E8F0; letter-spacing:-0.3px;">Groundwork</span>
</div>
<div style="font-size:0.85rem; color:#6366F1; font-weight:500; margin-bottom:0.5rem; padding-left:30px;">AI Research Assistant</div>
""", unsafe_allow_html=True)
    st.caption(
        "Ask as many questions as you need, then generate a report that's "
        "audited for accuracy before you export it."
    )
    st.radio("Mode", options=["Research", "Document"], key="mode")
    st.markdown("""
<div style="padding-top:1rem; border-top:1px solid rgba(255,255,255,0.06); display:flex; align-items:center; gap:6px;">
    <span style="width:6px; height:6px; background:#FF4B4B; border-radius:50%; display:inline-block; flex-shrink:0;"></span>
    <span style="font-size:0.75rem; color:#555570;">Private session</span>
</div>
""", unsafe_allow_html=True)


# ── Main panel ────────────────────────────────────────────────────────────────

# ── Research mode ─────────────────────────────────────────────────────────────

if st.session_state.mode == "Research":

    for entry in st.session_state.research_material:
        render_history_entry(entry)

    if len(st.session_state.research_material) > 0:
        st.caption(f"📋 {len(st.session_state.research_material)} question(s) gathered")
        if not st.session_state.research_report_flag:
            if st.button("Generate Full Report", key="research_generate_report"):
                st.session_state.research_report_flag = True
                st.rerun()

        if st.session_state.research_report_flag and st.session_state.research_report is None:
            source_material = "\n\n---\n\n".join(
                f"Topic: {e['topic']}\n{e['notes']}" for e in st.session_state.research_material
            )
            with st.spinner("Generating your report…"):
                synthesis_result = synthesis_node({
                    "topic": "Research findings",
                    "source_material": source_material,
                    "source_list": st.session_state.research_sources,
                    "previous_report": "",
                    "weakest_dimension": "",
                    "weakest_dimension_feedback": "",
                })
                judge_result = judge_node({
                    "topic": "Research findings",
                    "source_material": source_material,
                    "report": synthesis_result["brief"],
                })
            st.session_state.research_report = synthesis_result
            st.session_state.research_judge_result = judge_result
            st.rerun()

        if st.session_state.research_report is not None:
            st.markdown(st.session_state.research_report["brief"])
            if st.session_state.research_report.get("sources_section"):
                st.markdown(st.session_state.research_report["sources_section"])

            judge = st.session_state.research_judge_result
            st.divider()
            st.markdown(f"### Quality Score: {judge['overall_score']}/5.0")
            st.caption(judge["score_explanation"])
            st.table([
                {"Dimension": dim.capitalize(), "Score": f"{data['score']}/5", "Reasoning": data["reasoning"]}
                for dim, data in judge["scores"].items()
            ])

            if st.session_state.research_regen_count < 1:
                if st.button("Regenerate Report", key="research_regenerate_report"):
                    source_material = "\n\n---\n\n".join(
                        f"Topic: {e['topic']}\n{e['notes']}" for e in st.session_state.research_material
                    )
                    weakest_dim = min(
                        st.session_state.research_judge_result["scores"],
                        key=lambda d: st.session_state.research_judge_result["scores"][d]["score"],
                    )
                    weakest_feedback = st.session_state.research_judge_result["scores"][weakest_dim]["reasoning"]
                    with st.spinner("Regenerating your report…"):
                        new_synthesis = synthesis_node({
                            "topic": "Research findings",
                            "source_material": source_material,
                            "source_list": st.session_state.research_sources,
                            "previous_report": st.session_state.research_report["brief"],
                            "weakest_dimension": weakest_dim,
                            "weakest_dimension_feedback": weakest_feedback,
                        })
                        new_judge = judge_node({
                            "topic": "Research findings",
                            "source_material": source_material,
                            "report": new_synthesis["brief"],
                        })
                    st.session_state.research_report_regen = new_synthesis
                    st.session_state.research_judge_regen = new_judge
                    st.session_state.research_regen_count += 1
                    st.rerun()
            else:
                st.caption("Maximum regenerations reached.")

            if st.session_state.research_report_regen is not None:
                st.divider()
                st.markdown("#### 📈 Regenerated Report")
                weakest_dim = min(
                    st.session_state.research_judge_result["scores"],
                    key=lambda d: st.session_state.research_judge_result["scores"][d]["score"],
                )
                original_score = st.session_state.research_judge_result["overall_score"]
                regen_score = st.session_state.research_judge_regen["overall_score"]
                st.caption(f"Targeted **{weakest_dim}** dimension — overall score {original_score}/5.0 → {regen_score}/5.0")
                st.markdown(st.session_state.research_report_regen["brief"])
                if st.session_state.research_report_regen.get("sources_section"):
                    st.markdown(st.session_state.research_report_regen["sources_section"])
                st.divider()
                st.markdown(f"#### Updated Quality Score: {st.session_state.research_judge_regen['overall_score']}/5.0")
                st.caption(st.session_state.research_judge_regen["score_explanation"])
                st.table([
                    {"Dimension": dim.capitalize(), "Score": f"{data['score']}/5", "Reasoning": data["reasoning"]}
                    for dim, data in st.session_state.research_judge_regen["scores"].items()
                ])

            version_options = (
                ["Original Report", "Regenerated Report"]
                if st.session_state.research_report_regen is not None
                else ["Original Report"]
            )
            st.divider()
            st.markdown("#### Export Report")
            export_version = st.radio("Version", options=version_options, horizontal=True, key="research_export_version")
            export_format = st.selectbox("Format", options=["DOCX", "PPTX"], key="research_export_format_select")
            cache_key = f"{export_format}-{export_version}"
            if st.session_state.research_export_format != cache_key:
                st.session_state.research_export_bytes = None
                st.session_state.research_export_format = None
            if st.button("Download Report", key="research_download_btn"):
                if "Regenerated" in export_version:
                    brief = st.session_state.research_report_regen["brief"]
                    sources_section = st.session_state.research_report_regen.get("sources_section", "")
                else:
                    brief = st.session_state.research_report["brief"]
                    sources_section = st.session_state.research_report.get("sources_section", "")
                with st.spinner("Preparing export…"):
                    if export_format == "DOCX":
                        path = export_to_docx(topic="Research findings", brief=brief, sources_section=sources_section)
                    else:
                        path = export_to_pptx(topic="Research findings", brief=brief, sources_section=sources_section)
                if path is not None:
                    with open(path, "rb") as f:
                        st.session_state.research_export_bytes = f.read()
                    st.session_state.research_export_format = cache_key
                    st.rerun()
                else:
                    st.error("Export failed — please try again.")
            if st.session_state.research_export_bytes is not None:
                mime = (
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    if export_format == "DOCX"
                    else "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )
                st.download_button(
                    label=f"Click to download {export_format}",
                    data=st.session_state.research_export_bytes,
                    file_name=f"groundwork_{st.session_state.research_material[0]['topic'].lower().replace(' ', '_')[:40]}.{export_format.lower()}",
                    mime=mime,
                    key=f"research_download_file_{export_format}",
                )

        st.divider()

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
                    result = run_research(topic, prior_context=get_research_context())
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

        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button("Ask another question", key="research_ask_another"):
                if st.session_state.research_result is not None:
                    st.session_state.research_material.append({
                        "mode": "research",
                        "topic": st.session_state.current_topic,
                        "notes": st.session_state.research_result["research_notes"],
                        "sources": st.session_state.research_result.get("sources", []),
                    })
                    existing_urls = {s.get("url") or s.get("link") for s in st.session_state.research_sources}
                    for source in st.session_state.research_result.get("sources", []):
                        key = source.get("url") or source.get("link")
                        if key not in existing_urls:
                            st.session_state.research_sources.append(source)
                            existing_urls.add(key)
                st.session_state.research_result = None
                st.session_state.current_topic = None
                st.rerun()

        with col2:
            if not st.session_state.research_report_flag:
                if st.button("Generate Full Report", key="research_generate_from_result"):
                    if st.session_state.research_result is not None:
                        st.session_state.research_material.append({
                            "mode": "research",
                            "topic": st.session_state.current_topic,
                            "notes": st.session_state.research_result["research_notes"],
                            "sources": st.session_state.research_result.get("sources", []),
                        })
                        existing_urls = {s.get("url") or s.get("link") for s in st.session_state.research_sources}
                        for source in st.session_state.research_result.get("sources", []):
                            key = source.get("url") or source.get("link")
                            if key not in existing_urls:
                                st.session_state.research_sources.append(source)
                                existing_urls.add(key)
                    st.session_state.research_result = None
                    st.session_state.current_topic = None
                    st.session_state.research_report_flag = True
                    st.rerun()


# ── Document mode ─────────────────────────────────────────────────────────────

elif st.session_state.mode == "Document":

    for entry in st.session_state.document_material:
        render_history_entry(entry)

    if len(st.session_state.document_material) > 0:
        st.caption(f"📋 {len(st.session_state.document_material)} question(s) gathered")
        if not st.session_state.document_report_flag:
            if st.button("Generate Full Report", key="document_generate_report"):
                st.session_state.document_report_flag = True
                st.rerun()

        if st.session_state.document_report_flag and st.session_state.document_report is None:
            source_material = "\n\n---\n\n".join(
                f"Question: {e['question']}\n{e['answer']}" for e in st.session_state.document_material
            )
            with st.spinner("Generating your report…"):
                synthesis_result = synthesis_node({
                    "topic": "Document findings",
                    "source_material": source_material,
                    "source_list": st.session_state.document_sources,
                    "previous_report": "",
                    "weakest_dimension": "",
                    "weakest_dimension_feedback": "",
                })
                judge_result = judge_node({
                    "topic": "Document findings",
                    "source_material": source_material,
                    "report": synthesis_result["brief"],
                })
            st.session_state.document_report = synthesis_result
            st.session_state.document_judge_result = judge_result
            st.rerun()

        if st.session_state.document_report is not None:
            st.markdown(st.session_state.document_report["brief"])
            if st.session_state.document_report.get("sources_section"):
                st.markdown(st.session_state.document_report["sources_section"])

            judge = st.session_state.document_judge_result
            st.divider()
            st.markdown(f"### Quality Score: {judge['overall_score']}/5.0")
            st.caption(judge["score_explanation"])
            st.table([
                {"Dimension": dim.capitalize(), "Score": f"{data['score']}/5", "Reasoning": data["reasoning"]}
                for dim, data in judge["scores"].items()
            ])

            if st.session_state.document_regen_count < 1:
                if st.button("Regenerate Report", key="document_regenerate_report"):
                    source_material = "\n\n---\n\n".join(
                        f"Question: {e['question']}\n{e['answer']}" for e in st.session_state.document_material
                    )
                    weakest_dim = min(
                        st.session_state.document_judge_result["scores"],
                        key=lambda d: st.session_state.document_judge_result["scores"][d]["score"],
                    )
                    weakest_feedback = st.session_state.document_judge_result["scores"][weakest_dim]["reasoning"]
                    with st.spinner("Regenerating your report…"):
                        new_synthesis = synthesis_node({
                            "topic": "Document findings",
                            "source_material": source_material,
                            "source_list": st.session_state.document_sources,
                            "previous_report": st.session_state.document_report["brief"],
                            "weakest_dimension": weakest_dim,
                            "weakest_dimension_feedback": weakest_feedback,
                        })
                        new_judge = judge_node({
                            "topic": "Document findings",
                            "source_material": source_material,
                            "report": new_synthesis["brief"],
                        })
                    st.session_state.document_report_regen = new_synthesis
                    st.session_state.document_judge_regen = new_judge
                    st.session_state.document_regen_count += 1
                    st.rerun()
            else:
                st.caption("Maximum regenerations reached.")

            if st.session_state.document_report_regen is not None:
                st.divider()
                st.markdown("#### 📈 Regenerated Report")
                weakest_dim = min(
                    st.session_state.document_judge_result["scores"],
                    key=lambda d: st.session_state.document_judge_result["scores"][d]["score"],
                )
                original_score = st.session_state.document_judge_result["overall_score"]
                regen_score = st.session_state.document_judge_regen["overall_score"]
                st.caption(f"Targeted **{weakest_dim}** dimension — overall score {original_score}/5.0 → {regen_score}/5.0")
                st.markdown(st.session_state.document_report_regen["brief"])
                if st.session_state.document_report_regen.get("sources_section"):
                    st.markdown(st.session_state.document_report_regen["sources_section"])
                st.divider()
                st.markdown(f"#### Updated Quality Score: {st.session_state.document_judge_regen['overall_score']}/5.0")
                st.caption(st.session_state.document_judge_regen["score_explanation"])
                st.table([
                    {"Dimension": dim.capitalize(), "Score": f"{data['score']}/5", "Reasoning": data["reasoning"]}
                    for dim, data in st.session_state.document_judge_regen["scores"].items()
                ])

            version_options = (
                ["Original Report", "Regenerated Report"]
                if st.session_state.document_report_regen is not None
                else ["Original Report"]
            )
            st.divider()
            st.markdown("#### Export Report")
            export_version = st.radio("Version", options=version_options, horizontal=True, key="document_export_version")
            export_format = st.selectbox("Format", options=["DOCX", "PPTX"], key="document_export_format_select")
            cache_key = f"{export_format}-{export_version}"
            if st.session_state.document_export_format != cache_key:
                st.session_state.document_export_bytes = None
                st.session_state.document_export_format = None
            if st.button("Download Report", key="document_download_btn"):
                if "Regenerated" in export_version:
                    brief = st.session_state.document_report_regen["brief"]
                    sources_section = st.session_state.document_report_regen.get("sources_section", "")
                else:
                    brief = st.session_state.document_report["brief"]
                    sources_section = st.session_state.document_report.get("sources_section", "")
                with st.spinner("Preparing export…"):
                    if export_format == "DOCX":
                        path = export_to_docx(topic="Document findings", brief=brief, sources_section=sources_section)
                    else:
                        path = export_to_pptx(topic="Document findings", brief=brief, sources_section=sources_section)
                if path is not None:
                    with open(path, "rb") as f:
                        st.session_state.document_export_bytes = f.read()
                    st.session_state.document_export_format = cache_key
                    st.rerun()
                else:
                    st.error("Export failed — please try again.")
            if st.session_state.document_export_bytes is not None:
                mime = (
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    if export_format == "DOCX"
                    else "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )
                st.download_button(
                    label=f"Click to download {export_format}",
                    data=st.session_state.document_export_bytes,
                    file_name=f"groundwork_{st.session_state.document_material[0]['question'].lower().replace(' ', '_')[:40]}.{export_format.lower()}",
                    mime=mime,
                    key=f"document_download_file_{export_format}",
                )

        st.divider()

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )
    st.caption("Supported file types: PDF and Word (.docx) only.")

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

            file_suffix = os.path.splitext(file.name)[1].lower()

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
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

            col1, col2 = st.columns([3, 1])

            with col1:
                if st.button("Ask another question", key="document_ask_another"):
                    if st.session_state.document_result is not None:
                        st.session_state.document_material.append({
                            "mode": "document",
                            "question": st.session_state.document_question,
                            "answer": st.session_state.document_result["answer"],
                            "sources": st.session_state.document_result.get("sources", []),
                        })
                        existing_urls = {s.get("url") or s.get("link") for s in st.session_state.document_sources}
                        for source in st.session_state.document_result.get("sources", []):
                            key = source.get("url") or source.get("link")
                            if key not in existing_urls:
                                st.session_state.document_sources.append(source)
                                existing_urls.add(key)
                    st.session_state.document_result = None
                    st.session_state.document_question = None
                    st.rerun()

            with col2:
                if not st.session_state.document_report_flag:
                    if st.button("Generate Full Report", key="document_generate_from_result"):
                        if st.session_state.document_result is not None:
                            st.session_state.document_material.append({
                                "mode": "document",
                                "question": st.session_state.document_question,
                                "answer": st.session_state.document_result["answer"],
                                "sources": st.session_state.document_result.get("sources", []),
                            })
                            existing_urls = {s.get("url") or s.get("link") for s in st.session_state.document_sources}
                            for source in st.session_state.document_result.get("sources", []):
                                key = source.get("url") or source.get("link")
                                if key not in existing_urls:
                                    st.session_state.document_sources.append(source)
                                    existing_urls.add(key)
                        st.session_state.document_result = None
                        st.session_state.document_question = None
                        st.session_state.document_report_flag = True
                        st.rerun()

    else:

        st.markdown("## Upload a document to get started")
        st.markdown(
            "Upload one or more PDF files above and Groundwork will index them "
            "so you can ask questions grounded in their content."
        )
