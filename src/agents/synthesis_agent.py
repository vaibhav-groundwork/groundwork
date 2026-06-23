"""
synthesis_agent.py — LangGraph node that transforms research material into a
polished, structured brief.

Pipeline position:
  graph.py routes to synthesis_node as the final step in both the web-research
  and document-RAG pipelines, after all information gathering is complete.

Mode-agnostic and turn-agnostic design:
  source_material is a normalised string already assembled by app.py/graph.py.
  It may contain research notes from research_agent, an answer from rag_agent,
  or several of either concatenated across multiple user questions in one
  session. This node does not know or care which agent(s) produced it, how
  many turns it spans, or whether it came from web search or a document — it
  receives a string and produces a brief. That boundary is intentional: keeping
  the synthesis step decoupled from the collection mechanism makes both halves
  easier to change independently.

Always-structured design:
  This node always produces a full structured report (title, executive summary,
  themed sections, conclusion). The decision of WHETHER a user wants a structured
  report versus a quick answer is made earlier, by the user themselves, via the
  "Generate Full Report" CTA in app.py — by the time this node runs, that decision
  has already been made. This avoids the inherent unreliability of having an LLM
  infer structural intent from content alone.

Sources separation:
  Inline citations are explicitly prohibited in the brief body. Sources are
  assembled independently as a formatted string (sources_section) from the
  structured source_list already extracted upstream — no LLM call needed.
  Keeping sources separate prevents Claude from hallucinating citation markers
  and gives the UI clean control over how sources are rendered.

Truncation safeguard:
  source_material is capped at _SOURCE_MATERIAL_CHAR_LIMIT before being sent
  to Claude, mirroring the same safeguard applied to retrieved chunks in
  rag_agent.py. Multi-turn accumulation could otherwise cause unbounded token
  growth.

Exports:
  SynthesisState  — TypedDict shared with graph.py
  synthesis_node  — single LangGraph node for brief generation
"""

import logging
from typing import TypedDict

from src.config import MAX_TOKENS_SONNET, SONNET_MODEL
from src.tracing import call_llm

logger = logging.getLogger(__name__)

# Character ceiling applied to source_material before sending to Claude.
# Keeps token cost predictable regardless of how much multi-turn material
# has accumulated upstream. Mirrors _CONTEXT_CHAR_LIMIT in rag_agent.py.
_SOURCE_MATERIAL_CHAR_LIMIT = 12000


# ── State schema ──────────────────────────────────────────────────────────────

class SynthesisState(TypedDict):
    topic: str
    source_material: str        # normalised input: research notes, RAG answer,
                                # or multi-turn accumulation — assembled upstream
    source_list: list[dict]     # structured sources extracted upstream, each
                                # containing filename/URL info for citation display
    brief: str
    sources_section: str
    status_message: str


# ── Node: synthesis_node ──────────────────────────────────────────────────────

def synthesis_node(state: SynthesisState) -> dict:
    """
    LangGraph node — produces a polished, structured brief from pre-assembled
    source material using Claude Sonnet.

    See module docstring for full design rationale. Key behaviours:

    - source_material is truncated to _SOURCE_MATERIAL_CHAR_LIMIT if needed,
      with a WARNING logged so unexpected truncation is visible in dev.

    - Claude chooses structure based on scope: narrow material → concise prose;
      broad/multi-theme material → titled sections with executive summary.

    - Inline citations are explicitly prohibited in the brief body; sources are
      formatted separately from source_list as a plain string, with no LLM call.

    - call_llm() is wrapped in try/except; on failure a friendly error message
      is returned rather than propagating the exception into the pipeline.

    - cache_system_prompt=False: synthesis runs once per user-triggered request,
      so there is no repeated system prompt within a run to cache.

    Returns a partial-state dict; LangGraph merges it into the full state.
    """
    topic: str = state["topic"]
    source_material: str = state.get("source_material", "")
    source_list: list[dict] = state.get("source_list", [])

    # ── Truncation safeguard ──────────────────────────────────────────────────
    original_length = len(source_material)
    if original_length > _SOURCE_MATERIAL_CHAR_LIMIT:
        source_material = source_material[:_SOURCE_MATERIAL_CHAR_LIMIT]
        logger.warning(
            "synthesis_node: source_material truncated from %d to %d characters "
            "before sending to Claude.",
            original_length,
            _SOURCE_MATERIAL_CHAR_LIMIT,
        )

    logger.info(
        "synthesis_node: topic=%r, source_material length=%d characters.",
        topic,
        len(source_material),
    )

    # ── System prompt ─────────────────────────────────────────────────────────
    system_prompt = (
        "You are a professional brief writer preparing a structured report for a business "
        "audience. You will be given a topic and source material — research notes or an "
        "extracted answer, possibly accumulated across multiple questions — and must produce "
        "a well-written, fully structured report grounded strictly in that material. Do not "
        "add outside information or make inferences beyond what the source material explicitly "
        "supports.\n\n"
        "Always produce the following structure:\n"
        "- A clear, descriptive title\n"
        "- An Executive Summary (2-3 sentences capturing the key takeaway)\n"
        "- 2-4 themed body sections, each with a descriptive subheading, organizing the source "
        "material logically by topic\n"
        "- A brief Conclusion\n\n"
        "If the source material is limited, write shorter sections rather than padding with "
        "repetition or invented detail — the structure should still be present, but content "
        "should remain accurate and proportionate to what the source material actually supports.\n\n"
        "Do NOT include inline citations, footnote markers, or source references of any kind "
        "within the body text. Sources will be presented in a separate section by the application. "
        "Write in clear, professional prose appropriate for a business audience."
)

    user_message = (
        f"Topic: {topic}\n\n"
        f"Source material:\n\n{source_material}"
    )

    messages = [{"role": "user", "content": user_message}]

    # ── LLM call (with graceful error handling) ───────────────────────────────
    try:
        response = call_llm(
            model=SONNET_MODEL,
            messages=messages,
            system=system_prompt,
            max_tokens=MAX_TOKENS_SONNET,
            cache_system_prompt=False,
        )
    except Exception as exc:
        logger.warning(
            "synthesis_node: call_llm failed — %s: %s",
            type(exc).__name__,
            exc,
        )
        return {
            "brief": (
                "Something went wrong generating your brief — please try again."
            ),
            "sources_section": "",
            "status_message": "⚠️ Error generating brief — please try again.",
        }

    brief: str = response.choices[0].message.content.strip()
    logger.info("synthesis_node: brief generated, length=%d characters.", len(brief))

    # ── Sources section (pure string formatting — no LLM call) ────────────────
    sources_section = _format_sources(source_list)

    return {
        "brief": brief,
        "sources_section": sources_section,
        "status_message": "✅ Brief ready.",
    }


# ── Private helper ────────────────────────────────────────────────────────────

def _format_sources(source_list: list[dict]) -> str:
    """
    Formats source_list into a numbered plain-text block for UI display.

    Each dict may contain 'source' (filename) or 'link' (URL) or both.
    Returns an empty string if source_list is empty — no placeholder text.
    """
    if not source_list:
        return ""

    lines = []
    for i, s in enumerate(source_list, 1):
        filename = s.get("source", "")
        url = s.get("link", "")

        if filename and url:
            lines.append(f"{i}. {filename} — {url}")
        elif url:
            lines.append(f"{i}. {url}")
        elif filename:
            lines.append(f"{i}. {filename}")

    return "\n".join(lines)
