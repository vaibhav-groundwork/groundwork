"""
pptx_export.py — generates a professional PowerPoint deck from a synthesised
report using a two-stage architecture: one Claude call for narrative planning,
followed by fully deterministic Python-driven Node.js rendering.

Two-stage architecture rationale:
  Stage 1 — Claude plans the deck: given the brief, Claude decides how many
  slides to produce, which layout to use for each, what text goes on each slide,
  and which colour palette best fits the topic. The result is a structured JSON
  plan returned via OpenAI-style function calling (same Level-3 approach as
  judge_agent.py), guaranteeing a schema-conformant object.

  Stage 2 — Python renders the deck deterministically: _render_pptx_script()
  converts the validated plan into a self-contained Node.js script using the
  pptxgenjs npm package, then executes it via subprocess. Once the plan exists,
  no further LLM calls are made. This separation means layout bugs are fixed in
  Python code, not by re-prompting Claude.

Slide-count sanity check:
  After parsing Claude's tool response, the slides list is validated to contain
  between 3 and 20 items. Fewer than 3 suggests a degenerate response (title +
  nothing). More than 20 would produce an unusably long deck and likely indicates
  a malformed response. Both bounds are generous; the typical target is 8-10.

Sources as a dedicated final slide:
  Sources are not embedded in the narrative slides — they're rendered as a
  standalone final slide from the structured source_list already assembled by
  app.py. This mirrors the docx exporter's separate sources_section design:
  keeping citations out of the narrative gives Claude one less concern during
  slide planning and avoids hallucinated citation markers in body text.

Anthropic pptx skill guidance as code:
  Layout variety, bullet-length capping, speaker notes on every slide, and
  em-dash removal are enforced through enum constraints in the tool schema,
  deterministic post-processing in _enforce_layout_variety(), and strip_em_dashes()
  applied to every text field, rather than relying on a visual-QA feedback loop
  at runtime.

v1 scope boundary:
  Rendered charts and data visualisations are out of scope. All quantitative
  content is expressed as text or stat_callout layout (large-number callout).
  Chart support is tracked in docs/groundwork_v2_backlog.md.

Exports:
  export_to_pptx — single public entry point for app.py
"""

import json
import logging
import os
import subprocess

from src.config import MAX_TOKENS_SONNET, SONNET_MODEL, get_output_dir
from src.tracing import call_llm
from src.utils import _escape_for_js_string, strip_em_dashes

logger = logging.getLogger(__name__)

# Character ceiling applied to brief before sending to Claude in Stage 1.
# Matches synthesis_agent.py's same safeguard — multi-turn accumulation could
# otherwise produce unbounded token growth in Stage 1.
_SOURCE_MATERIAL_CHAR_LIMIT = 12000


# ── Colour palettes ───────────────────────────────────────────────────────────

_PALETTES: dict[str, dict[str, str]] = {
    "midnight_executive": {"primary": "1E2761", "secondary": "CADCFC", "accent": "FFFFFF"},
    "forest_moss":        {"primary": "2C5F2D", "secondary": "97BC62", "accent": "F5F5F5"},
    "coral_energy":       {"primary": "F96167", "secondary": "F9E795", "accent": "2F3C7E"},
    "warm_terracotta":    {"primary": "B85042", "secondary": "E7E8D1", "accent": "A7BEAE"},
    "ocean_gradient":     {"primary": "065A82", "secondary": "1C7293", "accent": "21295C"},
    "charcoal_minimal":   {"primary": "36454F", "secondary": "F2F2F2", "accent": "212121"},
    "teal_trust":         {"primary": "028090", "secondary": "00A896", "accent": "02C39A"},
    "berry_cream":        {"primary": "6D2E46", "secondary": "A26769", "accent": "ECE2D0"},
    "sage_calm":          {"primary": "84B59F", "secondary": "69A297", "accent": "50808E"},
    "cherry_bold":        {"primary": "990011", "secondary": "FCF6F5", "accent": "2F3C7E"},
}

_PALETTE_KEYS = list(_PALETTES.keys())


# ── Tool schema ───────────────────────────────────────────────────────────────

_SLIDE_PLAN_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "create_slide_plan",
        "description": (
            "Produce a complete PowerPoint slide plan for the given topic and brief, "
            "ready to render without further edits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "deck_title": {
                    "type": "string",
                    "description": "Short, punchy title for the deck (used on the title slide).",
                },
                "color_palette": {
                    "type": "string",
                    "enum": _PALETTE_KEYS,
                    "description": "Colour palette that best fits the topic's mood and industry.",
                },
                "slides": {
                    "type": "array",
                    "description": (
                        "8-10 narrative slides. Must include one title, one agenda, "
                        "several content slides with varied layouts, and one conclusion."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "layout_type": {
                                "type": "string",
                                "enum": [
                                    "title",
                                    "agenda",
                                    "stat_callout",
                                    "two_column",
                                    "icon_rows",
                                    "standard",
                                    "conclusion",
                                ],
                            },
                            "title": {
                                "type": "string",
                                "description": "Slide title (shown as the slide heading).",
                            },
                            "content": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Bullet text for the slide. Max 5 items, roughly 10-12 words "
                                    "each. Exception: stat_callout uses exactly 2 strings — "
                                    "the statistic/number first, then its explanatory label."
                                ),
                            },
                            "speaker_notes": {
                                "type": "string",
                                "description": (
                                    "Natural spoken-language notes for the presenter. Required on "
                                    "every slide including title and conclusion."
                                ),
                            },
                        },
                        "required": ["layout_type", "title", "content", "speaker_notes"],
                    },
                },
                "sources_slide": {
                    "type": "object",
                    "description": (
                        "Content for the final Sources slide, rendered separately after all "
                        "narrative slides."
                    ),
                    "properties": {
                        "content": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Formatted source strings (filename or URL per item).",
                        },
                    },
                    "required": ["content"],
                },
            },
            "required": ["deck_title", "color_palette", "slides", "sources_slide"],
        },
    },
}


# ── Stage 1: slide planning ───────────────────────────────────────────────────

def _build_slide_plan(topic: str, brief: str, sources_section: str) -> dict | None:
    """
    Calls Claude to produce a structured slide plan via function calling.
    Returns the validated, em-dash-cleaned plan dict, or None on any failure.
    """
    original_length = len(brief)
    if original_length > _SOURCE_MATERIAL_CHAR_LIMIT:
        brief = brief[:_SOURCE_MATERIAL_CHAR_LIMIT]
        logger.warning(
            "_build_slide_plan: brief truncated from %d to %d characters before "
            "sending to Claude.",
            original_length,
            _SOURCE_MATERIAL_CHAR_LIMIT,
        )

    system_prompt = (
        "You are a professional presentation designer producing a polished business deck "
        "that is ready to present with no further edits. Use the create_slide_plan tool "
        "to return your complete plan.\n\n"
        "Requirements:\n"
        "- Produce 8-10 narrative slides: one title slide, one agenda slide, several "
        "varied content slides using different layout types, and one conclusion slide.\n"
        "- Never repeat the same layout_type more than 2 slides in a row.\n"
        "- Every slide must contain specific, real content drawn from the brief — no "
        "generic, placeholder, or invented text.\n"
        "- Keep bullet text concise: roughly 10-12 words per item, maximum 5 items per "
        "slide. stat_callout slides must have exactly 2 content strings: the statistic "
        "or number first, then its explanatory label.\n"
        "- Use stat_callout sparingly: maximum 1-2 slides in the entire deck, reserved "
        "only for a single headline number genuinely dramatic enough to warrant an "
        "entire slide on its own. Any other statistic should be woven naturally into "
        "the bullet content of a standard, agenda, or two_column slide alongside "
        "related points, not given its own dedicated slide.\n"
        "- Write speaker_notes for every slide including the title and conclusion slide, "
        "in natural spoken language as if rehearsing aloud.\n"
        "- Do not use em dashes anywhere in your output.\n"
        "- Choose the color_palette that best fits the topic's tone and industry context."
    )

    user_message = (
        f"Topic: {topic}\n\n"
        f"Brief:\n{brief}\n\n"
        f"Sources:\n{sources_section}"
    )

    try:
        response = call_llm(
            model=SONNET_MODEL,
            messages=[{"role": "user", "content": user_message}],
            system=system_prompt,
            max_tokens=MAX_TOKENS_SONNET,
            cache_system_prompt=False,
            tools=[_SLIDE_PLAN_TOOL],
            tool_choice={"type": "function", "function": {"name": "create_slide_plan"}},
        )

        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise ValueError(
                "Claude returned a response with no tool_calls — "
                "expected create_slide_plan to be called."
            )

        args: dict = json.loads(tool_calls[0].function.arguments)

    except Exception as exc:
        logger.warning(
            "_build_slide_plan: failed — %s: %s",
            type(exc).__name__,
            exc,
        )
        return None

    # Validate slide count
    slides = args.get("slides", [])
    if not isinstance(slides, list) or not (3 <= len(slides) <= 20):
        logger.warning(
            "_build_slide_plan: invalid slide count %d (expected 3-20) — "
            "returning None.",
            len(slides) if isinstance(slides, list) else -1,
        )
        return None

    # Apply em-dash removal to all text fields deterministically
    cleaned_slides = []
    for slide in slides:
        cleaned_slides.append({
            "layout_type": slide.get("layout_type", "standard"),
            "title": strip_em_dashes(slide.get("title", "")),
            "content": [strip_em_dashes(item) for item in slide.get("content", [])],
            "speaker_notes": strip_em_dashes(slide.get("speaker_notes", "")),
        })

    sources_slide = args.get("sources_slide", {"content": []})
    cleaned_sources = {
        "content": [strip_em_dashes(s) for s in sources_slide.get("content", [])],
    }

    return {
        "deck_title": strip_em_dashes(args.get("deck_title", topic)),
        "color_palette": args.get("color_palette", "midnight_executive"),
        "slides": cleaned_slides,
        "sources_slide": cleaned_sources,
    }


# ── Layout variety enforcement ────────────────────────────────────────────────

def _enforce_layout_variety(slides: list[dict]) -> list[dict]:
    """
    Ensures no layout_type appears 3+ times consecutively by replacing slides
    after the second in any such run, alternating between 'standard' and
    'icon_rows' as the replacement type.

    Uses a single forward pass — checked against the (possibly already-modified)
    previous two slides so that each replacement naturally breaks the streak
    for subsequent slides.
    """
    if len(slides) < 3:
        return slides

    result = [dict(s) for s in slides]
    replacements = ["standard", "icon_rows"]
    repl_idx = 0

    for i in range(2, len(result)):
        if (
            result[i]["layout_type"]
            == result[i - 1]["layout_type"]
            == result[i - 2]["layout_type"]
        ):
            result[i]["layout_type"] = replacements[repl_idx % 2]
            repl_idx += 1

    return result


# ── Stage 2: deterministic pptxgenjs script generation ───────────────────────

def _render_pptx_script(
    deck_title: str,
    color_palette: str,
    slides: list[dict],
    sources_slide: dict,
    output_path: str,
) -> str:
    """
    Generates a complete, self-contained Node.js pptxgenjs script.

    Slide data is embedded as a raw JSON literal (valid JS object syntax),
    which avoids double-escaping — json.dumps handles all string escaping
    within the data. _escape_for_js_string is applied only to strings embedded
    directly in JS template literals (output path, deck title).

    Uses <<PLACEHOLDER>> substitution rather than f-strings to avoid escaping
    the many curly braces in the JS object literals.
    """
    pal = _PALETTES.get(color_palette, _PALETTES["midnight_executive"])

    slides_json = json.dumps(slides, ensure_ascii=False)
    sources_json = json.dumps(sources_slide.get("content", []), ensure_ascii=False)
    safe_out = _escape_for_js_string(os.path.abspath(output_path))

    # Note: text content inside slide JSON is already safely encoded by
    # json.dumps; only template-literal-embedded strings need _escape_for_js_string.
    template = (
        "'use strict';\n"
        "const pptxgen = require('pptxgenjs');\n"
        "\n"
        "const prs = new pptxgen();\n"
        "prs.layout = 'LAYOUT_WIDE';  // 13.3 × 7.5 inches\n"
        "\n"
        "const PAL = {\n"
        "  primary:   '<<PRI>>',\n"
        "  secondary: '<<SEC>>',\n"
        "  accent:    '<<ACC>>'\n"
        "};\n"
        "\n"
        "const SLIDES  = <<SLIDES_JSON>>;\n"
        "const SOURCES = <<SOURCES_JSON>>;\n"
        "\n"
        "// ── shared layout constants ──────────────────────────────────────────\n"
        "const TX   = 0.5;    // title / body x\n"
        "const TY   = 0.35;   // slide-title y\n"
        "const TW   = 12.3;   // usable width\n"
        "const TH   = 1.0;    // slide-title height\n"
        "const BY   = 1.55;   // body start y\n"
        "const IH   = 0.65;   // per-item row height\n"
        "const DARK = 'F5F5F5';\n"
        "const BODY = '333333';\n"
        "\n"
        "function addSlideTitle(slide, text) {\n"
        "  slide.addText(text, {\n"
        "    x: TX, y: TY, w: TW, h: TH,\n"
        "    fontSize: 24, bold: true, color: PAL.primary,\n"
        "    fontFace: 'Arial', valign: 'middle'\n"
        "  });\n"
        "}\n"
        "\n"
        "// ── layout renderers ─────────────────────────────────────────────────\n"
        "\n"
        "function renderTitle(slide, data) {\n"
        "  slide.background = { color: PAL.primary };\n"
        "  slide.addText(data.title, {\n"
        "    x: TX, y: 1.8, w: TW, h: 2.4,\n"
        "    fontSize: 44, bold: true, color: 'FFFFFF',\n"
        "    fontFace: 'Arial', align: 'center', valign: 'middle'\n"
        "  });\n"
        "  if (data.content && data.content.length) {\n"
        "    slide.addText(data.content.join('  \u00b7  '), {\n"
        "      x: TX, y: 4.4, w: TW, h: 0.9,\n"
        "      fontSize: 20, color: PAL.secondary,\n"
        "      fontFace: 'Calibri', align: 'center', valign: 'middle'\n"
        "    });\n"
        "  }\n"
        "}\n"
        "\n"
        "function renderConclusion(slide, data) {\n"
        "  slide.background = { color: PAL.primary };\n"
        "  slide.addText(data.title, {\n"
        "    x: TX, y: 1.8, w: TW, h: 2.2,\n"
        "    fontSize: 40, bold: true, color: 'FFFFFF',\n"
        "    fontFace: 'Arial', align: 'center', valign: 'middle'\n"
        "  });\n"
        "  if (data.content && data.content.length) {\n"
        "    const rows = data.content.map(t => ({ text: t + '\\n' }));\n"
        "    slide.addText(rows, {\n"
        "      x: TX, y: 4.2, w: TW, h: 2.7,\n"
        "      fontSize: 18, color: PAL.secondary,\n"
        "      fontFace: 'Calibri', align: 'center', valign: 'top'\n"
        "    });\n"
        "  }\n"
        "}\n"
        "\n"
        "function renderAgenda(slide, data) {\n"
        "  slide.background = { color: DARK };\n"
        "  addSlideTitle(slide, data.title);\n"
        "  data.content.forEach((item, i) => {\n"
        "    slide.addText((i + 1) + '.  ' + item, {\n"
        "      x: TX, y: BY + i * IH, w: TW, h: IH,\n"
        "      fontSize: 18, color: BODY,\n"
        "      fontFace: 'Calibri', valign: 'middle'\n"
        "    });\n"
        "  });\n"
        "}\n"
        "\n"
        "function renderStandard(slide, data) {\n"
        "  slide.background = { color: DARK };\n"
        "  addSlideTitle(slide, data.title);\n"
        "  data.content.forEach((item, i) => {\n"
        "    slide.addText(item, {\n"
        "      x: TX, y: BY + i * IH, w: TW, h: IH, fontSize: 16, fontFace: 'Calibri',\n"
        "      color: BODY, valign: 'middle',\n"
        "      bullet: { code: '25A0', indent: 18 }\n"
        "    });\n"
        "  });\n"
        "}\n"
        "function renderIconRows(slide, data) {\n"
        "  slide.background = { color: DARK };\n"
        "  addSlideTitle(slide, data.title);\n"
        "  data.content.forEach((item, i) => {\n"
        "    slide.addText(item, {\n"
        "      x: TX, y: BY + i * (IH + 0.05), w: TW, h: IH, fontSize: 16, fontFace: 'Calibri',\n"
        "      color: BODY, valign: 'middle',\n"
        "      bullet: { code: '25CF', indent: 18 }\n"
        "    });\n"
        "  });\n"
        "}\n"
        "function renderStatCallout(slide, data) {\n"
        "  slide.background = { color: PAL.primary };\n"
        "  const stat  = data.content[0] || '';\n"
        "  const label = data.content[1] || '';\n"
        "  slide.addText(stat, {\n"
        "    x: TX, y: 1.0, w: TW, h: 3.5,\n"
        "    fontSize: 96, bold: true, color: 'FFFFFF',\n"
        "    fontFace: 'Arial', align: 'center', valign: 'middle'\n"
        "  });\n"
        "  slide.addText(label, {\n"
        "    x: TX, y: 4.7, w: TW, h: 1.4,\n"
        "    fontSize: 24, color: PAL.secondary,\n"
        "    fontFace: 'Calibri', align: 'center', valign: 'middle'\n"
        "  });\n"
        "  slide.addText(data.title, {\n"
        "    x: TX, y: TY, w: TW, h: TH,\n"
        "    fontSize: 20, color: PAL.secondary,\n"
        "    fontFace: 'Arial', valign: 'middle'\n"
        "  });\n"
        "}\n"
        "\n"
        "function renderTwoColumn(slide, data) {\n"
        "  slide.background = { color: DARK };\n"
        "  addSlideTitle(slide, data.title);\n"
        "  const half  = Math.ceil(data.content.length / 2);\n"
        "  const left  = data.content.slice(0, half);\n"
        "  const right = data.content.slice(half);\n"
        "  const cw    = 5.85;\n"
        "  const gap   = 0.6;\n"
        "  [left, right].forEach((col, ci) => {\n"
        "    const cx = TX + ci * (cw + gap);\n"
        "    col.forEach((item, i) => {\n"
        "      slide.addText(item, {\n"
        "        x: cx, y: BY + i * IH, w: cw, h: IH, fontSize: 15, fontFace: 'Calibri',\n"
        "        color: BODY, valign: 'middle',\n"
        "        bullet: { code: '2022', indent: 14 }\n"
        "      });\n"
        "    });\n"
        "  });\n"
        "}\n"
        "\n"
        "// ── main render loop ─────────────────────────────────────────────────\n"
        "\n"
        "SLIDES.forEach(data => {\n"
        "  const slide = prs.addSlide();\n"
        "  switch (data.layout_type) {\n"
        "    case 'title':        renderTitle(slide, data);       break;\n"
        "    case 'conclusion':   renderConclusion(slide, data);  break;\n"
        "    case 'agenda':       renderAgenda(slide, data);      break;\n"
        "    case 'stat_callout': renderStatCallout(slide, data); break;\n"
        "    case 'two_column':   renderTwoColumn(slide, data);   break;\n"
        "    case 'icon_rows':    renderIconRows(slide, data);    break;\n"
        "    default:             renderStandard(slide, data);    break;\n"
        "  }\n"
        "  if (data.speaker_notes) slide.addNotes(data.speaker_notes);\n"
        "});\n"
        "\n"
        "// ── sources slide ────────────────────────────────────────────────────\n"
        "\n"
        "const srcSlide = prs.addSlide();\n"
        "srcSlide.background = { color: DARK };\n"
        "srcSlide.addText('Sources', {\n"
        "  x: TX, y: TY, w: TW, h: TH,\n"
        "  fontSize: 24, bold: true, color: PAL.primary,\n"
        "  fontFace: 'Arial', valign: 'middle'\n"
        "});\n"
        "SOURCES.forEach((src, i) => {\n"
        "  srcSlide.addText(src, {\n"
        "    x: TX, y: BY + i * 0.48, w: TW, h: 0.44,\n"
        "    fontSize: 13, color: BODY,\n"
        "    fontFace: 'Calibri', valign: 'middle'\n"
        "  });\n"
        "});\n"
        "\n"
        "// ── write file ───────────────────────────────────────────────────────\n"
        "\n"
        "prs.writeFile({ fileName: `<<OUT>>` })\n"
        "  .then(() => process.exit(0))\n"
        "  .catch(err => { process.stderr.write(String(err)); process.exit(1); });\n"
    )

    script = (
        template
        .replace("<<PRI>>", pal["primary"])
        .replace("<<SEC>>", pal["secondary"])
        .replace("<<ACC>>", pal["accent"])
        .replace("<<SLIDES_JSON>>", slides_json)
        .replace("<<SOURCES_JSON>>", sources_json)
        .replace("<<OUT>>", safe_out)
    )
    return script


# ── Public export function ────────────────────────────────────────────────────

def export_to_pptx(topic: str, brief: str, sources_section: str) -> str | None:
    """
    Generates a PowerPoint deck from the synthesised brief and returns its
    absolute path on success, or None on any failure.

    Steps:
      1. Validates that brief is non-empty.
      2. Calls Claude (Stage 1) to produce a validated slide plan.
      3. Enforces layout variety with _enforce_layout_variety().
      4. Builds the output directory.
      5. Generates the Node.js pptxgenjs script (Stage 2).
      6. Writes and executes the script via subprocess (30s timeout).
      7. Checks return code explicitly (non-zero is a failure, not an exception).
      8. Verifies the output file exists and has size > 0.
      9. Cleans up the temporary script file (non-fatal on failure).
      10. Returns the absolute path to the .pptx file.

    Returns None (never raises) on any failure — app.py checks for this and
    shows a friendly error message rather than propagating a stack trace to
    the UI.
    """
    if not brief or not brief.strip():
        logger.warning(
            "export_to_pptx: brief is empty — no content to export, returning None."
        )
        return None

    # Stage 1: ask Claude to plan the deck
    plan = _build_slide_plan(topic, brief, sources_section)
    if plan is None:
        return None

    slides = _enforce_layout_variety(plan["slides"])

    # Build output directory
    output_dir = get_output_dir(topic)
    os.makedirs(output_dir, exist_ok=True)

    pptx_path = os.path.join(output_dir, "report.pptx")
    script_path = os.path.join(output_dir, "_gen_pptx.js")

    # Stage 2: generate, run, and verify
    script = _render_pptx_script(
        deck_title=plan["deck_title"],
        color_palette=plan["color_palette"],
        slides=slides,
        sources_slide=plan["sources_slide"],
        output_path=pptx_path,
    )

    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        result = subprocess.run(
            ["node", script_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.warning(
                "export_to_pptx: Node.js script exited with code %d. stderr: %s",
                result.returncode,
                result.stderr.strip(),
            )
            return None

        if not os.path.exists(pptx_path) or os.path.getsize(pptx_path) == 0:
            logger.warning(
                "export_to_pptx: subprocess reported success but output file "
                "is missing or empty at %s.",
                pptx_path,
            )
            return None

        # Cleanup temp script — only the .pptx is the real deliverable
        try:
            os.remove(script_path)
        except OSError as cleanup_err:
            logger.warning(
                "export_to_pptx: failed to remove temporary script %s — %s",
                script_path,
                cleanup_err,
            )

        abs_path = os.path.abspath(pptx_path)
        logger.info("export_to_pptx: deck written to %s.", abs_path)
        return abs_path

    except Exception as exc:
        logger.warning(
            "export_to_pptx: export failed — %s: %s",
            type(exc).__name__,
            exc,
        )
        return None
