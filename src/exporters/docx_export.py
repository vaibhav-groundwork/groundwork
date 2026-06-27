"""
docx_export.py — generates a professional Word (.docx) document from a
synthesised report, using a generated Node.js script and the docx npm package.

Role in the pipeline:
  app.py calls export_to_docx() after synthesis_node produces the brief.
  This file is responsible only for writing the file to a server-side temp
  location and returning its absolute path. The user-facing download filename
  (e.g. "AI_trends_brief.docx") is app.py's responsibility — it is not set
  here, keeping file naming and file generation as separate concerns.

Why Node.js / docx-js code generation:
  python-docx produces valid .docx files but its formatting API is verbose and
  requires careful style management for bold headings, font sizes, and section
  spacing. The docx npm package provides a clean, declarative API for the same
  output with less boilerplate. This file generates a self-contained Node.js
  script at runtime and executes it via subprocess — the same pattern used for
  PPTX generation elsewhere in this codebase.

Why markdown parsing is necessary:
  Claude's brief output contains markdown formatting (# Title, ## Headings,
  body paragraphs). The Word document must map these to proper Word paragraph
  styles rather than printing the raw # characters as literal text.
  _parse_markdown_structure() converts the flat string into a typed element
  list that the script generator can iterate over.

Why the multi-layer validation exists:
  Three separate checks are applied, not one:
    1. Empty-content check (before any file I/O): prevents generating an empty
       document and wasting a subprocess call.
    2. subprocess return code check: a non-zero exit code means the Node.js
       script itself failed, but subprocess.run() does not raise an exception
       for non-zero exit codes — it must be checked explicitly.
    3. Output file existence + non-zero size check: subprocess can exit 0 while
       still writing an empty or missing file if the docx library encountered
       an error it caught internally. Any one of these checks alone would miss
       a different class of real failure.

Failure behaviour:
  export_to_docx() returns None on any failure rather than raising an
  exception, so app.py can check for that and show a friendly error message
  without a stack trace reaching the UI.

Exports:
  export_to_docx            — main entry point for app.py
  _parse_markdown_structure — exported for testing
  _escape_for_js_string     — exported for testing
"""

import logging
import os
import subprocess

from src.config import get_output_dir

logger = logging.getLogger(__name__)


# ── Markdown parser ───────────────────────────────────────────────────────────

def _parse_markdown_structure(brief: str) -> list[dict]:
    """
    Converts a markdown-formatted brief into a flat list of typed elements.

    Element types:
      'title'     — lines starting with a single #  (stripped of # prefix)
      'heading'   — lines starting with ##           (stripped of ## prefix)
      'paragraph' — all other non-empty lines

    Empty lines are skipped entirely. The order of type-checking matters:
    ## must be tested before # to avoid ## lines matching the single-# branch.
    """
    elements = []
    for line in brief.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            elements.append({"type": "heading", "text": stripped[3:].strip()})
        elif stripped.startswith("##"):
            elements.append({"type": "heading", "text": stripped[2:].strip()})
        elif stripped.startswith("# "):
            elements.append({"type": "title", "text": stripped[2:].strip()})
        elif stripped.startswith("#"):
            elements.append({"type": "title", "text": stripped[1:].strip()})
        else:
            elements.append({"type": "paragraph", "text": stripped})
    return elements


# ── JS string escaper ─────────────────────────────────────────────────────────

def _escape_for_js_string(text: str) -> str:
    """
    Escapes a string for safe embedding inside a JavaScript template literal.

    Order is critical:
      1. Backslashes first — if any later escape sequence introduces a backslash
         it would be double-escaped if backslashes were processed afterwards.
      2. Backticks — would prematurely close the template literal.
      3. '${' sequences — the template literal interpolation syntax; only the
         combined two-character sequence is escaped, not bare $ or { alone.
      4. Newlines — replaced with the two-character literal \\n so the
         generated script string stays on one line.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = text.replace("${", "\\${")
    text = text.replace("\n", "\\n")
    return text


# ── Main export function ──────────────────────────────────────────────────────

def export_to_docx(topic: str, brief: str, sources_section: str) -> str | None:
    """
    Generates a Word document from the synthesised brief and returns its
    absolute path on success, or None on any failure.

    Steps:
      1. Validates that brief is non-empty.
      2. Parses the brief's markdown into typed elements.
      3. Builds the output directory.
      4. Generates a Node.js script that uses the docx npm package.
      5. Writes and executes the script via subprocess.
      6. Verifies the output file exists and has size > 0.
      7. Cleans up the temporary script file.
      8. Returns the absolute path to the .docx file.

    Returns None (never raises) on any failure — see module docstring.
    """
    # ── 1. Empty-content guard ────────────────────────────────────────────────
    if not brief or not brief.strip():
        logger.warning(
            "export_to_docx: brief is empty — no content to export, returning None."
        )
        return None

    # ── 2. Parse markdown structure ───────────────────────────────────────────
    elements = _parse_markdown_structure(brief)

    # ── 3. Build output directory ─────────────────────────────────────────────
    output_dir = get_output_dir(topic)
    os.makedirs(output_dir, exist_ok=True)

    docx_path = os.path.join(output_dir, "report.docx")
    script_path = os.path.join(output_dir, "_gen_docx.js")

    # ── 4. Generate Node.js script ────────────────────────────────────────────
    children_lines = []

    for el in elements:
        safe_text = _escape_for_js_string(el["text"])

        if el["type"] == "title":
            children_lines.append(
                f"    new Paragraph({{"
                f" heading: HeadingLevel.HEADING_1,"
                f" spacing: {{ after: 240 }},"
                f" children: [new TextRun({{ text: `{safe_text}`, bold: true }})]"
                f" }}),"
            )    
        elif el["type"] == "heading":
            children_lines.append(
                f"    new Paragraph({{"
                f" text: `{safe_text}`,"
                f" heading: HeadingLevel.HEADING_2,"
                f" spacing: {{ before: 240, after: 120 }}"
                f" }}),"
            )
        else:
            children_lines.append(
                f"    new Paragraph({{"
                f" children: [new TextRun({{ text: `{safe_text}`, font: 'Arial', size: 22 }})],"
                f" spacing: {{ after: 200 }}"
                f" }}),"
            )    

    # Sources section
    if sources_section and sources_section.strip():
        children_lines.append(
            "    new Paragraph({ text: `Sources`, heading: HeadingLevel.HEADING_2 }),"
        )
        for src_line in sources_section.splitlines():
            safe_src = _escape_for_js_string(src_line.strip())
            if safe_src:
                children_lines.append(
                    f"    new Paragraph({{"
                    f" children: [new TextRun({{ text: `{safe_src}`, font: 'Arial', size: 22 }})]"
                    f" }}),"
                )

    children_block = "\n".join(children_lines)
    safe_docx_path = _escape_for_js_string(os.path.abspath(docx_path))

    script = f"""\
const {{ Document, Paragraph, TextRun, HeadingLevel, PageSize, Packer }} = require('docx');
const fs = require('fs');

const doc = new Document({{
  sections: [{{
    properties: {{
      page: {{
        size: {{ width: 12240, height: 15840 }},
      }},
    }},
    children: [
{children_block}
    ],
  }}],
}});

Packer.toBuffer(doc).then((buffer) => {{
  fs.writeFileSync(`{safe_docx_path}`, buffer);
}}).catch((err) => {{
  process.stderr.write(String(err));
  process.exit(1);
}});
"""

    # ── 5–7. Write script, run it, verify output, clean up ───────────────────
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
                "export_to_docx: Node.js script exited with code %d. stderr: %s",
                result.returncode,
                result.stderr.strip(),
            )
            return None

        # Verify the output file exists and is non-empty
        if not os.path.exists(docx_path) or os.path.getsize(docx_path) == 0:
            logger.warning(
                "export_to_docx: subprocess reported success but output file "
                "is missing or empty at %s.",
                docx_path,
            )
            return None

        # Clean up the temporary script — only the .docx is the deliverable
        try:
            os.remove(script_path)
        except OSError as cleanup_err:
            # Non-fatal: log but don't fail the export over a cleanup issue
            logger.warning(
                "export_to_docx: failed to remove temporary script %s — %s",
                script_path,
                cleanup_err,
            )

        abs_path = os.path.abspath(docx_path)
        logger.info("export_to_docx: document written to %s.", abs_path)
        return abs_path

    except Exception as exc:
        logger.warning(
            "export_to_docx: export failed — %s: %s",
            type(exc).__name__,
            exc,
        )
        return None
