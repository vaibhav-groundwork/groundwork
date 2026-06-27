"""
utils.py — small, shared text-cleaning helpers used across multiple agents.

Why this exists separately from config.py:
  config.py holds configuration values and constants. This file holds actual
  logic — pure functions with no side effects, no API calls, no state — that
  multiple agent files need identically. Keeping these separate from config.py
  preserves a clean distinction: config.py answers "what are the settings?",
  utils.py answers "what small, repeatable text transformations do we apply?"

Deterministic backstops over prompt instructions:
  Every function here exists because a prompt-only instruction proved
  unreliable for the same pattern elsewhere in this codebase — see
  synthesis_agent.py's structure-length fix and regeneration under-revision
  fix. The lesson generalizes: stylistic or structural patterns Claude should
  follow consistently are more reliably enforced as deterministic code than
  as a hope embedded in a system prompt.
"""


def strip_em_dashes(text: str) -> str:
    """
    Replaces em dashes with a comma, since em dashes are a common LLM writing
    tic explicitly out of scope for Groundwork's user-facing text. Applied as
    a deterministic backstop to a system-prompt instruction, not a substitute
    for it — both layers are used together, consistent with this project's
    established pattern for enforcing stylistic/structural rules reliably.

    Handles both ' — ' (spaced, most common) and a bare '—' (unspaced),
    falling back to a comma in both cases for natural-reading replacement.
    """
    if not text:
        return text
    return text.replace(" — ", ", ").replace("—", ", ")

def _escape_for_js_string(text: str) -> str:
    """
    Escapes a string for safe embedding inside a JavaScript template literal.

    Shared between docx_export.py and pptx_export.py, both of which generate
    Node.js scripts at runtime using the same code-generation pattern. Moved
    here rather than duplicated once a second file needed it identically —
    consistent with this project's "one source of truth" principle.

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