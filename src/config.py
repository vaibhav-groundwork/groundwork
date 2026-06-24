import os
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Models ────────────────────────────────────────────────
# Haiku  → fast, cheap: classification, routing, gap detection
# Sonnet → balanced: main synthesis, writing, analysis
# Judge  → deliberately separate to avoid self-grading bias (GPT-4o-mini via LiteLLM)
HAIKU_MODEL = "claude-haiku-4-5"
SONNET_MODEL = "claude-sonnet-4-5"
JUDGE_MODEL = "gpt-4o-mini"  # deliberately cross-provider (not Claude) to avoid self-grading bias

# ── Token limits ──────────────────────────────────────────
# max_tokens is a ceiling not a fixed charge — erring higher prevents silent truncation
MAX_TOKENS_HAIKU = 2048
MAX_TOKENS_SONNET = 8192
MAX_TOKENS_JUDGE = 1024

# ── RAG settings ──────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # free, runs locally, zero API cost
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
TOP_K_RESULTS = 3

# ── Agent limits ──────────────────────────────────────────
MAX_SEARCH_CALLS = 3          # hard cap on web searches per research run
MAX_REQUESTS_PER_SESSION = 10 # prevents runaway usage on deployed instance
MAX_FILE_SIZE_MB = 10         # cap on uploaded PDF size
MAX_REWRITE_ATTEMPTS = 2   # user-triggered regenerations per brief, after initial write

# ── Deployment ────────────────────────────────────────────
# Files are generated temporarily and served as downloads — not stored on server
# ChromaDB persists within a session but resets on redeploy (documented v1 limitation)
# v2 upgrade path: Pinecone for persistent memory, Railway for hosting
SERVE_FILES_AS_DOWNLOADS = True
TEMP_OUTPUT_DIR = "temp_outputs"

# ── Memory / run history ──────────────────────────────────
# Each completed run saves a summary to ChromaDB so users can
# browse previous research in the sidebar without re-running agents
MEMORY_COLLECTION = "groundwork_runs"
MEMORY_SUMMARY_MAX_CHARS = 500  # keeps summaries concise for retrieval

# ── LangSmith tracing ─────────────────────────────────────
# Set LANGCHAIN_TRACING_V2=true in .env to enable — defaults to off locally
LANGCHAIN_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "groundwork")

# ── Output helpers ────────────────────────────────────────
def get_output_dir(topic: str) -> str:
    """
    Returns a timestamped, topic-named temp directory for a single run.
    e.g. 'temp_outputs/2026-06-18_AI_product_trends'
    Used as staging area — files served as downloads then cleaned up.
    """
    date = datetime.now().strftime("%Y-%m-%d")
    slug = re.sub(r'[^a-zA-Z0-9_]', '_', topic.strip())[:40]
    return os.path.join(TEMP_OUTPUT_DIR, f"{date}_{slug}")


# ── Memory schema ─────────────────────────────────────────
def build_run_summary(
    topic: str,
    mode: str,
    eval_score: float,
    key_findings: str,
    output_dir: str
) -> dict:
    """
    Builds the metadata dict saved to ChromaDB after each completed run.
    Retrievable later to populate the 'Previous research' sidebar.
    mode: 'research' (web search) or 'document' (RAG on uploaded PDF)
    """
    return {
        "topic": topic,
        "mode": mode,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "eval_score": str(round(eval_score, 1)),
        "key_findings": key_findings[:MEMORY_SUMMARY_MAX_CHARS],
        "output_dir": output_dir,
        "type": "run_summary"
    }