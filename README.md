# Groundwork

## Motivation
I built this app as a learning as well as a solution to problems I had during my MBA. I wanted to apply the new AI infrastructure concepts to build a personal app and also wanted to have the ability to have grounded research work combining web and constrained to a particular document source. I didn't want to just build an LLM wrapper, I wanted to implement sound logic and modern computational concepts. During the MBA I was spending too much time searching the web & LLM models, digging through course readings and case studies, then pulling everything together into structured reports for my coursework and capstone project. I wanted a tool that I built and could do the heavy lifting intelligently, providing me with synthesized reports as either documents or powerpoint presentations.

Groundwork is the result. It is a multi-agent research and document intelligence tool that lets you gather information from the web or your own files, build up a conversation across multiple questions, and generate an audited, exportable report. Every architectural decision in this project maps to a real AI infrastructure concept I learned and implemented from scratch.

**Live app:** [groundwork.streamlit.app](https://groundwork.streamlit.app)

---

## What it does

Groundwork has two fully independent modes.

**Research mode** lets you ask any question and the app autonomously searches the web, identifies gaps in what it found, runs additional targeted searches, and synthesises structured notes. Follow-up questions are handled intelligently: if your follow-up can be answered from what has already been found, it answers directly without running a new search. If it needs new information, it searches again with full awareness of what came before.

**Document mode** lets you upload PDFs or Word documents and ask questions grounded strictly in their content. The app finds the most relevant passages in your files and answers only from those. No information is added from outside the documents you upload.

In both modes you can ask multiple questions in sequence, building up a conversation and a body of material. When you are ready, you generate a full report. The report is automatically evaluated by a separate AI judge (a different model from a different provider, to avoid self-grading bias), with a quality score across four dimensions. If the score is not satisfactory, you can regenerate once, and the new version specifically targets whichever dimension scored lowest. Both the original and regenerated report stay visible side by side. You can export either version as a Word document or a PowerPoint deck.

The two modes are kept strictly separate throughout. Research findings come from the open web. Document findings come strictly from your uploaded files. They are never mixed, because their reliability guarantees are fundamentally different.

---

## AI infrastructure concepts implemented

This project was built as a deliberate exercise in learning and applying the concepts that matter most for AI-native product development. Every item below is actually in the codebase, not just referenced.

**RAG (Retrieval-Augmented Generation)**
Document mode uses a full RAG pipeline: uploaded files are chunked into overlapping segments, embedded using a local sentence-transformer model, stored in ChromaDB, and retrieved semantically at query time. Answers are grounded strictly in retrieved chunks.

**Chunking and Embedding**
Text from PDFs and Word documents is split into 300-token chunks with 30-token overlap to preserve context across boundaries. Embeddings are generated locally using `sentence-transformers/all-MiniLM-L6-v2` with no API cost.

**Vector Database**
ChromaDB stores and retrieves document embeddings. Each user session gets its own collection, keeping document contexts isolated.

**LangGraph (Agentic Orchestration)**
The research pipeline is built as a LangGraph StateGraph. The graph loops autonomously through search and analysis nodes until Claude decides it has enough information or a hard cap is reached. LangGraph was chosen specifically for this part of the pipeline because it is the only part that genuinely loops and branches without human input between steps. Everything else is plain function calls.

**Agentic Routing (LLM-as-Router)**
A routing node at the start of the research graph reads the user's follow-up question and the accumulated prior context, then decides whether to answer from what has already been found or trigger a new web search. This is a tool-call-based routing decision, not a keyword match.

**Tool-Based Structured Output (Level 3)**
All LLM decisions that drive graph behaviour use OpenAI-style function calling with explicit schemas, not prompt-based JSON parsing. This includes gap analysis, follow-up routing, and judge scoring. Structured output at this level guarantees schema compliance regardless of how the model chooses to phrase its response.

**Multi-LLM / Multi-Provider Architecture**
Different models are used for different tasks based on their strengths and cost profiles. Claude Haiku handles fast, cheap classification and routing decisions. Claude Sonnet handles synthesis and PPTX slide planning. GPT-4o-mini handles quality evaluation. LiteLLM provides a unified interface across providers.

**LLM-as-Judge Evaluation**
After every report is generated, a separate judge agent (GPT-4o-mini) scores it across four dimensions: accuracy, groundedness, helpfulness, and conciseness. Using a different model from a different company as the judge avoids the self-grading bias that comes from asking the same model to evaluate its own output.

**Prompt Caching**
System prompts for frequently-called nodes use Anthropic's prompt caching feature, reducing token costs and latency on repeated calls within a session.

**Multi-Turn State Accumulation**
Both modes maintain a running conversation history across multiple questions within a session. Each question and its answer are stored in session state and included as context when generating the final report.

**LangSmith Tracing**
All LLM calls are instrumented with LangSmith tracing when enabled. This makes it possible to inspect every prompt, response, token count, and latency for every call in the pipeline, which was essential for debugging during development.

**Rate Limiting and Cost Guardrails**
The research loop enforces a hard cap on web searches per run. Regeneration is capped at one attempt per mode. File uploads have a size limit. These guardrails keep the app safe and cost-predictable when used by anyone other than the developer.

**Two-Stage Generative Export**
Both the DOCX and PPTX exporters use a two-stage architecture. In stage one, Claude produces a structured plan for the document (slide layout and content for PPTX, parsed markdown structure for DOCX). In stage two, Python generates a self-contained Node.js script that renders the file deterministically using npm packages. Once the plan is made, no further LLM calls happen. Layout issues are fixed in code, not by re-prompting.

---

## Architecture overview

```
app.py (Streamlit)
|
+-- Research mode
|   +-- LangGraph StateGraph
|   |   +-- route_followup_node     (LLM decides: answer from context or new search)
|   |   +-- answer_from_notes_node  (answers follow-ups from accumulated notes)
|   |   +-- search_node             (SerpAPI web search)
|   |   +-- analyse_node            (Claude Haiku gap analysis + synthesis)
|   +-- run_research()              (validated public entry point)
|
+-- Document mode
|   +-- ingest_document()           (PDF/DOCX chunking + ChromaDB indexing)
|   +-- rag_node()                  (semantic retrieval + grounded answer)
|
+-- Report generation (both modes, independently)
|   +-- synthesis_node()            (Claude Sonnet structured report writing)
|   +-- judge_node()                (GPT-4o-mini cross-provider quality scoring)
|   +-- Regeneration                (capped at 1, targets weakest dimension)
|
+-- Export
    +-- export_to_docx()            (Node.js docx package via subprocess)
    +-- export_to_pptx()            (Node.js pptxgenjs + Claude slide planning)
```

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent orchestration | LangGraph |
| LLM (research, synthesis, routing) | Anthropic Claude Haiku + Sonnet |
| LLM (judge) | OpenAI GPT-4o-mini via LiteLLM |
| Web search | SerpAPI |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (local, no API cost) |
| DOCX export | Node.js docx npm package |
| PPTX export | Node.js pptxgenjs npm package |
| Tracing | LangSmith |

---

## Local setup

Note: activate the virtual environment each time you open a new terminal session before running the app.

### Prerequisites

- Python 3.11 or higher
- Node.js 18 or higher (required for DOCX and PPTX export)
- npm (comes with Node.js)

### 1. Clone the repo

```bash
git clone https://github.com/vaibhav-groundwork/groundwork.git
cd groundwork
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS and Linux
venv\Scripts\activate           # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Node.js dependencies

```bash
npm install
```

### 5. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:

```
# Required
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key
SERPAPI_KEY=your-serpapi-key

# Optional (LangSmith tracing)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=groundwork
```

Where to get each key:

- `ANTHROPIC_API_KEY` at [console.anthropic.com](https://console.anthropic.com)
- `OPENAI_API_KEY` at [platform.openai.com](https://platform.openai.com)
- `SERPAPI_KEY` at [serpapi.com](https://serpapi.com) — 100 free searches per month on the free tier
- `LANGCHAIN_API_KEY` at [smith.langchain.com](https://smith.langchain.com) — only needed if you enable tracing

### 6. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Project structure

```
groundwork/
+-- app.py                          (Streamlit UI)
+-- src/
|   +-- agents/
|   |   +-- research_agent.py       (LangGraph nodes for research pipeline)
|   |   +-- rag_agent.py            (document Q&A with ChromaDB retrieval)
|   |   +-- synthesis_agent.py      (Claude Sonnet report writer)
|   |   +-- judge_agent.py          (GPT-4o-mini quality scorer)
|   +-- exporters/
|   |   +-- docx_export.py          (Word document generation)
|   |   +-- pptx_export.py          (PowerPoint generation)
|   +-- graph.py                    (LangGraph StateGraph)
|   +-- ingestion.py                (chunking, embedding, ChromaDB indexing)
|   +-- config.py                   (constants, model names, API keys)
|   +-- tracing.py                  (LangSmith setup and LLM client)
|   +-- utils.py                    (shared helpers)
+-- docs/
|   +-- architecture.md             (detailed architecture decisions)
|   +-- groundwork_v2_backlog.md    (known issues and future features)
+-- .env.example                    (environment variable template)
+-- package.json                    (Node.js dependencies)
+-- requirements.txt                (Python dependencies)
```

---

## Environment variables reference

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key (Haiku and Sonnet) |
| `OPENAI_API_KEY` | Yes | OpenAI API key (GPT-4o-mini for judge) |
| `SERPAPI_KEY` | Yes | SerpAPI key for web search |
| `LANGCHAIN_TRACING_V2` | No | Set to true to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | LangSmith API key (required if tracing is on) |
| `LANGCHAIN_PROJECT` | No | LangSmith project name (defaults to groundwork) |

---

## Known limitations and backlog

See [`docs/groundwork_v2_backlog.md`](docs/groundwork_v2_backlog.md) for the full list. Key items:

- Context-dependent follow-up search queries may not fully resolve conversational references on the first search pass
- PPTX charts and data visualisations are out of scope for v1
- Multi-session and parallel conversation threads are not implemented

---

## License

MIT