# Singapore Housing Rental Assistant

An intelligent RAG (Retrieval-Augmented Generation) system powered by LangGraph that helps international students navigate the Singapore rental housing market.

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2.3-green)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0.5-orange)
![Gradio](https://img.shields.io/badge/Gradio-6.3.0-ff7c00)
![Tests](https://img.shields.io/badge/tests-73%20passed-brightgreen)

## Overview

An AI-powered conversational assistant for international students seeking rental accommodation in Singapore. It combines advanced RAG techniques with LangGraph's agentic workflow to answer questions about HDB vs Condo comparisons, rental prices, area recommendations, commute times, and more — with full bilingual support (English/Chinese).

## Key Features

- **Agentic RAG Workflow**: LangGraph multi-step pipeline — summarize, analyze, rewrite, parallel agent execution, aggregate
- **Hybrid Retrieval**: Dense (all-mpnet-base-v2) + Sparse (BM25) search via Qdrant
- **Parent-Child Chunking**: Two-stage retrieval for precise search with rich context
- **Query Decomposition**: Complex questions are split into sub-questions and processed in parallel
- **Google Maps Integration**: Real-time commute times, directions, and nearby amenities
- **Human-in-the-Loop**: Asks clarifying questions when the query is ambiguous
- **Token-Level Streaming**: Real-time response generation in the web UI
- **Bilingual Support**: English and Chinese UI and responses
- **Conversation Persistence**: SQLite-backed checkpointing — conversations survive restarts
- **Source Attribution**: Every answer cites its knowledge base sources

## Architecture

```
Main Graph:
  START → summarize → analyze_rewrite → route_after_rewrite
                                         ├→ human_input (query unclear → interrupt)
                                         └→ [Send] process_question ×N (parallel)
                                                    ↓
                                               agent ⇄ tools → extract_answer
                                                    ↓
                                                aggregate → END
```

**Nodes:**
| Node | Purpose |
|------|---------|
| `summarize` | Compresses conversation history (last 6 turns) for context management |
| `analyze_rewrite` | Rewrites query for retrieval, may split into 1-3 sub-questions |
| `process_question` | Each sub-question runs its own agent subgraph in parallel via `Send` |
| `aggregate` | Combines all agent answers into a coherent response with citations |

**Tools available to each agent:**
| Tool | Description |
|------|-------------|
| `search_child_chunks` | Hybrid search (dense + sparse), top 7, threshold 0.7 |
| `retrieve_parent_chunks` | Fetch full parent context by chunk ID |
| `get_commute_info` | Transit + driving time via Google Maps Distance Matrix |
| `get_directions` | Step-by-step transit directions |
| `search_nearby` | Nearby amenities (MRT, supermarkets, restaurants, etc.) |

### Retrieval Strategy
- **Parent Chunks** (2000-10000 chars): Full context stored as JSON in `parent_store/`
- **Child Chunks** (500 chars, 100 overlap): Searchable units indexed in Qdrant
- **Hybrid Search**: Dense semantic similarity + BM25 keyword matching

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Workflow** | LangGraph 1.0.5 (main graph + agent subgraph) |
| **RAG** | LangChain 1.2.3 |
| **LLM** | Google Gemini `gemini-2.5-flash` (configurable to OpenAI) |
| **Embeddings** | `all-mpnet-base-v2` (768d dense) + `Qdrant/bm25` (sparse) |
| **Vector DB** | Qdrant (local mode) |
| **Persistence** | SQLite (conversation checkpoints) |
| **Maps** | Google Maps API (Distance Matrix, Directions, Places) |
| **UI** | Gradio 6.3.0 |
| **Language** | Python 3.12+ |

## Installation

### Prerequisites
- Python 3.12+
- Google Gemini API key (or OpenAI)
- Google Maps API key (optional, for location features)

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/singapore-housing-assistant.git
cd singapore-housing-assistant

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your keys (see Environment Variables below)

# Index documents (first time)
python indexing.py --rebuild

# Run assistant
python app.py        # Web UI (recommended)
python test_chat.py  # Command line
```

## Usage

### Web Interface (Recommended)

```bash
python app.py
```

Open http://localhost:7860 in your browser. Features:
- Language toggle (English / Chinese)
- Token-level streaming responses
- Conversation history persisted across restarts
- Clear chat to start a new conversation

### Command Line Chat

```bash
python test_chat.py
```

Commands: `help`, `clear`, `exit`

### Example Questions

```
What is the difference between HDB and Condo?
How much does it cost to rent in Clementi?
Which areas are good for NUS students?
How long is the commute from Jurong East to NUS by MRT?
What should I watch out for in rental scams?
How do I set up utilities after moving in?
```

## Project Structure

```
singapore-housing-assistant/
├── app.py                         # Gradio web UI entry point
├── indexing.py                    # Document indexing CLI (--rebuild / --append)
├── test_chat.py                   # CLI chat interface
├── requirements.txt
├── .env.example
├── docs/                          # Knowledge base (11 markdown documents)
│   ├── hdb_vs_condo.md
│   ├── rental_guide.md
│   ├── price_range.md
│   ├── area_guide_central.md
│   ├── area_guide_east.md
│   ├── area_guide_west.md
│   ├── rental_scams.md
│   ├── student_budget_tips.md
│   ├── transport_guide.md
│   ├── utilities_setup.md
│   └── visa_housing_rules.md
├── src/
│   ├── config.py                  # All settings and constants
│   ├── i18n.py                    # EN/ZH translations and UI text
│   ├── db/
│   │   └── parent_store_manager.py  # Parent chunk CRUD (JSON + LRU cache)
│   ├── rag_agent/
│   │   ├── base.py                # BaseToolFactory ABC + timed_tool decorator
│   │   ├── graph.py               # LangGraph graph construction
│   │   ├── graph_state.py         # State classes + custom reducers
│   │   ├── nodes.py               # Node implementations
│   │   ├── prompts.py             # System prompts for LLM
│   │   ├── tools.py               # RAG retrieval tools (ToolFactory)
│   │   └── maps_tools.py          # Google Maps tools (MapsToolFactory)
│   └── ui/
│       └── gradio_app.py          # Gradio UI + streaming + ChatSession
├── tests/                         # Unit tests (73 test cases)
│   ├── test_config.py
│   ├── test_graph_state.py
│   ├── test_i18n.py
│   ├── test_maps_normalize.py
│   ├── test_parent_store.py
│   └── test_prompts.py
├── parent_store/                  # Parent chunk JSON storage
└── qdrant_db/                     # Qdrant local vector database
```

## Configuration

Edit `src/config.py` to customize:

```python
# LLM
LLM_PROVIDER = "gemini"       # or "openai"
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0

# Chunking
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 10000

# Retrieval
TOP_K_CHILD_CHUNKS = 7
SIMILARITY_THRESHOLD = 0.7
MAX_PARENT_RETRIEVAL = 3

# Google Maps
MAPS_SEARCH_RADIUS = 1000     # meters
MAPS_MAX_RESULTS = 8
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required: At least one LLM API key
GOOGLE_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key       # Optional alternative

# Optional: Google Maps features
GOOGLE_MAPS_API_KEY=your-maps-api-key

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=singapore-housing-assistant
LANGCHAIN_API_KEY=your-langsmith-key
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v
```

73 unit tests covering configuration, state management, i18n, location normalization, parent store CRUD, and prompt generation.

## Engineering Highlights

| Feature | Why It Matters |
|---------|---------------|
| **LangGraph multi-step workflow** | Not a simple chain — summarize → analyze → parallel agents → aggregate |
| **Parent-Child two-stage retrieval** | Understands context window limits; precise search with rich context |
| **Hybrid search (dense + sparse)** | Combines semantic understanding with keyword precision |
| **Query decomposition + parallel execution** | Complex questions split into sub-questions processed concurrently |
| **Human-in-the-loop** | Asks for clarification instead of guessing — production-grade UX |
| **Token-level streaming** | `astream_events` with tag filtering for real-time output |
| **Google Maps tool integration** | External API tools alongside document RAG |
| **SQLite persistence** | Conversations survive app restarts |
| **BaseToolFactory + timed_tool** | Clean abstractions with automatic performance logging |

## Future Enhancements

- [x] Web UI with Gradio
- [x] Bilingual support (English / Chinese)
- [x] Google Maps integration
- [x] Token-level streaming
- [x] Conversation persistence (SQLite)
- [x] Knowledge base expansion (3 → 11 documents)
- [ ] RAG evaluation system (answer relevance, retrieval precision)
- [ ] Cross-Encoder reranking for improved retrieval quality
- [ ] Property listing integration (PropertyGuru API)
- [ ] PDF document upload via Web UI

## Acknowledgments

- Inspired by [agentic-rag-for-dummies](https://github.com/GiovanniPasq/agentic-rag-for-dummies)
- Built with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by Google Gemini / OpenAI

## License

This project is licensed under the MIT License.

---

**Built for international students navigating Singapore's rental market**
