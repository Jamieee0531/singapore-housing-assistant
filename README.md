# Singapore Housing Rental Assistant

An intelligent RAG (Retrieval-Augmented Generation) system powered by LangGraph that helps international students navigate the Singapore rental housing market.

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2.3-green)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0.5-orange)
![Gradio](https://img.shields.io/badge/Gradio-6.3.0-ff7c00)
![Tests](https://img.shields.io/badge/tests-84%20passed-brightgreen)

## Overview

An AI-powered conversational assistant for international students seeking rental accommodation in Singapore. It combines advanced RAG techniques with LangGraph's agentic workflow to answer questions about HDB vs Condo comparisons, rental prices, area recommendations, commute times, and more — with full bilingual support (English/Chinese).

## Key Features

- **Agentic RAG Workflow**: LangGraph multi-step pipeline — summarize, analyze, rewrite, parallel agent execution, aggregate
- **Hybrid Retrieval + Re-ranking**: Dense + Sparse search via Qdrant, re-ranked by Cross-Encoder for higher precision
- **Metadata Filtering**: Topic-based filtering narrows search scope before retrieval
- **Parent-Child Chunking**: Two-stage retrieval for precise search with rich context
- **Four-Layer Memory Architecture**: Long-term (Redis) + summarization + sliding window + current input
- **Query Decomposition**: Complex questions split into sub-questions and processed in parallel
- **Google Maps Integration**: Real-time commute times, directions, and nearby amenities
- **Human-in-the-Loop**: Asks clarifying questions when the query is ambiguous
- **RAG Evaluation System**: Precision@k, MRR, Faithfulness, Answer Relevance (LLM-as-Judge)
- **Token-Level Streaming**: Real-time response generation in the web UI
- **Bilingual Support**: English and Chinese UI and responses

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
| `summarize` | Compresses history + loads user profile from Redis (long-term memory) |
| `analyze_rewrite` | Rewrites query, extracts user preferences → Redis, classifies topics for filtering |
| `process_question` | Each sub-question runs its own agent subgraph in parallel via `Send` |
| `aggregate` | Combines all agent answers into a coherent response with citations |

**Tools available to each agent:**
| Tool | Description |
|------|-------------|
| `search_child_chunks` | Hybrid search + metadata filtering + cross-encoder re-ranking |
| `retrieve_parent_chunks` | Fetch full parent context by chunk ID |
| `get_area_history` | Recall previously explored areas from Redis |
| `get_commute_info` | Transit + driving time via Google Maps |
| `get_directions` | Step-by-step transit directions |
| `search_nearby` | Nearby amenities (MRT, supermarkets, restaurants) |

### Retrieval Pipeline

```
Query → Hybrid Search (dense + sparse) → Top 10 candidates
    → Metadata Filter (topic labels) → Narrow scope
    → Cross-Encoder Re-ranking → Top 5 results
    → Agent retrieves parent chunks for full context
```

- **Parent Chunks** (2000-10000 chars): Full context stored as JSON
- **Child Chunks** (500 chars, 100 overlap): Searchable units indexed in Qdrant
- **7 Topic Labels**: `housing_types`, `pricing`, `area`, `transport`, `utilities`, `rental_process`, `legal`

### Memory Architecture

| Layer | Content | Strategy | Storage |
|-------|---------|----------|---------|
| Layer 1 | User preferences (school, budget, area) | Persistent, injected into prompt | Redis Hash + Sorted Set |
| Layer 2 | Conversation summary | LLM summarization | LangGraph State |
| Layer 3 | Recent messages | Sliding window | LangGraph State |
| Layer 4 | Current user input | Full retention | LangGraph State |

- **Long-term memory** (Redis): User preferences persist across sessions. Area exploration history tracked with timestamps.
- **Short-term memory** (SQLite checkpoint): Conversation state with summarization after 10 messages.

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Workflow** | LangGraph 1.0.5 (main graph + agent subgraph) |
| **RAG** | LangChain 1.2.3 |
| **LLM** | Google Gemini `gemini-2.5-flash` (configurable to OpenAI) |
| **Embeddings** | `all-mpnet-base-v2` (768d dense) + `Qdrant/bm25` (sparse) |
| **Re-ranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Vector DB** | Qdrant (local mode, hybrid search) |
| **Long-term Memory** | Redis (Docker, user profiles + area history) |
| **Short-term Memory** | SQLite (conversation checkpoints) |
| **Maps** | Google Maps API (Distance Matrix, Directions, Places) |
| **UI** | Gradio 6.3.0 |
| **Language** | Python 3.12+ |

## Installation

### Prerequisites
- Python 3.12+
- Docker (for Redis)
- Google Gemini API key (or OpenAI)
- Google Maps API key (optional, for location features)

### Quick Start

```bash
# Clone repository
git clone https://github.com/Jamieee0531/singapore-housing-assistant.git
cd singapore-housing-assistant

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start Redis (Docker)
docker run -d --name redis -p 6379:6379 -p 8001:8001 redis/redis-stack

# Configure API keys
cp .env.example .env
# Edit .env and add your keys

# Index documents
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
- User preferences remembered across sessions (Redis)

### Example Questions

```
What is the difference between HDB and Condo?
How much does it cost to rent in Clementi?
Which areas are good for NUS students?
How long is the commute from Jurong East to NUS by MRT?
What should I watch out for in rental scams?
How do I set up utilities after moving in?
What areas have I looked at before?
```

### RAG Evaluation

```bash
# Run full evaluation (retrieval + response)
python -m evaluate.run

# Retrieval only (Precision@k, MRR)
python -m evaluate.run --retrieval

# Response only (LLM-as-Judge: Faithfulness, Answer Relevance)
python -m evaluate.run --response
```

Reports are saved to `evaluate/results/`.

## Project Structure

```
singapore-housing-assistant/
├── app.py                         # Gradio web UI entry point
├── indexing.py                    # Document indexing CLI (--rebuild / --append)
├── test_chat.py                   # CLI chat interface
├── requirements.txt
├── .env.example
├── docs/                          # Knowledge base (11 markdown documents)
├── src/
│   ├── config.py                  # All settings, constants, and topic mappings
│   ├── i18n.py                    # EN/ZH translations and UI text
│   ├── db/
│   │   ├── parent_store_manager.py  # Parent chunk CRUD (JSON + LRU cache)
│   │   └── redis_manager.py         # Redis long-term memory (profiles + area history)
│   ├── rag_agent/
│   │   ├── base.py                # BaseToolFactory ABC + timed_tool decorator
│   │   ├── graph.py               # LangGraph graph construction
│   │   ├── graph_state.py         # State classes + custom reducers
│   │   ├── nodes.py               # Node implementations
│   │   ├── prompts.py             # System prompts for LLM
│   │   ├── tools.py               # RAG tools + re-ranking + metadata filter
│   │   └── maps_tools.py          # Google Maps tools (MapsToolFactory)
│   └── ui/
│       └── gradio_app.py          # Gradio UI + streaming + ChatSession
├── evaluate/                      # RAG evaluation system
│   ├── run.py                     # Entry point: python -m evaluate.run
│   ├── dataset.json               # 16 test cases (NotebookLM-verified)
│   ├── retrieval_eval.py          # Precision@k, MRR
│   ├── response_eval.py           # LLM-as-Judge (Faithfulness, Answer Relevance)
│   ├── report.py                  # Markdown report generator
│   └── results/                   # Generated reports
└── tests/                         # Unit tests (84 test cases)
    ├── test_config.py
    ├── test_graph_state.py
    ├── test_i18n.py
    ├── test_maps_normalize.py
    ├── test_parent_store.py
    ├── test_prompts.py
    └── test_metadata_filter.py
```

## Engineering Highlights

| Feature | Why It Matters |
|---------|---------------|
| **LangGraph multi-step workflow** | Not a simple chain — summarize → analyze → parallel agents → aggregate |
| **Four-layer memory architecture** | Redis long-term + summarization + sliding window + current input |
| **Hybrid search + Cross-Encoder re-ranking** | Dense + sparse retrieval, then cross-encoder precision ranking |
| **Metadata filtering** | Topic labels reduce search scope, preventing irrelevant documents from dominating results |
| **Parent-Child two-stage retrieval** | Precise search (small chunks) with rich context (parent chunks) |
| **Query decomposition + parallel execution** | Complex questions split into sub-questions processed concurrently |
| **RAG evaluation system** | Retrieval metrics (Precision@k, MRR) + response quality (LLM-as-Judge) |
| **Human-in-the-loop** | Asks for clarification instead of guessing — production-grade UX |
| **Token-level streaming** | `astream_events` with tag filtering for real-time output |
| **User preference extraction** | Piggybacks on existing LLM call — zero additional latency |

## Future Enhancements

- [ ] User onboarding flow (collect preferences on first visit)
- [ ] Property listing integration (PropertyGuru API)
- [ ] PDF document upload via Web UI
- [ ] Semantic chunking (SemanticChunker)

## Acknowledgments

- Inspired by [agentic-rag-for-dummies](https://github.com/GiovanniPasq/agentic-rag-for-dummies)
- Built with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by Google Gemini / OpenAI

## License

This project is licensed under the MIT License.

---

**Built for international students navigating Singapore's rental market**
