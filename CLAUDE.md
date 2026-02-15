# CLAUDE.md — Singapore Housing Assistant

## Project Overview

A LangGraph-based RAG system that helps international students navigate Singapore's rental market. The assistant answers housing questions using a curated knowledge base (markdown docs), supports hybrid retrieval (dense + sparse search via Qdrant), integrates Google Maps for location/commute queries, and provides a Gradio web UI with English/Chinese support.

## Tech Stack

- **Python** 3.12+
- **LangGraph** 1.0.5 — agentic workflow engine (main graph + agent subgraph)
- **LangChain** 1.2.3 — RAG orchestration, tool binding
- **Qdrant** (local, via qdrant-client 1.16.2) — hybrid vector search
- **Embeddings**: `all-mpnet-base-v2` (dense, 768d) + `Qdrant/bm25` (sparse)
- **LLM**: Google Gemini `gemini-2.5-flash` (configurable to OpenAI)
- **Gradio** 6.3.0 — web interface
- **Google Maps API** — commute, directions, nearby search

## Project Structure

```
├── app.py                 # Gradio web UI entry point
├── indexing.py            # Document indexing CLI (--rebuild / --append)
├── test_chat.py           # CLI chat interface for testing
├── docs/                  # Knowledge base (markdown files)
├── parent_store/          # Parent chunk JSON storage
├── qdrant_db/             # Qdrant local vector database
└── src/
    ├── config.py          # All settings: models, chunking params, thresholds
    ├── i18n.py            # EN/ZH UI translations
    ├── db/
    │   ├── parent_store_manager.py   # Parent chunk CRUD (JSON files)
    │   └── vector_db_manager.py      # (empty, planned)
    ├── rag_agent/
    │   ├── graph.py           # LangGraph graph construction
    │   ├── graph_state.py     # State classes + custom reducers
    │   ├── nodes.py           # Node implementations (summarize, analyze, aggregate)
    │   ├── prompts.py         # System prompts for LLM
    │   ├── tools.py           # RAG retrieval tools (ToolFactory)
    │   └── maps_tools.py      # Google Maps tools (MapsToolFactory)
    └── ui/
        └── gradio_app.py      # Gradio UI setup
```

## Architecture

```
Main Graph:
  START → summarize → analyze_rewrite → route_after_rewrite
                                         ├→ human_input (query unclear, interrupt)
                                         └→ [Send] process_question ×N → aggregate → END

Agent Subgraph (per sub-question):
  START → agent ⇄ tools → extract_answer → END
```

- **summarize**: Compresses conversation history (last 6 turns) for context management
- **analyze_rewrite**: Rewrites query for retrieval, may split into 1-3 sub-questions
- **process_question**: Each sub-question runs its own agent with RAG + Maps tools in parallel via `Send`
- **aggregate**: Combines agent answers into a coherent response with source citations

**Tools available to agents:**
- `search_child_chunks` — hybrid search, top 7, threshold 0.7
- `retrieve_parent_chunks` — fetch full parent context by ID
- `get_commute_info`, `get_directions`, `search_nearby` — Google Maps

## Development Guidelines

- **Language**: All docstrings, comments, and variable names in English
- **Type hints**: Required on all public functions
- **Error handling**: Return structured responses; no bare `except`; no silent swallowing
- **Logging**: Use `logging` module, not `print()` (migration in progress)
- **Testing**: pytest; aim for 80%+ coverage; mock external APIs (Gemini, Maps, Qdrant)
- **Code quality**: Portfolio-grade — clean, well-documented, production-ready patterns
- **Config**: All magic numbers belong in `src/config.py`, not inline
- **Immutability**: Create new state objects, never mutate existing ones (LangGraph pattern)

## Working with Claude Code

**Role**: Full-stack development partner for this project.

**Workflow rules**:
- For significant changes (new nodes, architecture changes, tool modifications): explain the approach and wait for confirmation before implementing
- For small fixes (typos, formatting, obvious bugs): go ahead and fix directly
- When making technical decisions: present 2-3 options with trade-offs + recommend the best one
- Before adding new features: read `OPTIMIZATION_PLAN.md` to check priorities and known issues

**Code conventions**:
- Follow existing patterns (e.g., ToolFactory for tools, custom reducers for state)
- Keep `config.py` as the single source of truth for all tunable parameters
- Graph modifications should update both the graph builder and the state classes

## Key References

- **`OPTIMIZATION_PLAN.md`** — Prioritized roadmap of bugs, improvements, and future features. Always check this before starting new work to align with existing plans and avoid duplicating effort.
