# 🏠 Singapore Housing Rental Assistant

An intelligent RAG (Retrieval-Augmented Generation) system powered by LangGraph that helps international students navigate the Singapore rental housing market.

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2.3-green)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0.5-orange)
![Gradio](https://img.shields.io/badge/Gradio-6.3.0-ff7c00)

## 📋 Overview

This project provides an AI-powered conversational assistant specifically designed for international students seeking rental accommodation in Singapore. It uses advanced RAG techniques with LangGraph's agentic workflow to deliver accurate, contextual information about HDB vs Condo comparisons, rental prices, processes, and area recommendations.

## ✨ Key Features

- **🌐 Web Interface**: Modern Gradio-based chat UI
- **📚 Intelligent Document Retrieval**: Hybrid search using dense and sparse embeddings
- **🔄 Multi-Turn Conversations**: Natural follow-up questions with context awareness
- **🎯 Query Analysis**: Automatic query rewriting and clarification
- **🔗 Parent-Child Chunking**: Two-stage retrieval for better context
- **📖 Source Attribution**: Always cites sources for transparency
- **🤖 Agentic Workflow**: LangGraph-powered multi-step reasoning

## 🏗️ Architecture

```
User Query → Query Analysis → Agent Subgraph → Answer Generation
                                    ↓
                            ┌───────────────┐
                            │ Search Tools  │
                            ├───────────────┤
                            │ Child Chunks  │ → Vector Search
                            │ Parent Chunks │ → Context Retrieval
                            └───────────────┘
```

### Retrieval Strategy
- **Parent Chunks** (2000-10000 chars): Context stored as JSON
- **Child Chunks** (500 chars): Searchable units in Qdrant
- **Hybrid Search**: Dense (semantic) + Sparse (keyword)

## 🛠️ Tech Stack

**Core**: LangChain 1.2.3, LangGraph 1.0.5, Python 3.12+
**LLM**: Google Gemini (gemini-2.5-flash) / OpenAI GPT
**Embeddings**: sentence-transformers/all-mpnet-base-v2, Qdrant/bm25
**Database**: Qdrant (local vector store)
**UI**: Gradio 6.3.0

## 📦 Installation

### Prerequisites
- Python 3.12+
- Google Gemini API key (or OpenAI)

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

# Configure API key
cp .env.example .env
# Edit .env and add: GOOGLE_API_KEY=your-key-here

# Index documents
python indexing.py

# Run assistant (choose one)
python app.py        # Web UI (recommended)
python test_chat.py  # Command line
```

## 🚀 Usage

### Web Interface (Recommended)

```bash
python app.py
```

Open http://localhost:7860 in your browser.

![Web UI Screenshot](docs/screenshot.png)

### Command Line Chat

```bash
python test_chat.py
```

### Example Questions

```
You: What is the difference between HDB and Condo?
You: How much does it cost to rent in Clementi?
You: Which areas are good for NUS students?
You: Tell me about rental deposits and contracts
```

### Commands
- `help` - Show example questions
- `clear` - Start new conversation
- `exit` - Quit application

## 📁 Project Structure

```
singapore-housing-assistant/
├── docs/                      # Knowledge base documents
│   ├── hdb_vs_condo.md
│   ├── rental_guide.md
│   ├── price_range.md
│   └── ...
├── src/
│   ├── config.py              # Configuration
│   ├── core/                  # Core utilities
│   │   ├── document_manager.py
│   │   └── embeddings.py
│   ├── db/                    # Data access layer
│   │   ├── parent_store_manager.py
│   │   └── vector_db_manager.py
│   ├── rag_agent/             # RAG logic
│   │   ├── graph_state.py    # State definitions
│   │   ├── prompts.py        # System prompts
│   │   ├── tools.py          # Retrieval tools
│   │   ├── nodes.py          # Graph nodes
│   │   └── graph.py          # Main graph
│   └── ui/                    # User interface
│       └── gradio_app.py     # Gradio web interface
├── app.py                     # Web UI entry point
├── indexing.py                # Document indexing script
├── test_chat.py               # CLI interface
├── requirements.txt
├── .env.example
└── README.md
```

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# LLM Configuration
LLM_PROVIDER = "gemini"  # or "openai"
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0

# Chunk Sizes
CHILD_CHUNK_SIZE = 500
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 10000

# Retrieval
TOP_K_CHILD_CHUNKS = 7
MAX_PARENT_RETRIEVAL = 3
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required: At least one LLM API key
GOOGLE_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key  # Optional alternative
```

## 🎓 How It Works

### 1. Document Indexing
```python
# Process: PDF/MD → Parent Chunks → Child Chunks → Vector DB
docs/ → [Split by headers] → parent_store/ (JSON)
                           → qdrant_db/ (vectors)
```

### 2. Query Processing
```python
# User Query → Analyze → Rewrite → Search → Generate Answer
"Clementi rent?" → "rental prices in Clementi area Singapore"
```

### 3. Retrieval Flow
```python
1. Search 7 child chunks (semantic + keyword)
2. Retrieve 1-3 parent chunks (full context)
3. Generate answer with LLM
4. Cite sources
```

## 📊 Performance

- **First run**: 3-5 minutes (downloads models)
- **Subsequent queries**: 5-15 seconds
- **Index 5 documents**: ~30 seconds
- **Vector DB size**: ~10MB for 200 chunks

## 🔮 Future Enhancements

- [x] Web UI with Gradio
- [ ] Property listing integration (PropertyGuru API)
- [ ] Multi-language support (Chinese, Malay)
- [ ] Image analysis for property photos
- [ ] Recommendation system based on preferences
- [ ] PDF document upload via Web UI

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Based on [agentic-rag-for-dummies](https://github.com/GiovanniPasq/agentic-rag-for-dummies)
- Built with LangChain and LangGraph
- Powered by Google Gemini / OpenAI

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for international students in Singapore**
