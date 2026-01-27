"""
Configuration module for Singapore Housing Assistant RAG System.
Loads environment variables and defines all system configurations.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# API Keys Configuration
# =============================================================================

# LLM Provider API Keys (you can use either)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validation: Ensure at least one LLM API key is provided
if not GOOGLE_API_KEY and not OPENAI_API_KEY:
    raise ValueError(
        "Please set at least one LLM API key in .env file:\n"
        "- GOOGLE_API_KEY (for Gemini)\n"
        "- OPENAI_API_KEY (for GPT)"
    )

# =============================================================================
# Directory Configuration
# =============================================================================

DOCS_DIR = "docs"                           # Original PDF/MD documents
MARKDOWN_DIR = "markdown"                    # PDF converted to markdown
PARENT_STORE_PATH = "parent_store"          # Parent chunk JSON storage
QDRANT_DB_PATH = "qdrant_db"                # Qdrant local database

# =============================================================================
# Qdrant Vector Database Configuration
# =============================================================================

CHILD_COLLECTION = "document_child_chunks"  # Collection name for child chunks
SPARSE_VECTOR_NAME = "sparse"               # Sparse vector field name

# =============================================================================
# Model Configuration
# =============================================================================

# Embedding Models
DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 768 dimensions
SPARSE_MODEL = "Qdrant/bm25"                              # BM25 sparse embeddings

# LLM Selection (choose which one to use)
# Option 1: Google Gemini (default)
LLM_PROVIDER = "gemini"  # Options: "gemini" or "openai"
LLM_MODEL = "gemini-2.0-flash-exp"  # For Gemini
# LLM_MODEL = "gpt-4-turbo-preview"  # For OpenAI (alternative)

LLM_TEMPERATURE = 0  # 0 for deterministic, 0.7 for creative

# =============================================================================
# Text Splitter Configuration
# =============================================================================

# Child chunks (small, for vector search)
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100

# Parent chunks (large, for context)
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 10000

# Markdown header splitting
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# =============================================================================
# RAG Configuration
# =============================================================================

# Retrieval settings
TOP_K_CHILD_CHUNKS = 7          # Number of child chunks to retrieve
SIMILARITY_THRESHOLD = 0.7      # Minimum similarity score (0-1)
MAX_PARENT_RETRIEVAL = 3        # Max parent chunks to fetch per query

# Query analysis
QUERY_ANALYSIS_TEMPERATURE = 0.1  # Low temp for query rewriting

# =============================================================================
# System Settings
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# =============================================================================
# Helper Functions
# =============================================================================

def get_llm_config():
    """Returns LLM configuration based on selected provider."""
    if LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set in .env file")
        return {
            "provider": "gemini",
            "api_key": GOOGLE_API_KEY,
            "model": LLM_MODEL,
            "temperature": LLM_TEMPERATURE
        }
    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        return {
            "provider": "openai",
            "api_key": OPENAI_API_KEY,
            "model": LLM_MODEL,
            "temperature": LLM_TEMPERATURE
        }
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")

def print_config():
    """Print current configuration (for debugging)."""
    print("=" * 60)
    print("🏠 Singapore Housing Assistant - Configuration")
    print("=" * 60)
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Docs Directory: {DOCS_DIR}")
    print(f"Qdrant DB Path: {QDRANT_DB_PATH}")
    print(f"Child Collection: {CHILD_COLLECTION}")
    print(f"Dense Model: {DENSE_MODEL}")
    print(f"Child Chunk Size: {CHILD_CHUNK_SIZE}")
    print(f"Parent Size Range: {MIN_PARENT_SIZE}-{MAX_PARENT_SIZE}")
    print("=" * 60)

# =============================================================================
# Auto-create directories on import
# =============================================================================

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)
os.makedirs(PARENT_STORE_PATH, exist_ok=True)
os.makedirs(QDRANT_DB_PATH, exist_ok=True)