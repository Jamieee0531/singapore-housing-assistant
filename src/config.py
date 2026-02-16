"""
Configuration module for Singapore Housing Assistant RAG System.
Loads environment variables and defines all system configurations.
"""

import logging
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

# Google Maps API Key (for location/commute features)
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

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
CHECKPOINT_DB_PATH = "checkpoints.db"       # SQLite checkpoint database
THREAD_ID_PATH = "thread_id.txt"            # Persisted conversation thread ID

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
LLM_MODEL = "gemini-2.5-flash"  # For Gemini
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

# Google Maps settings
MAPS_SEARCH_RADIUS = 1000          # Search radius in meters
MAPS_MAX_RESULTS = 8               # Max nearby places to return

# Conversation summarization
SUMMARY_MIN_MESSAGES = 4           # Min messages before summarizing
SUMMARY_LAST_N_MESSAGES = 6        # Last N messages to include in summary
SUMMARY_TEMPERATURE = 0.2          # Temperature for summary generation

# =============================================================================
# Tool Error Message Prefixes
# =============================================================================

TOOL_ERROR_PREFIX = "[ERROR]"
TOOL_NO_RESULTS_PREFIX = "[NO_RESULTS]"

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
            "google_api_key": GOOGLE_API_KEY,
            "model": LLM_MODEL,
            "temperature": LLM_TEMPERATURE
        }
    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        return {
            "openai_api_key": OPENAI_API_KEY,
            "model": LLM_MODEL,
            "temperature": LLM_TEMPERATURE
        }
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")

def setup_logging():
    """Configure logging based on LOG_LEVEL setting."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def print_config():
    """Print current configuration (for debugging)."""
    logger = logging.getLogger(__name__)
    logger.info("Singapore Housing Assistant - Configuration")
    logger.info("LLM Provider: %s", LLM_PROVIDER)
    logger.info("LLM Model: %s", LLM_MODEL)
    logger.info("Docs Directory: %s", DOCS_DIR)
    logger.info("Qdrant DB Path: %s", QDRANT_DB_PATH)
    logger.info("Child Collection: %s", CHILD_COLLECTION)
    logger.info("Dense Model: %s", DENSE_MODEL)
    logger.info("Child Chunk Size: %s", CHILD_CHUNK_SIZE)
    logger.info("Parent Size Range: %s-%s", MIN_PARENT_SIZE, MAX_PARENT_SIZE)

# =============================================================================
# Auto-create directories on import
# =============================================================================

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)
os.makedirs(PARENT_STORE_PATH, exist_ok=True)
os.makedirs(QDRANT_DB_PATH, exist_ok=True)