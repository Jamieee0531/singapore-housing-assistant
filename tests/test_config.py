"""Tests for configuration module."""

import pytest
from src.config import (
    TOOL_ERROR_PREFIX,
    TOOL_NO_RESULTS_PREFIX,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    MIN_PARENT_SIZE,
    MAX_PARENT_SIZE,
    TOP_K_CHILD_CHUNKS,
    SIMILARITY_THRESHOLD,
    CHECKPOINT_DB_PATH,
    THREAD_ID_PATH,
    get_llm_config,
)


class TestErrorConstants:
    """Tests for tool error message constants."""

    def test_error_prefix_format(self):
        assert TOOL_ERROR_PREFIX == "[ERROR]"

    def test_no_results_prefix_format(self):
        assert TOOL_NO_RESULTS_PREFIX == "[NO_RESULTS]"

    def test_prefixes_are_bracketed(self):
        assert TOOL_ERROR_PREFIX.startswith("[")
        assert TOOL_ERROR_PREFIX.endswith("]")
        assert TOOL_NO_RESULTS_PREFIX.startswith("[")
        assert TOOL_NO_RESULTS_PREFIX.endswith("]")


class TestChunkingConfig:
    """Tests for chunking configuration values."""

    def test_child_chunk_size_positive(self):
        assert CHILD_CHUNK_SIZE > 0

    def test_overlap_less_than_chunk_size(self):
        assert CHILD_CHUNK_OVERLAP < CHILD_CHUNK_SIZE

    def test_parent_size_range_valid(self):
        assert MIN_PARENT_SIZE < MAX_PARENT_SIZE

    def test_parent_larger_than_child(self):
        assert MIN_PARENT_SIZE > CHILD_CHUNK_SIZE


class TestRetrievalConfig:
    """Tests for retrieval configuration values."""

    def test_top_k_positive(self):
        assert TOP_K_CHILD_CHUNKS > 0

    def test_similarity_threshold_in_range(self):
        assert 0 <= SIMILARITY_THRESHOLD <= 1


class TestPathConfig:
    """Tests for path configuration values."""

    def test_checkpoint_db_path_set(self):
        assert CHECKPOINT_DB_PATH.endswith(".db")

    def test_thread_id_path_set(self):
        assert THREAD_ID_PATH.endswith(".txt")


class TestGetLlmConfig:
    """Tests for get_llm_config function."""

    def test_returns_dict_with_model(self):
        config = get_llm_config()
        assert "model" in config
        assert "temperature" in config
