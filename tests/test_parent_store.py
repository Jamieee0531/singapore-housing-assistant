"""Tests for ParentStoreManager."""

import json
import pytest
from pathlib import Path
from src.db.parent_store_manager import ParentStoreManager, _read_json_file


@pytest.fixture
def tmp_store(tmp_path):
    """Create a ParentStoreManager with a temporary directory."""
    manager = ParentStoreManager(store_path=str(tmp_path))
    return manager, tmp_path


class TestSaveAndLoad:
    """Tests for saving and loading parent chunks."""

    def test_save_creates_json_file(self, tmp_store):
        manager, tmp_path = tmp_store
        manager.save("chunk_parent_0", "Hello world", {"source": "test.md"})
        file_path = tmp_path / "chunk_parent_0.json"
        assert file_path.exists()

    def test_load_returns_saved_content(self, tmp_store):
        manager, _ = tmp_store
        manager.save("chunk_parent_0", "Hello world", {"source": "test.md"})
        # Clear cache to test fresh load
        _read_json_file.cache_clear()
        data = manager.load("chunk_parent_0")
        assert data["page_content"] == "Hello world"
        assert data["metadata"]["source"] == "test.md"

    def test_load_content_standardizes_format(self, tmp_store):
        manager, _ = tmp_store
        manager.save("chunk_parent_0", "Content here", {"source": "doc.md"})
        _read_json_file.cache_clear()
        result = manager.load_content("chunk_parent_0")
        assert result["content"] == "Content here"
        assert result["parent_id"] == "chunk_parent_0"
        assert result["metadata"]["source"] == "doc.md"

    def test_load_with_json_extension(self, tmp_store):
        manager, _ = tmp_store
        manager.save("chunk_parent_0", "Test", {"source": "x.md"})
        _read_json_file.cache_clear()
        data = manager.load("chunk_parent_0.json")
        assert data["page_content"] == "Test"

    def test_load_nonexistent_raises(self, tmp_store):
        manager, _ = tmp_store
        with pytest.raises(FileNotFoundError):
            _read_json_file.cache_clear()
            manager.load("nonexistent_id")


class TestLoadMany:
    """Tests for batch loading parent chunks."""

    def test_load_content_many(self, tmp_store):
        manager, _ = tmp_store
        manager.save("doc_parent_0", "First", {"source": "a.md"})
        manager.save("doc_parent_1", "Second", {"source": "b.md"})
        _read_json_file.cache_clear()
        results = manager.load_content_many(["doc_parent_0", "doc_parent_1"])
        assert len(results) == 2
        assert results[0]["content"] == "First"
        assert results[1]["content"] == "Second"

    def test_deduplicates_ids(self, tmp_store):
        manager, _ = tmp_store
        manager.save("doc_parent_0", "Only one", {"source": "a.md"})
        _read_json_file.cache_clear()
        results = manager.load_content_many(["doc_parent_0", "doc_parent_0", "doc_parent_0"])
        assert len(results) == 1

    def test_sorts_by_numeric_index(self, tmp_store):
        manager, _ = tmp_store
        manager.save("doc_parent_2", "Third", {"source": "c.md"})
        manager.save("doc_parent_0", "First", {"source": "a.md"})
        manager.save("doc_parent_1", "Second", {"source": "b.md"})
        _read_json_file.cache_clear()
        results = manager.load_content_many(["doc_parent_2", "doc_parent_0", "doc_parent_1"])
        assert results[0]["parent_id"] == "doc_parent_0"
        assert results[1]["parent_id"] == "doc_parent_1"
        assert results[2]["parent_id"] == "doc_parent_2"


class TestSortKey:
    """Tests for _get_sort_key static method."""

    def test_extracts_numeric_index(self):
        assert ParentStoreManager._get_sort_key("doc_parent_5") == 5

    def test_returns_zero_for_no_match(self):
        assert ParentStoreManager._get_sort_key("no_number_here") == 0

    def test_handles_large_numbers(self):
        assert ParentStoreManager._get_sort_key("doc_parent_123") == 123


class TestClearStore:
    """Tests for clearing the store."""

    def test_clear_removes_all_files(self, tmp_store):
        manager, tmp_path = tmp_store
        manager.save("chunk_parent_0", "Data", {"source": "x.md"})
        manager.save("chunk_parent_1", "Data2", {"source": "y.md"})
        assert len(list(tmp_path.glob("*.json"))) == 2
        manager.clear_store()
        assert len(list(tmp_path.glob("*.json"))) == 0

    def test_clear_recreates_directory(self, tmp_store):
        manager, tmp_path = tmp_store
        manager.clear_store()
        assert tmp_path.exists()


class TestCache:
    """Tests for LRU cache behavior."""

    def test_cached_reads(self, tmp_store):
        manager, tmp_path = tmp_store
        manager.save("cache_test_parent_0", "Cached", {"source": "c.md"})
        _read_json_file.cache_clear()

        # First load — cache miss
        result1 = manager.load("cache_test_parent_0")
        # Second load — should be from cache
        result2 = manager.load("cache_test_parent_0")
        assert result1 == result2

        # Check cache stats
        info = _read_json_file.cache_info()
        assert info.hits >= 1

    def test_clear_store_invalidates_cache(self, tmp_store):
        manager, _ = tmp_store
        manager.save("cache_test_parent_0", "Data", {"source": "x.md"})
        _read_json_file.cache_clear()
        manager.load("cache_test_parent_0")
        manager.clear_store()
        info = _read_json_file.cache_info()
        assert info.currsize == 0
