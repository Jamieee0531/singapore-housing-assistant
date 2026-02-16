"""
Parent Store Manager for handling parent chunk storage and retrieval.

Based on the original agentic-rag-for-dummies project.
Manages JSON file-based storage of parent chunks.
"""

import logging
import re
import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import List, Dict
from src.config import PARENT_STORE_PATH

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _read_json_file(file_path: str) -> Dict:
    """Read and cache a JSON file. Cached by file path."""
    return json.loads(Path(file_path).read_text(encoding="utf-8"))


class ParentStoreManager:
    """
    Manages storage and retrieval of parent chunks.
    
    Parent chunks are large text segments that contain full context.
    They are stored as JSON files in the parent_store directory.
    """
    
    __store_path: Path
    
    def __init__(self, store_path: str = PARENT_STORE_PATH):
        """
        Initialize the parent store manager.
        
        Args:
            store_path: Directory path for storing parent chunk JSON files
        """
        self.__store_path = Path(store_path)
        self.__store_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, parent_id: str, content: str, metadata: Dict) -> None:
        """
        Save a single parent chunk to disk.
        
        Args:
            parent_id: Unique identifier for the parent chunk
            content: The actual text content of the parent chunk
            metadata: Metadata dict (source file, headers, etc.)
        """
        file_path = self.__store_path / f"{parent_id}.json"
        file_path.write_text(
            json.dumps(
                {"page_content": content, "metadata": metadata},
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )
    
    def save_many(self, parents: List) -> None:
        """
        Save multiple parent chunks to disk.
        
        Args:
            parents: List of tuples (parent_id, document)
                    where document has .page_content and .metadata
        """
        for parent_id, doc in parents:
            self.save(parent_id, doc.page_content, doc.metadata)
    
    def load(self, parent_id: str) -> Dict:
        """
        Load raw parent chunk data from disk (cached).

        Args:
            parent_id: Parent chunk ID (with or without .json extension)

        Returns:
            Dictionary with 'page_content' and 'metadata' keys
        """
        file_path = self.__store_path / (
            parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
        )
        return _read_json_file(str(file_path))
    
    def load_content(self, parent_id: str) -> Dict:
        """
        Load parent chunk with standardized format.
        
        Args:
            parent_id: Parent chunk ID
            
        Returns:
            Dictionary with 'content', 'parent_id', and 'metadata' keys
        """
        data = self.load(parent_id)
        return {
            "content": data["page_content"],
            "parent_id": parent_id,
            "metadata": data["metadata"]
        }
    
    @staticmethod
    def _get_sort_key(id_str: str) -> int:
        """
        Extract numeric index from parent_id for sorting.
        
        Example: "hdb_guide_parent_5" -> 5
        
        Args:
            id_str: Parent chunk ID string
            
        Returns:
            Numeric index for sorting, or 0 if no match
        """
        match = re.search(r'_parent_(\d+)$', id_str)
        return int(match.group(1)) if match else 0
    
    def load_content_many(self, parent_ids: List[str]) -> List[Dict]:
        """
        Load multiple parent chunks with deduplication and sorting.
        
        Args:
            parent_ids: List of parent chunk IDs (may contain duplicates)
            
        Returns:
            List of parent chunk dicts, sorted by numeric index
        """
        unique_ids = set(parent_ids)
        return [
            self.load_content(pid)
            for pid in sorted(unique_ids, key=self._get_sort_key)
        ]
    
    def clear_store(self) -> None:
        """
        Clear all parent chunks from storage and invalidate cache.

        Useful for:
        - Re-indexing documents from scratch
        - Testing and cleanup
        """
        _read_json_file.cache_clear()
        if self.__store_path.exists():
            shutil.rmtree(self.__store_path)
        self.__store_path.mkdir(parents=True, exist_ok=True)