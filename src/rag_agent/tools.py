"""
Retrieval tools for Singapore Housing Assistant RAG System.
Provides two-stage retrieval: child chunks (search) and parent chunks (context).

Uses ToolFactory pattern for clean dependency injection.
"""

import logging
from typing import List
from langchain_core.tools import tool
from src.db import ParentStoreManager
from src.rag_agent.base import BaseToolFactory, timed_tool
from src.config import (
    TOP_K_CHILD_CHUNKS,
    SIMILARITY_THRESHOLD,
    TOOL_ERROR_PREFIX,
    TOOL_NO_RESULTS_PREFIX,
)

logger = logging.getLogger(__name__)


class ToolFactory(BaseToolFactory):
    """
    Factory for creating RAG retrieval tools.
    Encapsulates dependencies (vector store, parent store manager).
    """

    def __init__(self, collection):
        """
        Initialize the tool factory.

        Args:
            collection: QdrantVectorStore instance for child chunk retrieval
        """
        self.collection = collection
        self.parent_store_manager = ParentStoreManager()

    def _search_child_chunks(self, query: str, limit: int = TOP_K_CHILD_CHUNKS) -> str:
        """
        Search for the top K most relevant child chunks.

        This performs hybrid search (dense + sparse embeddings) to find
        small, relevant text chunks that match the user's query.

        Args:
            query: Search query string (e.g., "HDB rental prices Clementi")
            limit: Maximum number of results to return (default: 7)

        Returns:
            Formatted string containing:
            - Parent ID (for retrieving full context)
            - File name (source document)
            - Content (the actual text chunk)

            On failure:
            - "[NO_RESULTS] ..." if no relevant chunks found
            - "[ERROR] ..." if search fails
        """
        try:
            results = self.collection.similarity_search(
                query,
                k=limit,
                score_threshold=SIMILARITY_THRESHOLD
            )

            if not results:
                logger.warning("No results for query: '%s' (threshold=%.1f)", query, SIMILARITY_THRESHOLD)
                return (
                    f"{TOOL_NO_RESULTS_PREFIX} No relevant chunks found for this query. "
                    "Try rephrasing or broadening your search."
                )

            formatted_results = []
            for doc in results:
                formatted_results.append(
                    f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                    f"File Name: {doc.metadata.get('source', '')}\n"
                    f"Content: {doc.page_content.strip()}"
                )

            return "\n\n---\n\n".join(formatted_results)

        except Exception as e:
            logger.error("Search failed: %s", e, exc_info=True)
            return f"{TOOL_ERROR_PREFIX} Search failed: {e}"

    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """
        Retrieve a single full parent chunk by ID.

        When a child chunk is too fragmented or lacks context, this tool
        retrieves the full parent chunk that contains it.

        Args:
            parent_id: Parent chunk ID (e.g., "hdb_guide_parent_0")

        Returns:
            Formatted string containing:
            - Parent ID
            - File name (source document)
            - Content (the complete parent chunk with full context)

            On failure:
            - "[NO_RESULTS] ..." if parent chunk not found
            - "[ERROR] ..." if retrieval fails
        """
        try:
            parent = self.parent_store_manager.load_content(parent_id)

            if not parent:
                logger.warning("Parent chunk not found: '%s'", parent_id)
                return f"{TOOL_NO_RESULTS_PREFIX} Parent chunk '{parent_id}' not found."

            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )

        except Exception as e:
            logger.error("Parent chunk retrieval failed: %s", e, exc_info=True)
            return f"{TOOL_ERROR_PREFIX} Parent chunk retrieval failed: {e}"

    def _retrieve_many_parent_chunks(self, parent_ids: List[str]) -> str:
        """
        Retrieve multiple full parent chunks by their IDs.

        Batch version of retrieve_parent_chunks for efficiency.

        Args:
            parent_ids: List of parent chunk IDs or single ID string

        Returns:
            Formatted string containing all parent chunks.

            On failure:
            - "[NO_RESULTS] ..." if no parent chunks found
            - "[ERROR] ..." if retrieval fails
        """
        try:
            ids = [parent_ids] if isinstance(parent_ids, str) else list(parent_ids)

            raw_parents = self.parent_store_manager.load_content_many(ids)

            if not raw_parents:
                logger.warning("No parent chunks found for IDs: %s", ids)
                return f"{TOOL_NO_RESULTS_PREFIX} No parent chunks found for the given IDs."

            formatted_results = []
            for doc in raw_parents:
                formatted_results.append(
                    f"Parent ID: {doc.get('parent_id', 'n/a')}\n"
                    f"File Name: {doc.get('metadata', {}).get('source', 'unknown')}\n"
                    f"Content: {doc.get('content', '').strip()}"
                )

            return "\n\n---\n\n".join(formatted_results)

        except Exception as e:
            logger.error("Parent chunk batch retrieval failed: %s", e, exc_info=True)
            return f"{TOOL_ERROR_PREFIX} Parent chunk retrieval failed: {e}"

    def create_tools(self) -> List:
        """
        Create and return the list of tools for LangChain.

        Wraps the internal methods with @tool decorator.

        Returns:
            List of LangChain tools ready to be bound to LLM
        """
        search_tool = tool("search_child_chunks")(timed_tool(self._search_child_chunks))
        retrieve_tool = tool("retrieve_parent_chunks")(timed_tool(self._retrieve_parent_chunks))

        return [search_tool, retrieve_tool]
