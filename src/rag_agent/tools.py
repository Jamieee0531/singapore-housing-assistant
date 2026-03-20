"""
Retrieval tools for Singapore Housing Assistant RAG System.
Provides two-stage retrieval: child chunks (search) and parent chunks (context).

Uses ToolFactory pattern for clean dependency injection.
"""

import logging
from typing import List, Optional
from langchain_core.tools import tool
from sentence_transformers import CrossEncoder
from src.db import ParentStoreManager
from src.db.redis_manager import RedisManager
from src.rag_agent.base import BaseToolFactory, timed_tool
from qdrant_client.http import models as qmodels
from src.config import (
    TOP_K_CHILD_CHUNKS,
    RETRIEVAL_CANDIDATES,
    SIMILARITY_THRESHOLD,
    RERANK_MODEL,
    RERANK_ENABLED,
    ALL_TOPICS,
    TOOL_ERROR_PREFIX,
    TOOL_NO_RESULTS_PREFIX,
)


def _build_topic_filter(topics: List[str]) -> qmodels.Filter | None:
    """
    Build a Qdrant metadata filter from topic labels.

    Uses OR logic: matches chunks that have ANY of the given topics.
    Invalid topics are filtered out. Returns None if no valid topics.

    Args:
        topics: List of topic labels to filter by

    Returns:
        Qdrant Filter object, or None if no valid topics
    """
    valid_topics = [t for t in topics if t in ALL_TOPICS]
    if not valid_topics:
        return None

    return qmodels.Filter(
        should=[
            qmodels.FieldCondition(
                key="metadata.topics",
                match=qmodels.MatchAny(any=valid_topics)
            )
        ]
    )

logger = logging.getLogger(__name__)


class ToolFactory(BaseToolFactory):
    """
    Factory for creating RAG retrieval tools.
    Encapsulates dependencies (vector store, parent store manager).
    """

    def __init__(self, collection, redis_manager: RedisManager = None):
        """
        Initialize the tool factory.

        Args:
            collection: QdrantVectorStore instance for child chunk retrieval
            redis_manager: RedisManager instance for user memory tools
        """
        self.collection = collection
        self.parent_store_manager = ParentStoreManager()
        self.redis_manager = redis_manager or RedisManager()
        self._reranker: Optional[CrossEncoder] = None

    @property
    def reranker(self) -> Optional[CrossEncoder]:
        """Lazy-load cross-encoder model only when first needed."""
        if RERANK_ENABLED and self._reranker is None:
            logger.info("Loading rerank model: %s", RERANK_MODEL)
            self._reranker = CrossEncoder(RERANK_MODEL)
            logger.info("Rerank model loaded")
        return self._reranker

    def _rerank_results(self, query: str, results: list, top_k: int) -> list:
        """
        Re-rank search results using cross-encoder.

        Cross-encoder takes (query, document) pairs and scores them
        more accurately than bi-encoder similarity search.

        Args:
            query: Original search query
            results: List of Document objects from initial retrieval
            top_k: Number of top results to return after reranking

        Returns:
            Reranked list of Document objects, sorted by relevance
        """
        if not self.reranker or len(results) <= 1:
            return results[:top_k]

        pairs = [[query, doc.page_content] for doc in results]
        scores = self.reranker.predict(pairs)

        scored_results = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True
        )

        reranked = [doc for doc, score in scored_results[:top_k]]
        logger.info(
            "Reranked %d candidates → top %d (scores: %.3f to %.3f)",
            len(results), len(reranked),
            scored_results[0][1], scored_results[min(top_k - 1, len(scored_results) - 1)][1]
        )
        return reranked

    def _search_child_chunks(self, query: str, topics: List[str] = None, limit: int = TOP_K_CHILD_CHUNKS) -> str:
        """
        Search for the top K most relevant child chunks.

        Performs hybrid search (dense + sparse) with optional metadata filtering
        by topic, then optionally re-ranks results using a cross-encoder.

        Args:
            query: Search query string (e.g., "HDB rental prices Clementi")
            topics: Optional list of topic labels to filter by (e.g., ["transport", "pricing"])
            limit: Maximum number of results to return (default: 5)

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
            # Fetch more candidates if reranking is enabled
            fetch_k = RETRIEVAL_CANDIDATES if RERANK_ENABLED else limit

            # Build metadata filter from topics
            search_kwargs = {
                "k": fetch_k,
                "score_threshold": SIMILARITY_THRESHOLD,
            }
            topic_filter = _build_topic_filter(topics or [])
            if topic_filter:
                search_kwargs["filter"] = topic_filter
                logger.info("Filtering by topics: %s", topics)

            results = self.collection.similarity_search(
                query,
                **search_kwargs
            )

            if not results:
                logger.warning("No results for query: '%s' (threshold=%.1f)", query, SIMILARITY_THRESHOLD)
                return (
                    f"{TOOL_NO_RESULTS_PREFIX} No relevant chunks found for this query. "
                    "Try rephrasing or broadening your search."
                )

            # Re-rank results using cross-encoder
            if RERANK_ENABLED:
                results = self._rerank_results(query, results, limit)

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

    def _get_area_history(self, user_id: str = "default") -> str:
        """
        Get areas the user has previously explored or asked about.

        Use this tool when the user asks about their past searches,
        wants to compare areas they've looked at, or references
        previous locations they've considered.

        Args:
            user_id: User identifier (default: "default")

        Returns:
            List of previously explored areas (newest first),
            or a message if no history exists.
        """
        try:
            areas = self.redis_manager.get_area_history(user_id)
            if not areas:
                return f"{TOOL_NO_RESULTS_PREFIX} No area exploration history found for this user."

            formatted = "Previously explored areas (newest first):\n"
            for i, area in enumerate(areas, 1):
                formatted += f"{i}. {area}\n"
            return formatted

        except Exception as e:
            logger.error("Area history retrieval failed: %s", e, exc_info=True)
            return f"{TOOL_ERROR_PREFIX} Area history retrieval failed: {e}"

    def create_tools(self) -> List:
        """
        Create and return the list of tools for LangChain.

        Wraps the internal methods with @tool decorator.

        Returns:
            List of LangChain tools ready to be bound to LLM
        """
        search_tool = tool("search_child_chunks")(timed_tool(self._search_child_chunks))
        retrieve_tool = tool("retrieve_parent_chunks")(timed_tool(self._retrieve_parent_chunks))
        area_history_tool = tool("get_area_history")(timed_tool(self._get_area_history))

        return [search_tool, retrieve_tool, area_history_tool]
