"""
Retrieval tools for Singapore Housing Assistant RAG System.
Provides two-stage retrieval: child chunks (search) and parent chunks (context).

Uses ToolFactory pattern for clean dependency injection.
"""

from typing import List
from langchain_core.tools import tool
from src.db import ParentStoreManager


class ToolFactory:
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
    
    def _search_child_chunks(self, query: str, limit: int = 7) -> str:
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
            
            Or special strings:
            - "NO_RELEVANT_CHUNKS" if no results found
            - "RETRIEVAL_ERROR: ..." if search fails
        """
        try:
            # Perform hybrid search with similarity threshold
            results = self.collection.similarity_search(
                query, 
                k=limit, 
                score_threshold=0.7
            )
            
            if not results:
                return "NO_RELEVANT_CHUNKS"
            
            # Format results for LLM consumption
            formatted_results = []
            for doc in results:
                formatted_results.append(
                    f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                    f"File Name: {doc.metadata.get('source', '')}\n"
                    f"Content: {doc.page_content.strip()}"
                )
            
            return "\n\n---\n\n".join(formatted_results)
        
        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"
    
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
            
            Or "NO_PARENT_DOCUMENT" if not found.
        """
        try:
            parent = self.parent_store_manager.load_content(parent_id)
            
            if not parent:
                return f"NO_PARENT_DOCUMENT: Could not find parent chunk '{parent_id}'"
            
            # Format for LLM consumption
            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )
        
        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_many_parent_chunks(self, parent_ids: List[str]) -> str:
        """
        Retrieve multiple full parent chunks by their IDs.
        
        Batch version of retrieve_parent_chunks for efficiency.
        
        Args:
            parent_ids: List of parent chunk IDs or single ID string
            
        Returns:
            Formatted string containing all parent chunks
            Or "NO_PARENT_DOCUMENTS" if none found.
        """
        try:
            # Handle both single string and list input
            ids = [parent_ids] if isinstance(parent_ids, str) else list(parent_ids)
            
            # Load all parent chunks
            raw_parents = self.parent_store_manager.load_content_many(ids)
            
            if not raw_parents:
                return "NO_PARENT_DOCUMENTS"
            
            # Format for LLM consumption
            formatted_results = []
            for doc in raw_parents:
                formatted_results.append(
                    f"Parent ID: {doc.get('parent_id', 'n/a')}\n"
                    f"File Name: {doc.get('metadata', {}).get('source', 'unknown')}\n"
                    f"Content: {doc.get('content', '').strip()}"
                )
            
            return "\n\n---\n\n".join(formatted_results)
        
        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def create_tools(self) -> List:
        """
        Create and return the list of tools for LangChain.
        
        Wraps the internal methods with @tool decorator.
        
        Returns:
            List of LangChain tools ready to be bound to LLM
        """
        # Wrap methods with @tool decorator
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        
        return [search_tool, retrieve_tool]