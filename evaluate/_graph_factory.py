"""
Graph factory for evaluation.

Creates a fresh graph instance for running evaluation,
isolated from the Gradio UI session.
"""

import uuid

from src.config import (
    get_llm_config,
    LLM_PROVIDER,
    QDRANT_DB_PATH,
    CHILD_COLLECTION,
    DENSE_MODEL,
    SPARSE_MODEL,
    GOOGLE_MAPS_API_KEY,
)
from src.rag_agent.tools import ToolFactory
from src.rag_agent.maps_tools import MapsToolFactory
from src.rag_agent.graph import create_agent_graph

if LLM_PROVIDER == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI as LLMClass
else:
    from langchain_openai import ChatOpenAI as LLMClass


def create_eval_graph():
    """
    Create a fresh graph + config for evaluation.

    Returns:
        (graph, config) tuple
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant.fastembed_sparse import FastEmbedSparse
    from langchain_qdrant import QdrantVectorStore, RetrievalMode
    from qdrant_client import QdrantClient

    llm = LLMClass(**get_llm_config())

    dense_embeddings = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL)
    client = QdrantClient(path=QDRANT_DB_PATH)

    child_vector_store = QdrantVectorStore(
        client=client,
        collection_name=CHILD_COLLECTION,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="sparse"
    )

    tool_factory = ToolFactory(collection=child_vector_store)
    tools = tool_factory.create_tools()

    if GOOGLE_MAPS_API_KEY:
        try:
            maps_factory = MapsToolFactory(GOOGLE_MAPS_API_KEY)
            tools.extend(maps_factory.create_tools())
        except Exception:
            pass

    graph = create_agent_graph(llm, tools)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    return graph, config
