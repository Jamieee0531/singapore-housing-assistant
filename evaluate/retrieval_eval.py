"""
Retrieval evaluation for Singapore Housing Assistant.

Evaluates search quality using Precision@k and MRR.
Directly calls search_child_chunks and compares results
against labeled relevant sources in the dataset.
"""

import json
import logging
from pathlib import Path
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from src.config import (
    QDRANT_DB_PATH,
    CHILD_COLLECTION,
    DENSE_MODEL,
    SPARSE_MODEL,
    TOP_K_CHILD_CHUNKS,
    RETRIEVAL_CANDIDATES,
    SIMILARITY_THRESHOLD,
    RERANK_ENABLED,
)
from src.rag_agent.tools import ToolFactory

logger = logging.getLogger(__name__)


def _extract_sources_from_results(result_str: str) -> List[str]:
    """
    Parse the formatted tool output to extract source file names.

    Args:
        result_str: Formatted string from search_child_chunks

    Returns:
        List of source file names found in results
    """
    sources = []
    for line in result_str.split("\n"):
        if line.startswith("File Name:"):
            source = line.replace("File Name:", "").strip()
            if source and source not in sources:
                sources.append(source)
    return sources


def precision_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int) -> float:
    """
    Calculate Precision@k.

    What fraction of the top-k retrieved sources are actually relevant?

    Args:
        retrieved_sources: Sources returned by search (ordered by relevance)
        relevant_sources: Ground truth relevant sources from dataset
        k: Number of top results to consider

    Returns:
        Precision score between 0.0 and 1.0
    """
    top_k = retrieved_sources[:k]
    if not top_k:
        return 0.0

    relevant_count = sum(1 for s in top_k if s in relevant_sources)
    return relevant_count / len(top_k)


def reciprocal_rank(retrieved_sources: List[str], relevant_sources: List[str]) -> float:
    """
    Calculate Reciprocal Rank.

    How high is the first relevant document ranked?
    RR = 1/rank of first relevant result.

    Args:
        retrieved_sources: Sources returned by search (ordered by relevance)
        relevant_sources: Ground truth relevant sources from dataset

    Returns:
        Reciprocal rank between 0.0 and 1.0 (0.0 if no relevant doc found)
    """
    for i, source in enumerate(retrieved_sources):
        if source in relevant_sources:
            return 1.0 / (i + 1)
    return 0.0


def run_retrieval_eval(dataset_path: str = "evaluate/dataset.json") -> dict:
    """
    Run retrieval evaluation on the full dataset.

    For each question in the dataset:
    1. Call search_child_chunks
    2. Extract sources from results
    3. Compare against labeled relevant sources
    4. Calculate Precision@k and MRR

    Args:
        dataset_path: Path to the evaluation dataset JSON

    Returns:
        Dict with per-question results and aggregate metrics
    """
    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Initialize retrieval components
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

    # Evaluate each question
    results = []
    k = TOP_K_CHILD_CHUNKS

    for item in dataset:
        question = item["question"]
        relevant_sources = item["relevant_sources"]

        # Run search
        search_result = tool_factory._search_child_chunks(question)
        retrieved_sources = _extract_sources_from_results(search_result)

        # Calculate metrics
        p_at_k = precision_at_k(retrieved_sources, relevant_sources, k)
        rr = reciprocal_rank(retrieved_sources, relevant_sources)

        result = {
            "id": item["id"],
            "question": question,
            "category": item.get("category", "unknown"),
            "relevant_sources": relevant_sources,
            "retrieved_sources": retrieved_sources,
            "precision_at_k": round(p_at_k, 3),
            "reciprocal_rank": round(rr, 3),
        }
        results.append(result)

        logger.info(
            "[%s] P@%d=%.3f  RR=%.3f  retrieved=%s",
            item["id"], k, p_at_k, rr, retrieved_sources
        )

    # Aggregate metrics
    avg_precision = sum(r["precision_at_k"] for r in results) / len(results) if results else 0
    mrr = sum(r["reciprocal_rank"] for r in results) / len(results) if results else 0

    summary = {
        "config": {
            "top_k": k,
            "retrieval_candidates": RETRIEVAL_CANDIDATES,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "rerank_enabled": RERANK_ENABLED,
        },
        "aggregate": {
            "mean_precision_at_k": round(avg_precision, 3),
            "mrr": round(mrr, 3),
            "total_questions": len(results),
        },
        "per_question": results,
    }

    return summary
