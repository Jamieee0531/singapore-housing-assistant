"""
Response evaluation for Singapore Housing Assistant.

Uses LLM-as-Judge to evaluate:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevance: Does the answer actually address the question?

Runs the full graph end-to-end, then asks a judge LLM to score the output.
"""

import json
import logging
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from src.config import get_llm_config, LLM_PROVIDER

logger = logging.getLogger(__name__)

# Import LLM based on provider
if LLM_PROVIDER == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI as LLMClass
else:
    from langchain_openai import ChatOpenAI as LLMClass


class JudgeScore(BaseModel):
    """Structured output for LLM judge scoring."""
    faithfulness_score: int = Field(
        description="1-5 score. How well is the answer grounded in the context? "
        "5 = fully grounded, 1 = mostly hallucinated"
    )
    faithfulness_reason: str = Field(
        description="Brief explanation for the faithfulness score"
    )
    relevance_score: int = Field(
        description="1-5 score. How well does the answer address the original question? "
        "5 = fully answers the question, 1 = completely off-topic"
    )
    relevance_reason: str = Field(
        description="Brief explanation for the relevance score"
    )


JUDGE_PROMPT = """You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.
You will be given a question, the retrieved context, and the system's answer.

Evaluate the answer on two dimensions:

1. **Faithfulness** (1-5): Is the answer grounded in the provided context?
   - 5: Every claim in the answer can be traced back to the context
   - 4: Almost all claims are supported, minor unsupported details
   - 3: Some claims are supported, some are not
   - 2: Most claims cannot be found in the context
   - 1: The answer appears fabricated / hallucinated

2. **Answer Relevance** (1-5): Does the answer address the original question?
   - 5: Directly and completely answers the question
   - 4: Mostly answers the question, minor gaps
   - 3: Partially answers the question
   - 2: Tangentially related but doesn't answer the question
   - 1: Completely off-topic

Be strict but fair. Provide brief reasoning for each score."""


def judge_response(
    question: str,
    context: str,
    answer: str,
    judge_llm
) -> JudgeScore:
    """
    Use LLM-as-Judge to score a single response.

    Args:
        question: Original user question
        context: Retrieved context that was used to generate the answer
        answer: The system's generated answer
        judge_llm: LLM instance configured for structured output

    Returns:
        JudgeScore with faithfulness and relevance scores + reasoning
    """
    eval_input = (
        f"## Question\n{question}\n\n"
        f"## Retrieved Context\n{context}\n\n"
        f"## System Answer\n{answer}"
    )

    result = judge_llm.invoke([
        SystemMessage(content=JUDGE_PROMPT),
        HumanMessage(content=eval_input)
    ])

    return result


def run_response_eval(
    dataset_path: str = "evaluate/dataset.json",
    graph=None,
    graph_config=None,
) -> dict:
    """
    Run response evaluation on the full dataset.

    For each question:
    1. Run the full graph to get the system's answer
    2. Use LLM-as-Judge to score faithfulness and relevance
    3. Aggregate results

    Args:
        dataset_path: Path to the evaluation dataset JSON
        graph: Compiled LangGraph instance (if None, will be created)
        graph_config: Config dict with thread_id for the graph

    Returns:
        Dict with per-question scores and aggregate metrics
    """
    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Initialize judge LLM with structured output
    llm_config = get_llm_config()
    judge_llm = LLMClass(**llm_config).with_structured_output(JudgeScore)

    # If no graph provided, create one
    if graph is None:
        from evaluate._graph_factory import create_eval_graph
        graph, graph_config = create_eval_graph()

    results = []

    for item in dataset:
        question = item["question"]
        language = item.get("language", "en")

        try:
            # Run the full graph
            graph_result = graph.invoke(
                {
                    "messages": [HumanMessage(content=question)],
                    "language": language,
                },
                graph_config
            )

            # Extract the answer (last AI message)
            answer = ""
            for msg in reversed(graph_result["messages"]):
                if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                    answer = msg.content
                    break

            # Extract context from tool messages
            context_parts = []
            for msg in graph_result["messages"]:
                if msg.type == "tool" and msg.content:
                    context_parts.append(msg.content)
            context = "\n---\n".join(context_parts) if context_parts else "No context retrieved"

            # Judge the response
            score = judge_response(question, context, answer, judge_llm)

            result = {
                "id": item["id"],
                "question": question,
                "category": item.get("category", "unknown"),
                "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                "faithfulness_score": score.faithfulness_score,
                "faithfulness_reason": score.faithfulness_reason,
                "relevance_score": score.relevance_score,
                "relevance_reason": score.relevance_reason,
            }
            results.append(result)

            logger.info(
                "[%s] Faithfulness=%d  Relevance=%d",
                item["id"], score.faithfulness_score, score.relevance_score
            )

            # Reset thread for next question (clean state)
            import uuid
            graph_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        except Exception as e:
            logger.error("Failed to evaluate %s: %s", item["id"], e, exc_info=True)
            results.append({
                "id": item["id"],
                "question": question,
                "category": item.get("category", "unknown"),
                "error": str(e),
                "faithfulness_score": 0,
                "relevance_score": 0,
            })

    # Aggregate
    valid_results = [r for r in results if "error" not in r]
    avg_faithfulness = (
        sum(r["faithfulness_score"] for r in valid_results) / len(valid_results)
        if valid_results else 0
    )
    avg_relevance = (
        sum(r["relevance_score"] for r in valid_results) / len(valid_results)
        if valid_results else 0
    )

    summary = {
        "aggregate": {
            "mean_faithfulness": round(avg_faithfulness, 2),
            "mean_relevance": round(avg_relevance, 2),
            "total_questions": len(results),
            "errors": len(results) - len(valid_results),
        },
        "per_question": results,
    }

    return summary
