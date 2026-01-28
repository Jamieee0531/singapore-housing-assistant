"""
State definitions for Singapore Housing Assistant RAG System.
Defines the data structures that flow through the LangGraph workflow.
"""

from typing import List, Annotated
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    """
    Custom reducer for agent_answers.
    Allows resetting the list when a special '__reset__' marker is present.
    
    This is used when starting a new conversation to clear previous answers.
    """
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new


class State(MessagesState):
    """
    Main state for the housing assistant graph.
    
    Inherits from MessagesState to get message history management.
    Tracks conversation flow, query analysis, and aggregated answers.
    """
    
    # Query analysis
    questionIsClear: bool = False
    """Whether the user's question is clear enough to process"""
    
    conversation_summary: str = ""
    """Brief summary of recent conversation for context"""
    
    originalQuery: str = ""
    """The user's original query before rewriting"""
    
    rewrittenQuestions: List[str] = []
    """List of rewritten queries optimized for retrieval"""
    
    # Agent answers (with custom reducer)
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []
    """
    List of answers from individual agent executions.
    Each dict contains: {"index": int, "question": str, "answer": str}
    """


class AgentState(MessagesState):
    """
    State for individual agent subgraph.
    
    Each agent processes one question independently.
    This state tracks the agent's execution for that single question.
    """
    
    question: str = ""
    """The specific question this agent is answering"""
    
    question_index: int = 0
    """Index of this question in the list of rewritten questions"""
    
    final_answer: str = ""
    """The agent's final answer after retrieval and reasoning"""
    
    agent_answers: List[dict] = []
    """
    The answer to be passed back to main graph.
    Format: [{"index": int, "question": str, "answer": str}]
    """


class QueryAnalysis(BaseModel):
    """
    Structured output for query analysis.
    
    LLM outputs this structure to indicate:
    1. Whether the query is clear
    2. One or more rewritten queries
    3. What clarification is needed (if unclear)
    """
    
    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    
    questions: List[str] = Field(
        description=(
            "List of rewritten, self-contained questions optimized for retrieval. "
            "Can contain 1-3 questions if the original has multiple parts."
        )
    )
    
    clarification_needed: str = Field(
        description=(
            "Explanation if the question is unclear. "
            "Should guide the user on what information is needed. "
            "Empty if question is clear."
        ),
        default=""
    )