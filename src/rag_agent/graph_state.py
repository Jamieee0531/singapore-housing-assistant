"""
State definitions for Singapore Housing Assistant RAG System.
Defines the data structures that flow through the LangGraph workflow.
"""

from typing import List, Optional, Annotated
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


def use_last_value(existing: str, new: str) -> str:
    """
    Simple reducer that always uses the latest value.
    Used for fields like 'language' that may be set multiple times.
    """
    return new if new else existing


def use_last_list(existing: list, new: list) -> list:
    """
    Simple reducer for list fields that always uses the latest value.
    Used for fields like 'relevant_topics' that may be set by multiple nodes.
    """
    return new if new else existing


class State(MessagesState):
    """
    Main state for the housing assistant graph.

    Inherits from MessagesState to get message history management.
    Tracks conversation flow, query analysis, and aggregated answers.
    """

    # User identification
    user_id: str = "default"
    """User ID for Redis profile lookup"""

    # Language setting (with reducer to handle concurrent updates)
    language: Annotated[str, use_last_value] = "en"
    """User's selected language: 'en' for English, 'zh' for Chinese"""

    # Query analysis
    questionIsClear: bool = False
    """Whether the user's question is clear enough to process"""

    conversation_summary: str = ""
    """Brief summary of recent conversation for context"""

    originalQuery: str = ""
    """The user's original query before rewriting"""

    rewrittenQuestions: List[str] = []
    """List of rewritten queries optimized for retrieval"""

    relevant_topics: Annotated[List[str], use_last_list] = []
    """Topic labels for metadata filtering (e.g., ['transport', 'pricing'])"""

    # User profile (long-term memory from Redis)
    user_profile: dict = {}
    """User preferences loaded from Redis at session start"""

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

    language: Annotated[str, use_last_value] = "en"
    """User's selected language: 'en' for English, 'zh' for Chinese"""

    question: str = ""
    """The specific question this agent is answering"""

    question_index: int = 0
    """Index of this question in the list of rewritten questions"""

    relevant_topics: List[str] = []
    """Topic labels for metadata filtering, passed from main graph"""

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

    # User preference extraction (piggybacks on the same LLM call)
    extracted_preferences: Optional[dict] = Field(
        description=(
            "User preferences extracted from the current query. "
            "Possible keys: school, budget_range, preferred_area, rental_type, "
            "room_type, move_in_date, transport_requirement, environment_preference. "
            "Only include keys that the user explicitly mentioned. "
            "Return null if no preferences are detected."
        ),
        default=None
    )

    # Topic classification for metadata filtering
    relevant_topics: List[str] = Field(
        description=(
            "Which knowledge base topics are relevant to this query. "
            "Choose from: "
            "housing_types (HDB vs Condo differences, property types), "
            "pricing (rental prices, costs, budget, saving money), "
            "area (specific neighborhoods or regions), "
            "transport (MRT, bus, commuting, EZ-Link card), "
            "utilities (electricity, water, gas, aircon, internet), "
            "rental_process (how to rent, contracts, deposits, moving in/out), "
            "legal (visa, Student Pass, stamp duty, scams, agent verification). "
            "Select 1-3 most relevant topics."
        ),
        default=[]
    )