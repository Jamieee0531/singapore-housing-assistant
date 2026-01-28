"""
Node functions for Singapore Housing Assistant RAG workflow.

Implements the behavior of each node in the LangGraph:
- Conversation summarization
- Query analysis and rewriting
- Agent execution (retrieval + reasoning)
- Answer extraction and aggregation
"""

from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langgraph.types import Send

from src.rag_agent.graph_state import State, AgentState, QueryAnalysis
from src.rag_agent.prompts import (
    get_conversation_summary_prompt,
    get_query_analysis_prompt,
    get_rag_agent_prompt,
    get_aggregation_prompt
)
from src.i18n import get_language_instruction


# =============================================================================
# Node 1: Conversation Summarization
# =============================================================================

def analyze_chat_and_summarize(state: State, llm):
    """
    Analyzes chat history and summarizes key points for context.
    
    Args:
        state: Current conversation state
        llm: Language model instance
        
    Returns:
        Updated state with conversation_summary and reset agent_answers
    """
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}
    
    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage))
        and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}
    
    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke([
        SystemMessage(content=get_conversation_summary_prompt()),
        HumanMessage(content=conversation)
    ])
    
    return {
        "conversation_summary": summary_response.content,
        "agent_answers": [{"__reset__": True}]
    }


# =============================================================================
# Node 2: Query Analysis and Rewriting
# =============================================================================

def analyze_and_rewrite_query(state: State, llm):
    """
    Analyzes user query and rewrites it for optimal retrieval.
    
    Args:
        state: Current conversation state
        llm: Language model instance
        
    Returns:
        Updated state with questionIsClear and rewrittenQuestions or clarification
    """
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (
        f"Conversation Context:\n{conversation_summary}\n" 
        if conversation_summary.strip() else ""
    ) + f"User Query:\n{last_message.content}\n"

    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([
        SystemMessage(content=get_query_analysis_prompt()),
        HumanMessage(content=context_section)
    ])

    if len(response.questions) > 0 and response.is_clear:
        delete_all = [
            RemoveMessage(id=m.id)
            for m in state["messages"]
            if not isinstance(m, SystemMessage)
        ]
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": last_message.content,
            "rewrittenQuestions": response.questions
        }
    else:
        clarification = (
            response.clarification_needed 
            if (response.clarification_needed and len(response.clarification_needed.strip()) > 10)
            else "I need more information to understand your question."
        )
        return {
            "questionIsClear": False,
            "messages": [AIMessage(content=clarification)]
        }


# =============================================================================
# Node 3: Human Input (placeholder for interruption)
# =============================================================================

def human_input_node(state: State):
    """
    Placeholder node for human-in-the-loop interruption.
    """
    return {}


# =============================================================================
# Node 4: Routing Logic
# =============================================================================

def route_after_rewrite(state: State) -> Literal["human_input", "process_question"]:
    """
    Route based on whether question is clear.

    Returns:
        - "human_input": Question unclear, wait for user input
        - List[Send]: Question clear, spawn agents for each sub-question
    """
    if not state.get("questionIsClear", False):
        return "human_input"
    else:
        language = state.get("language", "en")
        return [
            Send("process_question", {
                "question": query,
                "question_index": idx,
                "language": language,
                "messages": []
            })
            for idx, query in enumerate(state["rewrittenQuestions"])
        ]


# =============================================================================
# Node 5: Agent Execution (RAG retrieval + reasoning)
# =============================================================================

def agent_node(state: AgentState, llm_with_tools):
    """
    Main agent node that executes RAG workflow.

    Args:
        state: Agent state with current question
        llm_with_tools: Language model with tools bound

    Returns:
        Updated state with new messages
    """
    # Get language instruction based on user's language preference
    language = state.get("language", "en")
    language_instruction = get_language_instruction(language)
    sys_msg = SystemMessage(content=get_rag_agent_prompt(language_instruction))

    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        response = llm_with_tools.invoke([sys_msg] + [human_msg])
        return {"messages": [human_msg, response]}

    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# =============================================================================
# Node 6: Answer Extraction
# =============================================================================

def extract_final_answer(state: AgentState):
    """
    Extract the final answer from agent's conversation.
    
    Args:
        state: Agent state after tool execution
        
    Returns:
        Updated state with final_answer and agent_answers list
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return {
                "final_answer": msg.content,
                "agent_answers": [{
                    "index": state["question_index"],
                    "question": state["question"],
                    "answer": msg.content
                }]
            }
    
    return {
        "final_answer": "Unable to generate an answer.",
        "agent_answers": [{
            "index": state["question_index"],
            "question": state["question"],
            "answer": "Unable to generate an answer."
        }]
    }


# =============================================================================
# Node 7: Response Aggregation
# =============================================================================

def aggregate_responses(state: State, llm):
    """
    Aggregate multiple agent answers into one coherent response.
    
    Args:
        state: State with agent_answers list
        llm: Language model instance
        
    Returns:
        Updated state with final aggregated message
    """
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += f"\nAnswer {i}:\n{ans['answer']}\n"

    user_message = HumanMessage(content=(
        f"Original user question: {state['originalQuery']}\n"
        f"Retrieved answers:{formatted_answers}"
    ))
    
    synthesis_response = llm.invoke([
        SystemMessage(content=get_aggregation_prompt()),
        user_message
    ])
    
    return {"messages": [AIMessage(content=synthesis_response.content)]}