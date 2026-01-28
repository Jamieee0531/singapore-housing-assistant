"""
Graph construction for Singapore Housing Assistant RAG System.

Builds the main LangGraph workflow with:
- Agent subgraph for RAG retrieval and reasoning
- Main graph for conversation management and query processing
"""

from functools import partial
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from src.rag_agent.graph_state import State, AgentState
from src.rag_agent.nodes import (
    analyze_chat_and_summarize,
    analyze_and_rewrite_query,
    human_input_node,
    route_after_rewrite,
    agent_node,
    extract_final_answer,
    aggregate_responses
)


def create_agent_graph(llm, tools_list):
    """
    Create the complete agent graph with subgraph architecture.
    
    Args:
        llm: Language model instance (e.g., ChatGoogleGenerativeAI)
        tools_list: List of LangChain tools created by ToolFactory
        
    Returns:
        Compiled LangGraph with checkpointer and human-in-the-loop support
    """
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)
    
    # Initialize checkpointer for conversation state persistence
    checkpointer = InMemorySaver()
    
    # =========================================================================
    # Build Agent Subgraph (handles individual question retrieval)
    # =========================================================================
    
    agent_builder = StateGraph(AgentState)
    
    # Add nodes
    agent_builder.add_node("agent", partial(agent_node, llm_with_tools=llm_with_tools))
    agent_builder.add_node("tools", tool_node)
    agent_builder.add_node("extract_answer", extract_final_answer)
    
    # Add edges
    agent_builder.add_edge(START, "agent")
    agent_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",      # Agent wants to use tools
            END: "extract_answer"  # Agent finished, extract answer
        }
    )
    agent_builder.add_edge("tools", "agent")  # After tool use, back to agent
    agent_builder.add_edge("extract_answer", END)
    
    # Compile subgraph
    agent_subgraph = agent_builder.compile()
    
    # =========================================================================
    # Build Main Graph (orchestrates conversation flow)
    # =========================================================================
    
    graph_builder = StateGraph(State)
    
    # Add nodes with bound LLM parameters
    graph_builder.add_node("summarize", partial(analyze_chat_and_summarize, llm=llm))
    graph_builder.add_node("analyze_rewrite", partial(analyze_and_rewrite_query, llm=llm))
    graph_builder.add_node("human_input", human_input_node)
    graph_builder.add_node("process_question", agent_subgraph)  # Use subgraph as a node
    graph_builder.add_node("aggregate", partial(aggregate_responses, llm=llm))
    
    # Add edges
    graph_builder.add_edge(START, "summarize")
    graph_builder.add_edge("summarize", "analyze_rewrite")
    graph_builder.add_conditional_edges("analyze_rewrite", route_after_rewrite)
    graph_builder.add_edge("human_input", "analyze_rewrite")
    graph_builder.add_edge(["process_question"], "aggregate")
    graph_builder.add_edge("aggregate", END)
    
    # Compile main graph with checkpointer and interruption support
    agent_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_input"]  # Pause for unclear queries
    )
    
    print("✓ Agent graph compiled successfully\n")
    
    return agent_graph