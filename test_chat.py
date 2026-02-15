"""
Command-line chat interface for testing Singapore Housing Assistant RAG system.

Usage:
    python test_chat.py

Commands:
    - Type your question and press Enter
    - Type 'exit' or 'quit' to exit
    - Type 'clear' to start a new conversation
    - Type 'help' for example questions
"""

import os
from langchain_core.messages import HumanMessage

from src.config import get_llm_config, setup_logging, CHILD_COLLECTION, QDRANT_DB_PATH
from src.rag_agent.tools import ToolFactory
from src.rag_agent.graph import create_agent_graph

# Import LLM based on provider
from src.config import LLM_PROVIDER
if LLM_PROVIDER == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI
else:
    from langchain_openai import ChatOpenAI


def print_welcome():
    """Print welcome message and instructions."""
    print("\n" + "="*60)
    print("🏠 Singapore Housing Rental Assistant")
    print("="*60)
    print("\nI can help you with:")
    print("  • HDB vs Condo comparisons")
    print("  • Rental prices by area")
    print("  • Rental process and tips")
    print("  • Area recommendations")
    print("\nType 'help' for example questions")
    print("Type 'exit' to quit\n")
    print("="*60 + "\n")


def print_help():
    """Print example questions."""
    print("\n💡 Example Questions:")
    print("  • What is the difference between HDB and Condo?")
    print("  • How much does it cost to rent in Clementi?")
    print("  • What should I know about the rental process?")
    print("  • Which areas are good for NUS students?")
    print("  • What are some budget-saving tips?")
    print("  • Tell me about rental deposits and contracts")
    print()


def initialize_system():
    """
    Initialize the RAG system with LLM, tools, and graph.
    
    Returns:
        tuple: (agent_graph, config) for running conversations
    """
    print("🔧 Initializing system...")
    
    # Step 1: Initialize LLM
    llm_config = get_llm_config()
    
    if LLM_PROVIDER == "gemini":
        llm = ChatGoogleGenerativeAI(**llm_config)
        print(f"✓ Using Google Gemini: {llm_config['model']}")
    else:
        llm = ChatOpenAI(**llm_config)
        print(f"✓ Using OpenAI: {llm_config['model']}")
    
    # Step 2: Initialize vector store and tools
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant.fastembed_sparse import FastEmbedSparse
    from langchain_qdrant import QdrantVectorStore, RetrievalMode
    from qdrant_client import QdrantClient
    from src.config import DENSE_MODEL, SPARSE_MODEL
    
    print("✓ Loading embeddings...")
    dense_embeddings = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL)
    
    print("✓ Connecting to Qdrant...")
    client = QdrantClient(path=QDRANT_DB_PATH)
    
    if not client.collection_exists(CHILD_COLLECTION):
        print("\n❌ Error: Vector database not found!")
        print("Please run 'python indexing.py' first to index your documents.\n")
        exit(1)
    
    child_vector_store = QdrantVectorStore(
        client=client,
        collection_name=CHILD_COLLECTION,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="sparse"
    )
    
    # Step 3: Create tools
    print("✓ Creating RAG tools...")
    tool_factory = ToolFactory(collection=child_vector_store)
    tools = tool_factory.create_tools()
    
    # Step 4: Build graph
    print("✓ Building agent graph...")
    agent_graph = create_agent_graph(llm, tools)
    
    # Step 5: Create conversation config
    import uuid
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    print("✓ System ready!\n")
    
    return agent_graph, config


def chat_loop(agent_graph, config):
    """
    Main chat loop for interactive conversation.
    
    Args:
        agent_graph: Compiled LangGraph
        config: Configuration dict with thread_id
    """
    print_welcome()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye! Thanks for using Singapore Housing Assistant.\n")
                break
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            if user_input.lower() == 'clear':
                # Start new conversation
                import uuid
                config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                print("\n🔄 Started new conversation\n")
                continue
            
            # Process query
            print("\n🤔 Thinking...\n")
            
            # Check if graph is waiting for input (human-in-the-loop)
            current_state = agent_graph.get_state(config)
            
            if current_state.next:
                # Resume from interruption
                agent_graph.update_state(
                    config,
                    {"messages": [HumanMessage(content=user_input)]}
                )
                result = agent_graph.invoke(None, config)
            else:
                # New query
                result = agent_graph.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config
                )
            
            # Display response
            assistant_message = result['messages'][-1].content
            print(f"Assistant: {assistant_message}\n")
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'exit' to quit.\n")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
            print()


def main():
    """Main entry point."""
    setup_logging()
    try:
        # Check if .env file exists
        if not os.path.exists('.env'):
            print("\n⚠️  Warning: .env file not found!")
            print("Please create a .env file with your API keys.")
            print("See .env.example for reference.\n")
            
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Exiting...")
                return
        
        # Initialize system
        agent_graph, config = initialize_system()
        
        # Start chat loop
        chat_loop(agent_graph, config)
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()