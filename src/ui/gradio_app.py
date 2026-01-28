"""
Gradio Chat Interface for Singapore Housing Assistant.

This module provides a web-based chat interface using Gradio.
"""

import uuid
import gradio as gr
from langchain_core.messages import HumanMessage

from src.config import (
    get_llm_config,
    CHILD_COLLECTION,
    QDRANT_DB_PATH,
    DENSE_MODEL,
    SPARSE_MODEL,
    LLM_PROVIDER
)
from src.rag_agent.tools import ToolFactory
from src.rag_agent.graph import create_agent_graph
from src.rag_agent.prompts import get_welcome_message

# Import LLM based on provider
if LLM_PROVIDER == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI
else:
    from langchain_openai import ChatOpenAI


class ChatSession:
    """Manages a single chat session with the RAG agent."""

    def __init__(self):
        self.agent_graph = None
        self.config = None
        self.initialized = False

    def initialize(self):
        """Initialize the RAG system."""
        if self.initialized:
            return

        # Initialize LLM
        llm_config = get_llm_config()

        if LLM_PROVIDER == "gemini":
            llm = ChatGoogleGenerativeAI(**llm_config)
        else:
            llm = ChatOpenAI(**llm_config)

        # Initialize vector store
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_qdrant.fastembed_sparse import FastEmbedSparse
        from langchain_qdrant import QdrantVectorStore, RetrievalMode
        from qdrant_client import QdrantClient

        dense_embeddings = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
        sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL)

        client = QdrantClient(path=QDRANT_DB_PATH)

        if not client.collection_exists(CHILD_COLLECTION):
            raise RuntimeError(
                "Vector database not found! Please run 'python indexing.py' first."
            )

        child_vector_store = QdrantVectorStore(
            client=client,
            collection_name=CHILD_COLLECTION,
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name="sparse"
        )

        # Create tools and graph
        tool_factory = ToolFactory(collection=child_vector_store)
        tools = tool_factory.create_tools()
        self.agent_graph = create_agent_graph(llm, tools)

        # Create conversation config
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        self.initialized = True

    def reset(self):
        """Reset the conversation session."""
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    def chat(self, message: str) -> str:
        """Process a user message and return the assistant's response."""
        if not self.initialized:
            self.initialize()

        try:
            # Check if graph is waiting for input
            current_state = self.agent_graph.get_state(self.config)

            if current_state.next:
                # Resume from interruption
                self.agent_graph.update_state(
                    self.config,
                    {"messages": [HumanMessage(content=message)]}
                )
                result = self.agent_graph.invoke(None, self.config)
            else:
                # New query
                result = self.agent_graph.invoke(
                    {"messages": [HumanMessage(content=message)]},
                    self.config
                )

            return result['messages'][-1].content

        except Exception as e:
            return f"Error: {str(e)}"


# Global session instance
_session = None


def get_session() -> ChatSession:
    """Get or create the global chat session."""
    global _session
    if _session is None:
        _session = ChatSession()
    return _session


def respond(message: str, history: list) -> str:
    """
    Gradio chat callback function.

    Args:
        message: User's input message
        history: List of [user_message, assistant_response] pairs

    Returns:
        Assistant's response string
    """
    if not message.strip():
        return ""

    session = get_session()
    response = session.chat(message)
    return response


def clear_chat():
    """Clear the chat history and reset the session."""
    session = get_session()
    session.reset()
    return [], ""


def create_gradio_app() -> gr.Blocks:
    """
    Create and configure the Gradio chat interface.

    Returns:
        Configured Gradio Blocks app
    """
    welcome_msg = get_welcome_message()

    with gr.Blocks(title="Singapore Housing Assistant") as app:
        gr.Markdown("# Singapore Housing Rental Assistant")
        gr.Markdown(welcome_msg)

        chatbot = gr.Chatbot(
            label="Chat",
            height=450
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Type your question here...",
                scale=9,
                show_label=False
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)

        clear_btn = gr.Button("Clear Chat", variant="secondary")

        # Event handlers
        def user_submit(user_message, history):
            """Handle user message submission."""
            if not user_message.strip():
                return "", history

            # Add user message to history (Gradio 6.x format)
            history = history + [{"role": "user", "content": user_message}]
            return "", history

        def bot_respond(history):
            """Generate bot response."""
            if not history:
                return history

            # Check if last message is from user (needs response)
            if history[-1]["role"] != "user":
                return history

            user_message = history[-1]["content"]
            session = get_session()
            response = session.chat(user_message)
            history = history + [{"role": "assistant", "content": response}]
            return history

        # Wire up events
        msg.submit(
            user_submit,
            [msg, chatbot],
            [msg, chatbot],
            queue=False
        ).then(
            bot_respond,
            chatbot,
            chatbot
        )

        submit_btn.click(
            user_submit,
            [msg, chatbot],
            [msg, chatbot],
            queue=False
        ).then(
            bot_respond,
            chatbot,
            chatbot
        )

        clear_btn.click(
            clear_chat,
            outputs=[chatbot, msg]
        )

    return app
