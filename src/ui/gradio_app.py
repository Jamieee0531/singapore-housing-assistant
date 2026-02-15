"""
Gradio Chat Interface for Singapore Housing Assistant.

This module provides a web-based chat interface using Gradio.
Supports English and Chinese languages.
"""

import logging
import uuid
import gradio as gr

logger = logging.getLogger(__name__)
from langchain_core.messages import HumanMessage

from src.config import (
    get_llm_config,
    CHILD_COLLECTION,
    QDRANT_DB_PATH,
    DENSE_MODEL,
    SPARSE_MODEL,
    LLM_PROVIDER,
    GOOGLE_MAPS_API_KEY
)
from src.rag_agent.tools import ToolFactory
from src.rag_agent.maps_tools import MapsToolFactory
from src.rag_agent.graph import create_agent_graph
from src.i18n import get_ui_text, get_welcome_message, Language

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

        # Add Maps tools if API key is configured
        if GOOGLE_MAPS_API_KEY:
            try:
                maps_factory = MapsToolFactory(GOOGLE_MAPS_API_KEY)
                maps_tools = maps_factory.create_tools()
                tools.extend(maps_tools)
                logger.info("Google Maps tools loaded")
            except Exception as e:
                logger.warning("Maps tools not loaded: %s", e)

        self.agent_graph = create_agent_graph(llm, tools)

        # Create conversation config
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        self.initialized = True

    def reset(self):
        """Reset the conversation session."""
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    def chat(self, message: str, language: Language = "en") -> str:
        """
        Process a user message and return the assistant's response.

        Args:
            message: User's input message
            language: Response language ('en' or 'zh')

        Returns:
            Assistant's response string
        """
        if not self.initialized:
            self.initialize()

        try:
            # Check if graph is waiting for input
            current_state = self.agent_graph.get_state(self.config)

            if current_state.next:
                # Resume from interruption
                self.agent_graph.update_state(
                    self.config,
                    {"messages": [HumanMessage(content=message)], "language": language}
                )
                result = self.agent_graph.invoke(None, self.config)
            else:
                # New query with language setting
                result = self.agent_graph.invoke(
                    {"messages": [HumanMessage(content=message)], "language": language},
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


def create_gradio_app() -> gr.Blocks:
    """
    Create and configure the Gradio chat interface.

    Returns:
        Configured Gradio Blocks app
    """
    # Initial UI text (English)
    initial_lang = "en"
    ui_text = get_ui_text(initial_lang)
    welcome_msg = get_welcome_message(initial_lang)

    with gr.Blocks(title="Singapore Housing Assistant") as app:
        # State to track current language
        current_lang = gr.State(value=initial_lang)

        # Header with language selector
        with gr.Row():
            with gr.Column(scale=9):
                title_md = gr.Markdown(ui_text["title"])
            with gr.Column(scale=1):
                lang_dropdown = gr.Dropdown(
                    choices=[("English", "en"), ("中文", "zh")],
                    value=initial_lang,
                    label="Language / 语言",
                    interactive=True
                )

        welcome_md = gr.Markdown(welcome_msg)

        chatbot = gr.Chatbot(
            label=ui_text["chat_label"],
            height=450
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder=ui_text["input_placeholder"],
                scale=9,
                show_label=False
            )
            submit_btn = gr.Button(ui_text["send_button"], variant="primary", scale=1)

        clear_btn = gr.Button(ui_text["clear_button"], variant="secondary")

        # =====================================================================
        # Event Handlers
        # =====================================================================

        def on_language_change(lang: str):
            """Update UI when language changes."""
            ui = get_ui_text(lang)
            welcome = get_welcome_message(lang)
            return (
                lang,                           # current_lang state
                ui["title"],                    # title_md
                welcome,                        # welcome_md
                gr.update(placeholder=ui["input_placeholder"]),  # msg
                ui["send_button"],              # submit_btn
                ui["clear_button"]              # clear_btn
            )

        def user_submit(user_message, history):
            """Handle user message submission."""
            if not user_message.strip():
                return "", history

            # Add user message to history (Gradio 6.x format)
            history = history + [{"role": "user", "content": user_message}]
            return "", history

        def bot_respond(history, lang):
            """Generate bot response."""
            if not history:
                return history

            # Check if last message is from user (needs response)
            if history[-1]["role"] != "user":
                return history

            user_message = history[-1]["content"]
            session = get_session()
            response = session.chat(user_message, language=lang)
            history = history + [{"role": "assistant", "content": response}]
            return history

        def clear_chat(lang):
            """Clear the chat history and reset the session."""
            session = get_session()
            session.reset()
            return [], ""

        # =====================================================================
        # Wire up events
        # =====================================================================

        # Language change
        lang_dropdown.change(
            on_language_change,
            inputs=[lang_dropdown],
            outputs=[current_lang, title_md, welcome_md, msg, submit_btn, clear_btn]
        )

        # Submit on Enter
        msg.submit(
            user_submit,
            [msg, chatbot],
            [msg, chatbot],
            queue=False
        ).then(
            bot_respond,
            [chatbot, current_lang],
            chatbot
        )

        # Submit on button click
        submit_btn.click(
            user_submit,
            [msg, chatbot],
            [msg, chatbot],
            queue=False
        ).then(
            bot_respond,
            [chatbot, current_lang],
            chatbot
        )

        # Clear chat
        clear_btn.click(
            clear_chat,
            inputs=[current_lang],
            outputs=[chatbot, msg]
        )

    return app
