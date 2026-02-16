"""
Gradio Chat Interface for Singapore Housing Assistant.

This module provides a web-based chat interface using Gradio.
Supports English and Chinese languages with token-level streaming output.
"""

import logging
import uuid
from typing import AsyncGenerator

import gradio as gr

logger = logging.getLogger(__name__)
from langchain_core.messages import HumanMessage, AIMessage

from src.config import (
    get_llm_config,
    CHILD_COLLECTION,
    QDRANT_DB_PATH,
    DENSE_MODEL,
    SPARSE_MODEL,
    LLM_PROVIDER,
    GOOGLE_MAPS_API_KEY,
    THREAD_ID_PATH,
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

# Node name to user-facing progress message mapping
NODE_PROGRESS = {
    "en": {
        "summarize": "Analyzing conversation...",
        "analyze_rewrite": "Understanding your question...",
        "process_question": "Searching knowledge base...",
        "aggregate": "Generating answer...",
    },
    "zh": {
        "summarize": "分析对话中...",
        "analyze_rewrite": "理解您的问题...",
        "process_question": "搜索知识库...",
        "aggregate": "生成回答...",
    },
}


def _load_or_create_thread_id() -> str:
    """Load persisted thread ID from file, or create a new one."""
    try:
        with open(THREAD_ID_PATH, "r") as f:
            thread_id = f.read().strip()
            if thread_id:
                return thread_id
    except FileNotFoundError:
        pass
    return _save_new_thread_id()


def _save_new_thread_id() -> str:
    """Generate a new thread ID and persist it to file."""
    thread_id = str(uuid.uuid4())
    with open(THREAD_ID_PATH, "w") as f:
        f.write(thread_id)
    return thread_id


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

        # Load or create persisted thread ID
        thread_id = _load_or_create_thread_id()
        self.config = {"configurable": {"thread_id": thread_id}}
        logger.info("Session initialized with thread_id: %s", thread_id)
        self.initialized = True

    def reset(self):
        """Reset the conversation session with a new thread ID."""
        thread_id = _save_new_thread_id()
        self.config = {"configurable": {"thread_id": thread_id}}
        logger.info("Session reset with new thread_id: %s", thread_id)

    def chat(self, message: str, language: Language = "en") -> str:
        """
        Process a user message and return the assistant's response (blocking).

        Args:
            message: User's input message
            language: Response language ('en' or 'zh')

        Returns:
            Assistant's response string
        """
        if not self.initialized:
            self.initialize()

        try:
            current_state = self.agent_graph.get_state(self.config)

            if current_state.next:
                self.agent_graph.update_state(
                    self.config,
                    {"messages": [HumanMessage(content=message)], "language": language}
                )
                result = self.agent_graph.invoke(None, self.config)
            else:
                result = self.agent_graph.invoke(
                    {"messages": [HumanMessage(content=message)], "language": language},
                    self.config
                )

            return result['messages'][-1].content

        except Exception as e:
            return f"Error: {str(e)}"

    async def chat_stream(
        self, message: str, language: Language = "en"
    ) -> AsyncGenerator[tuple[str, str], None]:
        """
        Streaming chat using astream_events.

        Yields (status, accumulated_content) tuples where:
        - status: progress indicator (node name or empty when streaming tokens)
        - accumulated_content: the text to display (progress message or growing answer)

        Args:
            message: User's input message
            language: Response language ('en' or 'zh')
        """
        if not self.initialized:
            self.initialize()

        progress_map = NODE_PROGRESS.get(language, NODE_PROGRESS["en"])

        try:
            current_state = self.agent_graph.get_state(self.config)
            is_resuming = bool(current_state.next)

            if is_resuming:
                self.agent_graph.update_state(
                    self.config,
                    {"messages": [HumanMessage(content=message)], "language": language}
                )
                stream_input = None
            else:
                stream_input = {
                    "messages": [HumanMessage(content=message)],
                    "language": language,
                }

            accumulated = ""
            is_streaming_tokens = False

            async for event in self.agent_graph.astream_events(
                stream_input, config=self.config, version="v2"
            ):
                kind = event.get("event", "")
                name = event.get("name", "")
                tags = event.get("tags", [])

                # Node-level progress: show status when a node starts
                if kind == "on_chain_start" and name in progress_map:
                    status_msg = progress_map[name]
                    if not is_streaming_tokens:
                        yield status_msg, ""

                # Token streaming from the aggregate LLM only
                if (
                    kind == "on_chat_model_stream"
                    and "aggregate_llm" in tags
                ):
                    is_streaming_tokens = True
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        accumulated += chunk.content
                        yield "", accumulated

            # If no tokens were streamed, the graph was likely interrupted
            # (unclear question at human_input node) or completed without aggregate.
            if not accumulated:
                final_state = self.agent_graph.get_state(self.config)

                if final_state.next:
                    # Graph paused at interrupt (human_input node)
                    messages = final_state.values.get("messages", [])
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            yield "", msg.content
                            return
                    yield "", "Could you please clarify your question?"
                else:
                    # Graph completed but no streaming tokens captured
                    messages = final_state.values.get("messages", [])
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            yield "", msg.content
                            return
                    yield "", "No response generated."

        except Exception as e:
            logger.exception("Streaming error")
            yield "", f"Error: {str(e)}"


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

        async def bot_respond(history, lang):
            """Generate bot response with token-level streaming."""
            if not history or history[-1]["role"] != "user":
                yield history
                return

            user_message = history[-1]["content"]
            session = get_session()

            # Create base history with empty assistant message
            base_history = history + [{"role": "assistant", "content": ""}]

            async for status, content in session.chat_stream(user_message, language=lang):
                if content:
                    new_msg = {"role": "assistant", "content": content}
                elif status:
                    new_msg = {"role": "assistant", "content": f"⏳ {status}"}
                else:
                    continue
                yield base_history[:-1] + [new_msg]

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
