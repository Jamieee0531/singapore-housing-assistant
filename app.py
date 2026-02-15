"""
Singapore Housing Assistant - Gradio Web Interface

Usage:
    python app.py

Then open http://localhost:7860 in your browser.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)


def main():
    """Launch the Gradio web interface."""
    # Check for .env file
    if not os.path.exists('.env'):
        print("\nWarning: .env file not found!")
        print("Please create a .env file with your API keys.")
        print("See .env.example for reference.\n")
        sys.exit(1)

    from src.config import setup_logging
    setup_logging()

    logger.info("Starting Singapore Housing Assistant...")
    logger.info("Loading models and initializing system...")

    try:
        from src.ui.gradio_app import create_gradio_app, get_session

        # Pre-initialize the session to show loading progress
        logger.info("Initializing RAG system...")
        session = get_session()
        session.initialize()
        logger.info("System ready!")

        # Create and launch app
        app = create_gradio_app()

        logger.info("Singapore Housing Assistant is running!")
        logger.info("Open http://localhost:7860 in your browser")

        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
