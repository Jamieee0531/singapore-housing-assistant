"""
Singapore Housing Assistant - Gradio Web Interface

Usage:
    python app.py

Then open http://localhost:7860 in your browser.
"""

import os
import sys


def main():
    """Launch the Gradio web interface."""
    # Check for .env file
    if not os.path.exists('.env'):
        print("\nWarning: .env file not found!")
        print("Please create a .env file with your API keys.")
        print("See .env.example for reference.\n")
        sys.exit(1)

    print("\nStarting Singapore Housing Assistant...")
    print("Loading models and initializing system...\n")

    try:
        from src.ui.gradio_app import create_gradio_app, get_session

        # Pre-initialize the session to show loading progress
        print("Initializing RAG system...")
        session = get_session()
        session.initialize()
        print("System ready!\n")

        # Create and launch app
        app = create_gradio_app()

        print("=" * 50)
        print("Singapore Housing Assistant is running!")
        print("Open http://localhost:7860 in your browser")
        print("Press Ctrl+C to stop")
        print("=" * 50 + "\n")

        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
