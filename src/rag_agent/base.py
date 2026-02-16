"""Base class and utilities for tool factories in the RAG agent system."""

import logging
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import List

logger = logging.getLogger(__name__)


def timed_tool(func):
    """Decorator that logs execution time of a tool function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.info("%s completed in %.2fs", func.__name__, duration)
        return result
    return wrapper


class BaseToolFactory(ABC):
    """
    Abstract base class for tool factories.

    All tool factories (RAG retrieval, Google Maps, etc.) should inherit
    from this class to ensure a consistent interface.
    """

    @abstractmethod
    def create_tools(self) -> List:
        """
        Create and return a list of LangChain tools.

        Returns:
            List of tools ready to be bound to an LLM
        """
