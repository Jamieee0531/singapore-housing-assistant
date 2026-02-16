"""Tests for prompt templates."""

import pytest
from src.rag_agent.prompts import (
    get_conversation_summary_prompt,
    get_query_analysis_prompt,
    get_rag_agent_prompt,
    get_aggregation_prompt,
)


class TestConversationSummaryPrompt:
    """Tests for the conversation summary prompt."""

    def test_returns_non_empty_string(self):
        prompt = get_conversation_summary_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_contains_key_instructions(self):
        prompt = get_conversation_summary_prompt()
        assert "summary" in prompt.lower()
        assert "conversation" in prompt.lower()


class TestQueryAnalysisPrompt:
    """Tests for the query analysis prompt."""

    def test_returns_non_empty_string(self):
        prompt = get_query_analysis_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_contains_rewrite_instructions(self):
        prompt = get_query_analysis_prompt()
        assert "rewrite" in prompt.lower()
        assert "query" in prompt.lower()


class TestRagAgentPrompt:
    """Tests for the RAG agent prompt."""

    def test_without_language_instruction(self):
        prompt = get_rag_agent_prompt()
        assert "IMPORTANT" not in prompt.split("Response guidelines")[0][-50:]

    def test_with_language_instruction(self):
        prompt = get_rag_agent_prompt("请用中文回复。")
        assert "IMPORTANT" in prompt
        assert "请用中文回复。" in prompt

    def test_mentions_available_tools(self):
        prompt = get_rag_agent_prompt()
        assert "search_child_chunks" in prompt
        assert "retrieve_parent_chunks" in prompt
        assert "get_commute_info" in prompt

    def test_empty_language_instruction_no_important(self):
        prompt = get_rag_agent_prompt("")
        # Empty string is falsy, so IMPORTANT should not be appended
        assert not prompt.endswith("IMPORTANT: ")


class TestAggregationPrompt:
    """Tests for the aggregation prompt."""

    def test_without_language_instruction(self):
        prompt = get_aggregation_prompt()
        assert "combine" in prompt.lower() or "aggregat" in prompt.lower()

    def test_with_language_instruction(self):
        prompt = get_aggregation_prompt("Please respond in English.")
        assert "IMPORTANT" in prompt
        assert "Please respond in English." in prompt

    def test_mentions_sources_format(self):
        prompt = get_aggregation_prompt()
        assert "Sources" in prompt
