"""Tests for graph state reducers and data models."""

import pytest
from src.rag_agent.graph_state import (
    accumulate_or_reset,
    use_last_value,
    QueryAnalysis,
)


class TestAccumulateOrReset:
    """Tests for the agent_answers custom reducer."""

    def test_accumulates_new_items(self):
        existing = [{"index": 0, "question": "q1", "answer": "a1"}]
        new = [{"index": 1, "question": "q2", "answer": "a2"}]
        result = accumulate_or_reset(existing, new)
        assert len(result) == 2
        assert result[0]["question"] == "q1"
        assert result[1]["question"] == "q2"

    def test_accumulates_to_empty_list(self):
        result = accumulate_or_reset([], [{"index": 0, "answer": "a1"}])
        assert len(result) == 1

    def test_reset_clears_all(self):
        existing = [{"index": 0, "answer": "a1"}, {"index": 1, "answer": "a2"}]
        new = [{"__reset__": True}]
        result = accumulate_or_reset(existing, new)
        assert result == []

    def test_reset_marker_among_others(self):
        existing = [{"index": 0, "answer": "a1"}]
        new = [{"__reset__": True}, {"index": 1, "answer": "a2"}]
        result = accumulate_or_reset(existing, new)
        assert result == []

    def test_empty_new_list(self):
        existing = [{"index": 0, "answer": "a1"}]
        result = accumulate_or_reset(existing, [])
        assert result == existing

    def test_both_empty(self):
        result = accumulate_or_reset([], [])
        assert result == []


class TestUseLastValue:
    """Tests for the use_last_value reducer."""

    def test_returns_new_when_present(self):
        assert use_last_value("old", "new") == "new"

    def test_returns_existing_when_new_is_empty(self):
        assert use_last_value("old", "") == "old"

    def test_returns_new_when_existing_is_empty(self):
        assert use_last_value("", "new") == "new"

    def test_both_empty(self):
        assert use_last_value("", "") == ""


class TestQueryAnalysis:
    """Tests for the QueryAnalysis Pydantic model."""

    def test_valid_clear_query(self):
        qa = QueryAnalysis(
            is_clear=True,
            questions=["HDB rental prices in Clementi"],
        )
        assert qa.is_clear is True
        assert len(qa.questions) == 1
        assert qa.clarification_needed == ""

    def test_unclear_query_with_clarification(self):
        qa = QueryAnalysis(
            is_clear=False,
            questions=[],
            clarification_needed="Please specify which area you are asking about.",
        )
        assert qa.is_clear is False
        assert qa.clarification_needed != ""

    def test_multiple_questions(self):
        qa = QueryAnalysis(
            is_clear=True,
            questions=["HDB prices?", "Condo prices?", "Which is better?"],
        )
        assert len(qa.questions) == 3

    def test_default_clarification_empty(self):
        qa = QueryAnalysis(is_clear=True, questions=["test"])
        assert qa.clarification_needed == ""
