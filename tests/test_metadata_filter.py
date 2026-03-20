"""
Tests for metadata filtering: topic tagging and filter construction.
"""

import pytest
from src.config import FILE_TOPIC_MAPPING, ALL_TOPICS


class TestFileTopicMapping:
    """Tests for document → topic label mapping."""

    def test_all_docs_have_topics(self):
        """Every document in the knowledge base should have at least one topic."""
        expected_files = [
            "hdb_vs_condo.md",
            "price_range.md",
            "area_guide_central.md",
            "area_guide_east.md",
            "area_guide_west.md",
            "transport_guide.md",
            "utilities_setup.md",
            "rental_guide.md",
            "rental_scams.md",
            "visa_housing_rules.md",
            "student_budget_tips.md",
        ]
        for doc in expected_files:
            assert doc in FILE_TOPIC_MAPPING, f"{doc} missing from FILE_TOPIC_MAPPING"
            assert len(FILE_TOPIC_MAPPING[doc]) > 0, f"{doc} has no topics"

    def test_topics_are_valid(self):
        """All topics in the mapping should be from the defined set."""
        for doc, topics in FILE_TOPIC_MAPPING.items():
            for topic in topics:
                assert topic in ALL_TOPICS, f"Unknown topic '{topic}' in {doc}"

    def test_every_topic_used_at_least_once(self):
        """Every defined topic should appear in at least one document."""
        used_topics = set()
        for topics in FILE_TOPIC_MAPPING.values():
            used_topics.update(topics)
        for topic in ALL_TOPICS:
            assert topic in used_topics, f"Topic '{topic}' is defined but never used"

    def test_area_guides_have_area_topic(self):
        """All area guide documents should have the 'area' topic."""
        for doc in ["area_guide_central.md", "area_guide_east.md", "area_guide_west.md"]:
            assert "area" in FILE_TOPIC_MAPPING[doc]

    def test_transport_guide_has_transport_topic(self):
        assert "transport" in FILE_TOPIC_MAPPING["transport_guide.md"]

    def test_utilities_guide_has_utilities_topic(self):
        assert "utilities" in FILE_TOPIC_MAPPING["utilities_setup.md"]

    def test_multi_topic_documents(self):
        """Documents that cover multiple topics should have multiple labels."""
        # area guides cover area + pricing + transport
        for doc in ["area_guide_central.md", "area_guide_east.md", "area_guide_west.md"]:
            assert len(FILE_TOPIC_MAPPING[doc]) >= 3, f"{doc} should have multiple topics"


class TestFilterConstruction:
    """Tests for building Qdrant metadata filters from topics."""

    def test_build_filter_single_topic(self):
        from src.rag_agent.tools import _build_topic_filter
        f = _build_topic_filter(["transport"])
        assert f is not None

    def test_build_filter_multiple_topics(self):
        from src.rag_agent.tools import _build_topic_filter
        f = _build_topic_filter(["transport", "pricing"])
        assert f is not None

    def test_build_filter_empty_topics_returns_none(self):
        from src.rag_agent.tools import _build_topic_filter
        f = _build_topic_filter([])
        assert f is None

    def test_build_filter_invalid_topic_ignored(self):
        from src.rag_agent.tools import _build_topic_filter
        f = _build_topic_filter(["nonexistent_topic"])
        assert f is None
