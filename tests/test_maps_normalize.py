"""Tests for maps_tools location normalization."""

import sys
from unittest.mock import MagicMock

# Mock googlemaps before importing maps_tools
sys.modules.setdefault("googlemaps", MagicMock())

import pytest
from src.rag_agent.maps_tools import _normalize_location, SINGAPORE_LOCATIONS


class TestNormalizeLocation:
    """Tests for the _normalize_location helper function."""

    def test_known_abbreviation_nus(self):
        assert _normalize_location("NUS") == "National University of Singapore"

    def test_known_abbreviation_ntu(self):
        assert _normalize_location("NTU") == "Nanyang Technological University"

    def test_known_abbreviation_case_insensitive(self):
        assert _normalize_location("nus") == "National University of Singapore"
        assert _normalize_location("Nus") == "National University of Singapore"

    def test_adds_singapore_suffix(self):
        assert _normalize_location("Clementi") == "Clementi, Singapore"

    def test_does_not_double_add_singapore(self):
        result = _normalize_location("Clementi, Singapore")
        assert result == "Clementi, Singapore"
        assert result.count("Singapore") == 1

    def test_case_insensitive_singapore_check(self):
        result = _normalize_location("Orchard Road singapore")
        assert result == "Orchard Road singapore"

    def test_strips_whitespace(self):
        assert _normalize_location("  NUS  ") == "National University of Singapore"

    def test_all_abbreviations_resolve(self):
        for abbr, expected in SINGAPORE_LOCATIONS.items():
            assert _normalize_location(abbr) == expected

    def test_orchard_resolves(self):
        result = _normalize_location("Orchard")
        assert result == "Orchard Road Singapore"

    def test_orchard_case_insensitive(self):
        assert _normalize_location("orchard") == "Orchard Road Singapore"

    def test_unknown_location(self):
        result = _normalize_location("Some Random Place")
        assert result == "Some Random Place, Singapore"
