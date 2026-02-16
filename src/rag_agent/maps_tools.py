"""
Google Maps tools for Singapore Housing Assistant.

Three tools:
1. get_commute_info - Calculate commute distance and duration
2. get_directions - Get detailed route directions
3. search_nearby - Search for nearby amenities
"""

import logging
from functools import lru_cache
from typing import List
from langchain_core.tools import tool
import googlemaps
from src.rag_agent.base import BaseToolFactory, timed_tool
from src.config import (
    MAPS_SEARCH_RADIUS,
    MAPS_MAX_RESULTS,
    TOOL_ERROR_PREFIX,
    TOOL_NO_RESULTS_PREFIX,
)

logger = logging.getLogger(__name__)

# Common Singapore location abbreviation mappings
SINGAPORE_LOCATIONS = {
    "NUS": "National University of Singapore",
    "NTU": "Nanyang Technological University",
    "SMU": "Singapore Management University",
    "SUTD": "Singapore University of Technology and Design",
    "CBD": "Central Business District Singapore",
    "MBS": "Marina Bay Sands Singapore",
    "ORCHARD": "Orchard Road Singapore",
}


def _normalize_location(location: str) -> str:
    """
    Normalize location names by expanding abbreviations and adding Singapore context.

    Args:
        location: Raw location string (e.g., "NUS", "Clementi")

    Returns:
        Normalized location string with Singapore context
    """
    upper_loc = location.upper().strip()
    if upper_loc in SINGAPORE_LOCATIONS:
        return SINGAPORE_LOCATIONS[upper_loc]

    if "singapore" not in location.lower():
        return f"{location}, Singapore"

    return location


class MapsToolFactory(BaseToolFactory):
    """
    Factory for creating Google Maps tools.

    Usage:
        factory = MapsToolFactory(api_key)
        tools = factory.create_tools()
    """

    def __init__(self, api_key: str):
        """
        Initialize the Maps tool factory.

        Args:
            api_key: Google Maps API Key
        """
        self.client = googlemaps.Client(key=api_key)
        self._init_cached_methods()

    def _init_cached_methods(self):
        """Create LRU-cached wrappers around Google Maps API calls."""

        @lru_cache(maxsize=100)
        def cached_distance_matrix(origin: str, destination: str, mode: str):
            return self.client.distance_matrix(
                origins=[origin], destinations=[destination],
                mode=mode, region="sg"
            )

        @lru_cache(maxsize=100)
        def cached_directions(origin: str, destination: str):
            return self.client.directions(
                origin=origin, destination=destination,
                mode="transit", region="sg"
            )

        @lru_cache(maxsize=100)
        def cached_geocode(location: str):
            return self.client.geocode(location)

        @lru_cache(maxsize=100)
        def cached_places_nearby(lat: float, lng: float, radius: int, place_type: str):
            return self.client.places_nearby(
                location=(lat, lng), radius=radius, type=place_type
            )

        self._distance_matrix = cached_distance_matrix
        self._directions = cached_directions
        self._geocode = cached_geocode
        self._places_nearby = cached_places_nearby

    def create_tools(self) -> List:
        """Create and return all Maps tools."""

        # Capture cached methods for use in tool closures
        distance_matrix = self._distance_matrix
        directions_api = self._directions
        geocode = self._geocode
        places_nearby = self._places_nearby

        @tool
        def get_commute_info(origin: str, destination: str) -> str:
            """
            Calculate commute distance and duration between two locations.

            Use cases:
            - "How long from Clementi to NUS?"
            - "How far is Clementi from CBD?"
            - "Which area is closest to my school?"

            Args:
                origin: Starting location (e.g., "Clementi")
                destination: Destination location (e.g., "NUS", "CBD")

            Returns:
                Commute information with distance and duration for transit and driving.

                On failure:
                - "[NO_RESULTS] ..." if route cannot be calculated
                - "[ERROR] ..." if API call fails
            """
            try:
                origin_normalized = _normalize_location(origin)
                destination_normalized = _normalize_location(destination)

                result = distance_matrix(origin_normalized, destination_normalized, "transit")

                if result["rows"][0]["elements"][0]["status"] != "OK":
                    logger.warning("Commute route not found: '%s' -> '%s'", origin, destination)
                    return (
                        f"{TOOL_NO_RESULTS_PREFIX} Could not calculate route from "
                        f"{origin} to {destination}. Please check the location names."
                    )

                element = result["rows"][0]["elements"][0]
                distance = element["distance"]["text"]
                duration = element["duration"]["text"]

                driving_result = distance_matrix(origin_normalized, destination_normalized, "driving")

                driving_duration = ""
                if driving_result["rows"][0]["elements"][0]["status"] == "OK":
                    driving_duration = driving_result["rows"][0]["elements"][0]["duration"]["text"]

                response = f"""Commute from {origin} to {destination}:

Public Transit:
  - Distance: {distance}
  - Duration: {duration}

Driving: {driving_duration if driving_duration else 'N/A'}

Tip: Singapore has excellent public transit; MRT is the most common commute option."""

                return response

            except Exception as e:
                logger.error("Commute info lookup failed: %s", e, exc_info=True)
                return f"{TOOL_ERROR_PREFIX} Commute info lookup failed: {e}"

        @tool
        def get_directions(origin: str, destination: str) -> str:
            """
            Get detailed step-by-step directions from origin to destination.

            Use cases:
            - "How do I get from Clementi to school?"
            - "How to go from Jurong East to NUS?"

            Args:
                origin: Starting location
                destination: Destination location

            Returns:
                Detailed route steps with distance and duration.

                On failure:
                - "[NO_RESULTS] ..." if directions cannot be found
                - "[ERROR] ..." if API call fails
            """
            try:
                origin_normalized = _normalize_location(origin)
                destination_normalized = _normalize_location(destination)

                directions = directions_api(origin_normalized, destination_normalized)

                if not directions:
                    logger.warning("Directions not found: '%s' -> '%s'", origin, destination)
                    return (
                        f"{TOOL_NO_RESULTS_PREFIX} Could not find directions from "
                        f"{origin} to {destination}. Please check the location names."
                    )

                route = directions[0]
                legs = route["legs"][0]

                total_distance = legs["distance"]["text"]
                total_duration = legs["duration"]["text"]

                steps_text = []
                for i, step in enumerate(legs["steps"], 1):
                    instruction = step["html_instructions"]
                    instruction = instruction.replace("<b>", "").replace("</b>", "")
                    instruction = instruction.replace("<div style=\"font-size:0.9em\">", " - ")
                    instruction = instruction.replace("</div>", "")

                    step_distance = step["distance"]["text"]
                    step_duration = step["duration"]["text"]

                    travel_mode = step.get("travel_mode", "")
                    if travel_mode == "TRANSIT":
                        transit_details = step.get("transit_details", {})
                        line = transit_details.get("line", {})
                        line_name = line.get("short_name", line.get("name", ""))
                        if line_name:
                            instruction = f"Take {line_name} - {instruction}"
                    elif travel_mode == "WALKING":
                        instruction = f"Walk: {instruction}"

                    steps_text.append(f"{i}. {instruction} ({step_distance}, {step_duration})")

                response = f"""Directions from {origin} to {destination}:

Total distance: {total_distance}
Estimated time: {total_duration}

Steps:
{chr(10).join(steps_text)}

Tip: Use Google Maps or Citymapper app for real-time navigation."""

                return response

            except Exception as e:
                logger.error("Directions lookup failed: %s", e, exc_info=True)
                return f"{TOOL_ERROR_PREFIX} Directions lookup failed: {e}"

        @tool
        def search_nearby(location: str, place_type: str = "transit_station") -> str:
            """
            Search for amenities near a specified location.

            Use cases:
            - "Any MRT near Clementi?"
            - "Are there supermarkets around this area?"
            - "Any good restaurants nearby?"

            Args:
                location: Center location for search (e.g., "Clementi", "Jurong East")
                place_type: Type of amenity, options:
                    - "transit_station" (MRT/bus stops, default)
                    - "supermarket"
                    - "restaurant"
                    - "shopping_mall"
                    - "hospital"
                    - "school"

            Returns:
                List of nearby amenities with names, ratings, and addresses.

                On failure:
                - "[NO_RESULTS] ..." if location not found or no results
                - "[ERROR] ..." if API call fails
            """
            try:
                location_normalized = _normalize_location(location)

                geocode_result = geocode(location_normalized)
                if not geocode_result:
                    logger.warning("Geocode failed for location: '%s'", location)
                    return (
                        f"{TOOL_NO_RESULTS_PREFIX} Could not geocode location: "
                        f"{location}. Please check the name."
                    )

                lat = geocode_result[0]["geometry"]["location"]["lat"]
                lng = geocode_result[0]["geometry"]["location"]["lng"]

                type_mapping = {
                    "transit_station": "transit_station",
                    "mrt": "transit_station",
                    "supermarket": "supermarket",
                    "restaurant": "restaurant",
                    "shopping_mall": "shopping_mall",
                    "hospital": "hospital",
                    "school": "school",
                    "gym": "gym",
                    "park": "park",
                }

                search_type = type_mapping.get(place_type.lower(), "transit_station")

                places_result = places_nearby(lat, lng, MAPS_SEARCH_RADIUS, search_type)

                if not places_result.get("results"):
                    logger.warning("No %s found near '%s' (radius=%dm)", place_type, location, MAPS_SEARCH_RADIUS)
                    return (
                        f"{TOOL_NO_RESULTS_PREFIX} No {place_type} found within "
                        f"{MAPS_SEARCH_RADIUS}m of {location}."
                    )

                type_names = {
                    "transit_station": "Transit Stations",
                    "supermarket": "Supermarkets",
                    "restaurant": "Restaurants",
                    "shopping_mall": "Shopping Malls",
                    "hospital": "Hospitals",
                    "school": "Schools",
                    "gym": "Gyms",
                    "park": "Parks",
                }

                type_name = type_names.get(search_type, place_type)

                places_list = []
                for place in places_result["results"][:MAPS_MAX_RESULTS]:
                    name = place["name"]
                    rating = place.get("rating")
                    vicinity = place.get("vicinity", "")

                    if rating:
                        places_list.append(f"  - {name} (Rating: {rating})\n    Address: {vicinity}")
                    else:
                        places_list.append(f"  - {name}\n    Address: {vicinity}")

                response = f"""{type_name} near {location} (within {MAPS_SEARCH_RADIUS}m):

{chr(10).join(places_list)}

Found {len(places_result['results'])} results, showing top {min(MAPS_MAX_RESULTS, len(places_result['results']))}."""

                return response

            except Exception as e:
                logger.error("Nearby search failed: %s", e, exc_info=True)
                return f"{TOOL_ERROR_PREFIX} Nearby search failed: {e}"

        return [get_commute_info, get_directions, search_nearby]
