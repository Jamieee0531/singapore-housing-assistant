"""
Redis manager for long-term user memory.

Two storage patterns:
- Hash: current user preferences (fast read, always up-to-date)
- Sorted Set: area exploration history (time-ordered, for recall)
"""

import logging
import time
from typing import Optional, List

import redis
from pydantic import BaseModel

from src.config import REDIS_HOST, REDIS_PORT, REDIS_DB

logger = logging.getLogger(__name__)


class UserProfile(BaseModel):
    """
    User preference profile extracted from conversations.
    All fields are Optional — they get filled gradually as the user reveals preferences.
    """
    school: Optional[str] = None
    budget_range: Optional[str] = None
    preferred_area: Optional[str] = None
    rental_type: Optional[str] = None          # "合租" / "整租"
    room_type: Optional[str] = None            # "主人房" / "客人房"
    move_in_date: Optional[str] = None
    transport_requirement: Optional[str] = None
    environment_preference: Optional[str] = None  # "安静" / "热闹"


class RedisManager:
    """
    Manages Redis connection, user profile CRUD, and area exploration history.

    Storage layout:
        user_profile:{user_id}   (Hash)       — current preferences
        area_history:{user_id}   (Sorted Set) — explored areas with timestamps
    """

    def __init__(self):
        self._client: Optional[redis.Redis] = None

    def connect(self) -> None:
        """Establish connection to Redis."""
        self._client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        self._client.ping()
        logger.info("Connected to Redis at %s:%s", REDIS_HOST, REDIS_PORT)

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self.connect()
        return self._client

    # =========================================================================
    # Key helpers
    # =========================================================================

    def _profile_key(self, user_id: str) -> str:
        return f"user_profile:{user_id}"

    def _area_history_key(self, user_id: str) -> str:
        return f"area_history:{user_id}"

    # =========================================================================
    # User Profile (Hash) — current preferences
    # =========================================================================

    def load_profile(self, user_id: str) -> UserProfile:
        """
        Load user profile from Redis.
        Returns empty UserProfile if no data exists.
        """
        data = self.client.hgetall(self._profile_key(user_id))
        if not data:
            logger.info("No existing profile for user %s, returning empty profile", user_id)
            return UserProfile()

        logger.info("Loaded profile for user %s: %s", user_id, data)
        return UserProfile(**data)

    def save_profile(self, user_id: str, profile: UserProfile) -> None:
        """
        Save user profile to Redis.
        Only writes non-None fields to avoid overwriting with empty values.
        """
        data = {k: v for k, v in profile.model_dump().items() if v is not None}
        if not data:
            return

        self.client.hset(self._profile_key(user_id), mapping=data)
        logger.info("Saved profile for user %s: %s", user_id, data)

    def update_profile(self, user_id: str, new_preferences: dict) -> UserProfile:
        """
        Merge new preferences into existing profile.
        If preferred_area changes, also records it in area history.
        Uses pipeline to ensure both writes succeed together.
        Returns the updated profile.
        """
        existing = self.load_profile(user_id)
        existing_data = existing.model_dump()

        new_area = new_preferences.get("preferred_area")
        old_area = existing_data.get("preferred_area")

        for key, value in new_preferences.items():
            if value is not None and key in existing_data:
                existing_data[key] = value

        updated = UserProfile(**existing_data)

        # Pipeline: update Hash + append area history atomically
        pipe = self.client.pipeline()

        profile_data = {k: v for k, v in updated.model_dump().items() if v is not None}
        if profile_data:
            pipe.hset(self._profile_key(user_id), mapping=profile_data)

        if new_area and new_area != old_area:
            pipe.zadd(self._area_history_key(user_id), {new_area: time.time()})

        pipe.execute()

        logger.info("Updated profile for user %s: %s", user_id, profile_data)
        return updated

    def delete_profile(self, user_id: str) -> None:
        """Delete a user's profile and area history from Redis."""
        pipe = self.client.pipeline()
        pipe.delete(self._profile_key(user_id))
        pipe.delete(self._area_history_key(user_id))
        pipe.execute()
        logger.info("Deleted all data for user %s", user_id)

    # =========================================================================
    # Area History (Sorted Set) — exploration tracking
    # =========================================================================

    def get_area_history(self, user_id: str) -> List[str]:
        """
        Get all explored areas, newest first.
        Default read: only called when user asks about past areas.
        """
        areas = self.client.zrevrange(self._area_history_key(user_id), 0, -1)
        return areas

    def get_latest_area(self, user_id: str) -> Optional[str]:
        """Get the most recently explored area."""
        areas = self.client.zrevrange(self._area_history_key(user_id), 0, 0)
        return areas[0] if areas else None
