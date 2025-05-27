# type: ignore
import redis
import json
import time
from typing import Optional, Any
from config import settings
from utils.log import setup_logging


logger = setup_logging(__name__)


class RedisClient:
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self._connect()

    def _connect(self) -> bool:
        """Establish Redis connection with retry logic"""
        for attempt in range(settings.REDIS_RETRY_ATTEMPTS):
            try:
                self.client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                self.client.ping()
                logger.info(
                    f"Connected at {settings.REDIS_HOST}:{settings.REDIS_PORT}"
                )
                return True
            except Exception as e:
                logger.warning(
                    f"Redis connection attempt {attempt + 1} failed: {e}"
                )
                if attempt < settings.REDIS_RETRY_ATTEMPTS - 1:
                    time.sleep(settings.REDIS_RETRY_DELAY)

        logger.error("Failed to connect to Redis after all attempts")
        return False

    def publish(self, channel: str, data: Any) -> bool:
        """Publish data to Redis channel"""
        try:
            if not self.client:
                if not self._connect():
                    return False

            message = json.dumps(data) if not isinstance(data, str) else data
            self.client.publish(channel, message)
            return True
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}")
            self.client = None  # Force reconnection on next attempt
            return False

    def subscribe(self, channel: str) -> Optional[redis.client.PubSub]:
        """Subscribe to Redis channel"""
        try:
            if not self.client:
                if not self._connect():
                    return None

            self.pubsub = self.client.pubsub()
            self.pubsub.subscribe(channel)
            logger.info(f"Subscribed to channel: {channel}")
            return self.pubsub
        except Exception as e:
            logger.error(f"Failed to subscribe to {channel}: {e}")
            return None

    def is_healthy(self) -> bool:
        """Check if Redis connection is healthy"""
        try:
            return self.client is not None and self.client.ping()
        except Exception as e:
            logger.error(f"Connection unhealthy: {e}")
            return False


# Global Redis client instance
redis_client = RedisClient()
