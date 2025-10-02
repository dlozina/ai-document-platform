"""
Redis client for event publishing and consumption
"""

import logging
import json
from typing import Dict, Any, Optional, List
import redis
from .config import get_settings

logger = logging.getLogger(__name__)


class RedisEventClient:
    """Redis client for event-driven architecture."""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = redis.Redis(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            db=self.settings.redis_db,
            decode_responses=True
        )
    
    def publish_event(self, stream_name: str, event_data: Dict[str, Any]) -> str:
        """
        Publish an event to a Redis stream.
        
        Args:
            stream_name: Name of the Redis stream
            event_data: Event data dictionary
            
        Returns:
            Message ID of the published event
        """
        try:
            # Convert any non-string values to JSON strings
            processed_data = {}
            for key, value in event_data.items():
                if isinstance(value, (dict, list)):
                    processed_data[key] = json.dumps(value)
                else:
                    processed_data[key] = str(value)
            
            message_id = self.redis_client.xadd(stream_name, processed_data)
            logger.info(f"Published event to {stream_name}: {message_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to publish event to {stream_name}: {e}")
            raise
    
    def consume_events(self, stream_name: str, consumer_group: str, consumer_name: str, 
                      count: int = 10, block: int = 1000) -> List[Dict[str, Any]]:
        """
        Consume events from a Redis stream using consumer groups.
        
        Args:
            stream_name: Name of the Redis stream
            consumer_group: Consumer group name
            consumer_name: Consumer name
            count: Maximum number of messages to read
            block: Block time in milliseconds
            
        Returns:
            List of event data dictionaries
        """
        try:
            # Ensure consumer group exists
            try:
                self.redis_client.xgroup_create(stream_name, consumer_group, id='0', mkstream=True)
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
            
            # Read messages
            messages = self.redis_client.xreadgroup(
                consumer_group, 
                consumer_name, 
                {stream_name: '>'}, 
                count=count, 
                block=block
            )
            
            events = []
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    # Parse JSON fields back to objects
                    event_data = {}
                    for key, value in fields.items():
                        try:
                            event_data[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            event_data[key] = value
                    
                    event_data['_message_id'] = msg_id
                    events.append(event_data)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to consume events from {stream_name}: {e}")
            raise
    
    def acknowledge_event(self, stream_name: str, consumer_group: str, message_id: str):
        """
        Acknowledge processing of an event.
        
        Args:
            stream_name: Name of the Redis stream
            consumer_group: Consumer group name
            message_id: Message ID to acknowledge
        """
        try:
            self.redis_client.xack(stream_name, consumer_group, message_id)
            logger.debug(f"Acknowledged event {message_id} in {stream_name}")
        except Exception as e:
            logger.error(f"Failed to acknowledge event {message_id}: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check Redis connection health."""
        try:
            self.redis_client.ping()
            return {"status": "healthy", "redis_available": True}
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {"status": "unhealthy", "redis_available": False, "error": str(e)}


# Global Redis client instance
redis_client = RedisEventClient()
