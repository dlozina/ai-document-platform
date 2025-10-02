"""
Event consumers for processing NER completion events
"""

import logging
import asyncio
import httpx
from typing import Dict, Any
from celery import current_task
from celery.exceptions import Retry, MaxRetriesExceededError

from ..celery_app import celery_app, RETRY_POLICIES
from ..database import get_db_manager
from ..config import get_settings
from ..redis_client import redis_client

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="src.tasks.ner_event_consumers.process_ner_completion_event",
    queue="ner_event_queue",
    priority=3,
    max_retries=RETRY_POLICIES["ner"]["max_retries"],
    default_retry_delay=RETRY_POLICIES["ner"]["countdown"],
)
def process_ner_completion_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process NER completion event and update Qdrant metadata.
    
    Args:
        event_data: Event data containing NER completion information
        
    Returns:
        Result dictionary
    """
    document_id = event_data.get("document_id")
    tenant_id = event_data.get("tenant_id")
    ner_entities = event_data.get("ner_entities", [])
    
    if not document_id or not tenant_id:
        logger.error("Missing required fields in NER completion event")
        return {"status": "failed", "error": "Missing required fields"}
    
    logger.info(f"Processing NER completion event for document {document_id}")
    
    try:
        # Update Qdrant with NER entities
        async def update_qdrant_metadata():
            settings = get_settings()
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get current document metadata from database
                db_manager = get_db_manager()
                with db_manager.get_session() as session:
                    document = db_manager.get_document(session, document_id)
                    if not document:
                        logger.warning(f"Document {document_id} not found in database")
                        return
                    
                    # Prepare metadata for Qdrant update
                    metadata = {
                        "filename": document.filename,
                        "content_type": document.content_type,
                        "file_type": document.file_type,
                        "file_size_bytes": document.file_size_bytes,
                        "upload_timestamp": document.upload_timestamp.isoformat() if document.upload_timestamp else None,
                        "created_by": document.created_by,
                        "processing_status": document.processing_status,
                        "ocr_status": document.ocr_status,
                        "ner_status": document.ner_status,
                        "embedding_status": document.embedding_status,
                        "tags": document.tags or [],
                        "description": document.description,
                        "ner_entities": document.ner_entities or []
                    }
                
                payload = {
                    "document_id": document_id,
                    "metadata": metadata
                }
                
                headers = {
                    "X-Tenant-ID": tenant_id
                }
                
                response = await client.put(
                    f"{settings.embedding_service_url}/update-metadata",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    logger.info(f"Updated Qdrant metadata for document {document_id}")
                else:
                    raise Exception(f"Failed to update Qdrant metadata: {response.status_code} - {response.text}")
        
        asyncio.run(update_qdrant_metadata())
        
        logger.info(f"Successfully processed NER completion event for document {document_id}")
        return {
            "document_id": document_id,
            "status": "completed",
            "ner_entities_count": len(ner_entities)
        }
        
    except Exception as exc:
        logger.error(f"Failed to process NER completion event for document {document_id}: {exc}")
        
        # Retry logic
        try:
            retry_countdown = RETRY_POLICIES["ner"]["countdown"] * (2 ** self.request.retries)
            max_countdown = RETRY_POLICIES["ner"]["max_countdown"]
            countdown = min(retry_countdown, max_countdown)
            
            logger.info(f"Retrying NER event processing for document {document_id} in {countdown} seconds")
            raise self.retry(countdown=countdown, exc=exc)
            
        except MaxRetriesExceededError:
            logger.error(f"Max retries exceeded for NER event processing of document {document_id}")
            return {
                "document_id": document_id,
                "status": "failed",
                "error": str(exc),
                "retries_exceeded": True
            }


@celery_app.task(
    name="src.tasks.ner_event_consumers.consume_ner_events",
    queue="ner_event_queue",
    priority=1
)
def consume_ner_events() -> Dict[str, Any]:
    """
    Consume NER completion events from Redis stream.
    
    Returns:
        Result dictionary with processing statistics
    """
    logger.info("Starting NER event consumption")
    
    try:
        # Consume events from Redis stream
        events = redis_client.consume_events(
            stream_name="ner_events",
            consumer_group="ner_processors",
            consumer_name="ner_consumer_1",
            count=10,
            block=1000
        )
        
        processed_count = 0
        failed_count = 0
        
        for event_data in events:
            try:
                # Process the event
                result = process_ner_completion_event.delay(event_data)
                processed_count += 1
                logger.info(f"Queued NER event processing for document {event_data.get('document_id')}")
                
            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to queue NER event processing: {e}")
        
        logger.info(f"NER event consumption completed: {processed_count} processed, {failed_count} failed")
        return {
            "status": "completed",
            "processed_count": processed_count,
            "failed_count": failed_count
        }
        
    except Exception as e:
        logger.error(f"NER event consumption failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


@celery_app.task(
    name="src.tasks.ner_event_consumers.start_ner_event_consumer",
    queue="ner_event_queue",
    priority=1
)
def start_ner_event_consumer() -> Dict[str, Any]:
    """
    Start continuous NER event consumption.
    This task runs continuously to process NER events.
    
    Returns:
        Result dictionary
    """
    logger.info("Starting continuous NER event consumer")
    
    try:
        # This task will run continuously
        # In production, you might want to use a separate consumer process
        while True:
            result = consume_ner_events.delay()
            logger.info(f"NER event consumption cycle completed: {result}")
            
            # Small delay to prevent excessive CPU usage
            import time
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"NER event consumer failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }
