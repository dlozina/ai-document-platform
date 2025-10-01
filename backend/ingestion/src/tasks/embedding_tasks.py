"""
Embedding Processing Tasks for Ingestion Service

Handles document embedding generation using Celery.
"""

import logging
import asyncio
import httpx
from celery import current_task
from celery.exceptions import Retry, MaxRetriesExceededError
from typing import Dict, Any

from ..celery_app import celery_app, RETRY_POLICIES
from ..database import get_db_manager
from ..models import ProcessingStatus
from ..config import get_settings

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="src.tasks.embedding_tasks.process_document_embedding",
    queue="embedding_queue",
    priority=5,  # Medium priority
    max_retries=RETRY_POLICIES["embedding"]["max_retries"],
    default_retry_delay=RETRY_POLICIES["embedding"]["countdown"],
)
def process_document_embedding(
    self,
    document_id: str,
    tenant_id: str
) -> Dict[str, Any]:
    """
    Process document through Embedding service.
    
    Args:
        document_id: Document identifier
        tenant_id: Tenant identifier
        
    Returns:
        Processing result dictionary
    """
    task_id = self.request.id
    logger.info(f"Starting Embedding processing task {task_id} for document {document_id}")
    
    try:
        # Get document and check if OCR text is available
        db_manager = get_db_manager()
        ocr_text = None
        
        with db_manager.get_session() as session:
            document = db_manager.get_document(session, document_id)
            
            if not document:
                raise Exception(f"Document {document_id} not found")
            
            if not document.ocr_text:
                raise Exception(f"No OCR text available for document {document_id}")
            
            # Extract all required attributes while session is active
            ocr_text = document.ocr_text
            filename = document.filename
            content_type = document.content_type
            file_type = document.file_type
            file_size_bytes = document.file_size_bytes
            upload_timestamp = document.upload_timestamp
            created_by = document.created_by
            processing_status = document.processing_status
            tags = document.tags
            description = document.description
            
            # Update document status
            db_manager.update_document(session, document_id, {
                "embedding_status": ProcessingStatus.PROCESSING
            })
            
            # Create processing job record
            job = db_manager.create_processing_job(session, {
                "id": task_id,
                "document_id": document_id,
                "tenant_id": tenant_id,
                "job_type": "embedding",
                "status": ProcessingStatus.PROCESSING,
                "message": "Embedding processing started"
            })
        
        # Call Embedding service
        settings = get_settings()
        
        async def call_embedding_service():
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "text": ocr_text,
                    "document_id": document_id,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "file_type": file_type,
                        "file_size_bytes": file_size_bytes,
                        "upload_timestamp": upload_timestamp.isoformat() if upload_timestamp else None,
                        "created_by": created_by,
                        "processing_status": processing_status,
                        "tags": tags or [],
                        "description": description
                    }
                }
                
                headers = {
                    "X-Tenant-ID": tenant_id,
                    "X-Document-ID": document_id
                }
                
                response = await client.post(
                    f"{settings.embedding_service_url}/embed",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code != 200:
                    raise Exception(f"Embedding service returned status {response.status_code}: {response.text}")
                
                return response.json()
        
        embedding_result = asyncio.run(call_embedding_service())
        
        # Update document with embedding results
        with db_manager.get_session() as session:
            db_manager.update_document(session, document_id, {
                "embedding_vector": embedding_result.get("embedding", []),
                "embedding_status": ProcessingStatus.COMPLETED
            })
            
            # Update processing job
            db_manager.update_processing_job(session, task_id, {
                "status": ProcessingStatus.COMPLETED,
                "message": f"Embedding completed: {len(embedding_result.get('embedding', []))} dimensions"
            })
        
        logger.info(f"Embedding processing completed for document {document_id}")
        
        # Check if all processing is complete
        check_processing_completion.delay(document_id, tenant_id)
        
        return {
            "document_id": document_id,
            "task_id": task_id,
            "status": "completed",
            "embedding_dimensions": len(embedding_result.get("embedding", [])),
            "embedding": embedding_result.get("embedding", [])
        }
        
    except Exception as exc:
        logger.error(f"Embedding processing failed for document {document_id}: {exc}", exc_info=True)
        
        # Update database with error
        try:
            with db_manager.get_session() as session:
                db_manager.update_document(session, document_id, {
                    "embedding_status": ProcessingStatus.FAILED
                })
                
                db_manager.update_processing_job(session, task_id, {
                    "status": ProcessingStatus.FAILED,
                    "error_message": str(exc)
                })
        except Exception as db_exc:
            logger.error(f"Failed to update database with error: {db_exc}")
        
        # Retry logic
        try:
            retry_countdown = RETRY_POLICIES["embedding"]["countdown"] * (2 ** self.request.retries)
            max_countdown = RETRY_POLICIES["embedding"]["max_countdown"]
            countdown = min(retry_countdown, max_countdown)
            
            logger.info(f"Retrying Embedding processing for document {document_id} in {countdown} seconds")
            raise self.retry(countdown=countdown, exc=exc)
            
        except MaxRetriesExceededError:
            logger.error(f"Max retries exceeded for Embedding processing of document {document_id}")
            
            # Move to dead letter queue
            from .ocr_tasks import move_to_dead_letter
            move_to_dead_letter.delay(
                document_id=document_id,
                tenant_id=tenant_id,
                task_type="embedding",
                error_message=str(exc),
                task_id=task_id
            )
            
            return {
                "document_id": document_id,
                "task_id": task_id,
                "status": "failed",
                "error": str(exc),
                "retries_exceeded": True
            }


@celery_app.task(
    name="src.tasks.embedding_tasks.check_processing_completion",
    queue="completion_queue",
    priority=1
)
def check_processing_completion(document_id: str, tenant_id: str) -> Dict[str, Any]:
    """
    Check if all processing steps are complete and update document status.
    
    Args:
        document_id: Document identifier
        tenant_id: Tenant identifier
        
    Returns:
        Result dictionary
    """
    logger.info(f"Checking processing completion for document {document_id}")
    
    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as session:
            document = db_manager.get_document(session, document_id)
            
            if not document:
                return {"document_id": document_id, "status": "document_not_found"}
            
            # Extract required attributes while session is active
            ocr_status = document.ocr_status
            ner_status = document.ner_status
            embedding_status = document.embedding_status
            
            # Check if all enabled processing is complete
            ocr_complete = ocr_status == ProcessingStatus.COMPLETED
            ner_complete = ner_status == ProcessingStatus.COMPLETED or ner_status is None
            embedding_complete = embedding_status == ProcessingStatus.COMPLETED or embedding_status is None
            
            # Check if any processing failed
            any_failed = (
                ocr_status == ProcessingStatus.FAILED or
                ner_status == ProcessingStatus.FAILED or
                embedding_status == ProcessingStatus.FAILED
            )
            
            if any_failed:
                # Update document status to failed
                db_manager.update_document(session, document_id, {
                    "processing_status": ProcessingStatus.FAILED
                })
                
                logger.warning(f"Document {document_id} processing failed")
                return {
                    "document_id": document_id,
                    "status": "failed",
                    "ocr_status": ocr_status,
                    "ner_status": ner_status,
                    "embedding_status": embedding_status
                }
            
            elif ocr_complete and ner_complete and embedding_complete:
                # All processing complete
                db_manager.update_document(session, document_id, {
                    "processing_status": ProcessingStatus.COMPLETED
                })
                
                logger.info(f"Document {document_id} processing completed successfully")
                return {
                    "document_id": document_id,
                    "status": "completed",
                    "ocr_status": ocr_status,
                    "ner_status": ner_status,
                    "embedding_status": embedding_status
                }
            else:
                # Still processing
                logger.info(f"Document {document_id} still processing...")
                return {
                    "document_id": document_id,
                    "status": "processing",
                    "ocr_status": ocr_status,
                    "ner_status": ner_status,
                    "embedding_status": embedding_status
                }
        
    except Exception as exc:
        logger.error(f"Failed to check processing completion for document {document_id}: {exc}")
        return {
            "document_id": document_id,
            "status": "error",
            "error": str(exc)
        }
