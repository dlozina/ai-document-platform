"""
OCR Processing Tasks for Ingestion Service

Handles OCR text extraction from uploaded documents using Celery.
"""

import asyncio
import logging
from typing import Any

import httpx
from celery.exceptions import MaxRetriesExceededError

from ..celery_app import RETRY_POLICIES, celery_app
from ..config import get_settings
from ..database import get_db_manager
from ..models import ProcessingStatus
from ..storage import get_minio_manager

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="src.tasks.ocr_tasks.process_document_ocr",
    queue="ocr_queue",
    priority=9,  # Highest priority
    max_retries=RETRY_POLICIES["ocr"]["max_retries"],
    default_retry_delay=RETRY_POLICIES["ocr"]["countdown"],
)
def process_document_ocr(
    self,
    document_id: str,
    tenant_id: str,
    filename: str,
    content_type: str,
    storage_path: str,
    force_ocr: bool = False,
) -> dict[str, Any]:
    """
    Process document through OCR service.

    Args:
        document_id: Document identifier
        tenant_id: Tenant identifier
        filename: Original filename
        content_type: MIME content type
        storage_path: Path in object storage
        force_ocr: Force OCR even for searchable PDFs

    Returns:
        Processing result dictionary
    """
    task_id = self.request.id
    logger.info(f"Starting OCR processing task {task_id} for document {document_id}")

    try:
        # Update task status in database
        db_manager = get_db_manager()
        with db_manager.get_session() as session:
            # Update document status
            db_manager.update_document(
                session, document_id, {"ocr_status": ProcessingStatus.PROCESSING}
            )

            # Create processing job record
            job = db_manager.create_processing_job(
                session,
                {
                    "id": task_id,
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                    "job_type": "ocr",
                    "status": ProcessingStatus.PROCESSING,
                    "message": "OCR processing started",
                },
            )

        # Download file from MinIO
        minio_manager = get_minio_manager()
        file_content = minio_manager.download_file(tenant_id, storage_path)

        if not file_content:
            raise Exception(f"Failed to download file {storage_path} from storage")

        logger.info(
            f"Downloaded file {filename} ({len(file_content)} bytes) for OCR processing"
        )

        # Call OCR service
        settings = get_settings()

        async def call_ocr_service():
            async with httpx.AsyncClient(timeout=300.0) as client:
                files = {"file": (filename, file_content, content_type)}
                params = {"force_ocr": force_ocr}
                headers = {"X-Tenant-ID": tenant_id, "X-Document-ID": document_id}

                response = await client.post(
                    f"{settings.ocr_service_url}/extract",
                    files=files,
                    params=params,
                    headers=headers,
                )

                if response.status_code != 200:
                    raise Exception(
                        f"OCR service returned status {response.status_code}: {response.text}"
                    )

                return response.json()

        ocr_result = asyncio.run(call_ocr_service())

        # Update document with OCR results
        with db_manager.get_session() as session:
            db_manager.update_document(
                session,
                document_id,
                {
                    "ocr_text": ocr_result.get("text", ""),
                    "ocr_status": ProcessingStatus.COMPLETED,
                },
            )

            # Update processing job
            db_manager.update_processing_job(
                session,
                task_id,
                {
                    "status": ProcessingStatus.COMPLETED,
                    "message": f"OCR completed: {len(ocr_result.get('text', ''))} characters extracted",
                },
            )

        logger.info(f"OCR processing completed for document {document_id}")

        # Trigger next processing steps (NER and Embedding) directly
        try:
            trigger_result = trigger_next_processing.delay(document_id, tenant_id)
            logger.info(
                f"Triggered next processing for document {document_id}: {trigger_result.id}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to trigger next processing for document {document_id}: {e}"
            )

        return {
            "document_id": document_id,
            "task_id": task_id,
            "status": "completed",
            "text_length": len(ocr_result.get("text", "")),
            "confidence": ocr_result.get("confidence"),
            "method": ocr_result.get("method"),
        }

    except Exception as exc:
        logger.error(
            f"OCR processing failed for document {document_id}: {exc}", exc_info=True
        )

        # Update database with error
        try:
            with db_manager.get_session() as session:
                db_manager.update_document(
                    session, document_id, {"ocr_status": ProcessingStatus.FAILED}
                )

                db_manager.update_processing_job(
                    session,
                    task_id,
                    {"status": ProcessingStatus.FAILED, "error_message": str(exc)},
                )
        except Exception as db_exc:
            logger.error(f"Failed to update database with error: {db_exc}")

        # Retry logic
        try:
            retry_countdown = RETRY_POLICIES["ocr"]["countdown"] * (
                2**self.request.retries
            )
            max_countdown = RETRY_POLICIES["ocr"]["max_countdown"]
            countdown = min(retry_countdown, max_countdown)

            logger.info(
                f"Retrying OCR processing for document {document_id} in {countdown} seconds"
            )
            raise self.retry(countdown=countdown, exc=exc)

        except MaxRetriesExceededError:
            logger.error(
                f"Max retries exceeded for OCR processing of document {document_id}"
            )

            # Move to dead letter queue
            move_to_dead_letter.delay(
                document_id=document_id,
                tenant_id=tenant_id,
                task_type="ocr",
                error_message=str(exc),
                task_id=task_id,
            )

            return {
                "document_id": document_id,
                "task_id": task_id,
                "status": "failed",
                "error": str(exc),
                "retries_exceeded": True,
            }


@celery_app.task(
    name="src.tasks.ocr_tasks.trigger_next_processing",
    queue="completion_queue",
    priority=1,
)
def trigger_next_processing(document_id: str, tenant_id: str) -> dict[str, Any]:
    """
    Trigger NER and Embedding processing after OCR completion.

    Args:
        document_id: Document identifier
        tenant_id: Tenant identifier

    Returns:
        Result dictionary
    """
    logger.info(f"Triggering next processing steps for document {document_id}")

    try:
        # Import here to avoid circular imports
        # Use Celery chain to ensure NER completes before Embedding starts
        from celery import chain

        from .embedding_tasks import process_document_embedding
        from .ner_tasks import process_document_ner

        # Create a chain: NER -> Embedding
        processing_chain = chain(
            process_document_ner.s(document_id, tenant_id),
            process_document_embedding.s(),
        )

        # Execute the chain
        chain_result = processing_chain.apply_async()
        logger.info(
            f"Started sequential processing chain for document {document_id}: NER -> Embedding"
        )

        return {
            "document_id": document_id,
            "chain_id": chain_result.id,
            "status": "triggered",
            "message": "Sequential processing chain started: NER -> Embedding",
        }

    except Exception as exc:
        logger.error(
            f"Failed to trigger next processing for document {document_id}: {exc}"
        )
        return {"document_id": document_id, "status": "failed", "error": str(exc)}


@celery_app.task(
    name="src.tasks.ocr_tasks.move_to_dead_letter",
    queue="dead_letter_queue",
    priority=1,
)
def move_to_dead_letter(
    document_id: str, tenant_id: str, task_type: str, error_message: str, task_id: str
) -> dict[str, Any]:
    """
    Move failed task to dead letter queue for manual inspection.

    Args:
        document_id: Document identifier
        tenant_id: Tenant identifier
        task_type: Type of task that failed
        error_message: Error message
        task_id: Celery task ID

    Returns:
        Result dictionary
    """
    logger.error(
        f"Moving document {document_id} to dead letter queue. Task: {task_type}, Error: {error_message}"
    )

    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as session:
            # Update document status
            db_manager.update_document(
                session, document_id, {"processing_status": ProcessingStatus.FAILED}
            )

            # Update processing job
            db_manager.update_processing_job(
                session,
                task_id,
                {
                    "status": ProcessingStatus.FAILED,
                    "error_message": f"DEAD_LETTER: {error_message}",
                },
            )

        return {
            "document_id": document_id,
            "task_type": task_type,
            "status": "moved_to_dead_letter",
            "error_message": error_message,
        }

    except Exception as exc:
        logger.error(
            f"Failed to move document {document_id} to dead letter queue: {exc}"
        )
        return {"document_id": document_id, "status": "failed", "error": str(exc)}
