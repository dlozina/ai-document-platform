"""
Completion Tasks for Ingestion Service

Handles final processing completion and cleanup tasks.
"""

import logging
from typing import Dict, Any

from ..celery_app import celery_app
from ..database import get_db_manager, Document, ProcessingJob
from ..models import ProcessingStatus
from datetime import datetime

logger = logging.getLogger(__name__)


@celery_app.task(
    name="src.tasks.completion_tasks.finalize_document_processing",
    queue="completion_queue",
    priority=1
)
def finalize_document_processing(document_id: str, tenant_id: str) -> Dict[str, Any]:
    """
    Finalize document processing and perform cleanup tasks.
    
    Args:
        document_id: Document identifier
        tenant_id: Tenant identifier
        
    Returns:
        Result dictionary
    """
    logger.info(f"Finalizing processing for document {document_id}")
    
    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as session:
            document = db_manager.get_document(session, document_id)
            
            if not document:
                return {"document_id": document_id, "status": "document_not_found"}
            
            # Update final processing status
            db_manager.update_document(session, document_id, {
                "processing_status": ProcessingStatus.COMPLETED,
                "updated_at": datetime.utcnow()
            })
            
            # Log processing completion metrics
            processing_time = (datetime.utcnow() - document.created_at).total_seconds()
            
            logger.info(
                f"Document {document_id} processing completed successfully. "
                f"Processing time: {processing_time:.2f} seconds, "
                f"File size: {document.file_size_bytes} bytes, "
                f"OCR text length: {len(document.ocr_text or '')} characters"
            )
        
        return {
            "document_id": document_id,
            "status": "completed",
            "processing_time_seconds": processing_time,
            "file_size_bytes": document.file_size_bytes,
            "ocr_text_length": len(document.ocr_text or ""),
            "ner_entities_count": len(document.ner_entities or []),
            "embedding_dimensions": len(document.embedding_vector or [])
        }
        
    except Exception as exc:
        logger.error(f"Failed to finalize processing for document {document_id}: {exc}")
        return {
            "document_id": document_id,
            "status": "error",
            "error": str(exc)
        }


@celery_app.task(
    name="src.tasks.completion_tasks.cleanup_failed_processing",
    queue="completion_queue",
    priority=1
)
def cleanup_failed_processing(document_id: str, tenant_id: str) -> Dict[str, Any]:
    """
    Cleanup failed processing jobs and update document status.
    
    Args:
        document_id: Document identifier
        tenant_id: Tenant identifier
        
    Returns:
        Result dictionary
    """
    logger.info(f"Cleaning up failed processing for document {document_id}")
    
    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as session:
            document = db_manager.get_document(session, document_id)
            
            if not document:
                return {"document_id": document_id, "status": "document_not_found"}
            
            # Update document status to failed
            db_manager.update_document(session, document_id, {
                "processing_status": ProcessingStatus.FAILED,
                "updated_at": datetime.utcnow()
            })
            
            # Get all failed processing jobs
            failed_jobs = session.query(ProcessingJob).filter(
                ProcessingJob.document_id == document_id,
                ProcessingJob.status == ProcessingStatus.FAILED
            ).all()
            
            logger.warning(
                f"Document {document_id} processing failed. "
                f"Failed jobs: {len(failed_jobs)}"
            )
        
        return {
            "document_id": document_id,
            "status": "failed",
            "failed_jobs_count": len(failed_jobs),
            "failed_jobs": [
                {
                    "job_id": job.id,
                    "job_type": job.job_type,
                    "error_message": job.error_message
                }
                for job in failed_jobs
            ]
        }
        
    except Exception as exc:
        logger.error(f"Failed to cleanup failed processing for document {document_id}: {exc}")
        return {
            "document_id": document_id,
            "status": "error",
            "error": str(exc)
        }


@celery_app.task(
    name="src.tasks.completion_tasks.generate_processing_report",
    queue="completion_queue",
    priority=1
)
def generate_processing_report(tenant_id: str, date_from: str = None, date_to: str = None) -> Dict[str, Any]:
    """
    Generate processing report for a tenant.
    
    Args:
        tenant_id: Tenant identifier
        date_from: Start date for report (ISO format)
        date_to: End date for report (ISO format)
        
    Returns:
        Report dictionary
    """
    logger.info(f"Generating processing report for tenant {tenant_id}")
    
    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as session:
            # Build query
            query = session.query(Document).filter(Document.tenant_id == tenant_id)
            
            if date_from:
                query = query.filter(Document.created_at >= datetime.fromisoformat(date_from))
            
            if date_to:
                query = query.filter(Document.created_at <= datetime.fromisoformat(date_to))
            
            documents = query.all()
            
            # Calculate statistics
            total_documents = len(documents)
            completed_documents = len([d for d in documents if d.processing_status == ProcessingStatus.COMPLETED])
            failed_documents = len([d for d in documents if d.processing_status == ProcessingStatus.FAILED])
            processing_documents = len([d for d in documents if d.processing_status == ProcessingStatus.PROCESSING])
            
            # Calculate processing times
            completed_docs = [d for d in documents if d.processing_status == ProcessingStatus.COMPLETED]
            processing_times = []
            
            for doc in completed_docs:
                if doc.updated_at and doc.created_at:
                    processing_time = (doc.updated_at - doc.created_at).total_seconds()
                    processing_times.append(processing_time)
            
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            # Calculate file size statistics
            file_sizes = [d.file_size_bytes for d in documents]
            total_size = sum(file_sizes)
            avg_size = total_size / len(file_sizes) if file_sizes else 0
            
            report = {
                "tenant_id": tenant_id,
                "period": {
                    "from": date_from,
                    "to": date_to
                },
                "statistics": {
                    "total_documents": total_documents,
                    "completed_documents": completed_documents,
                    "failed_documents": failed_documents,
                    "processing_documents": processing_documents,
                    "success_rate": (completed_documents / total_documents * 100) if total_documents > 0 else 0
                },
                "performance": {
                    "average_processing_time_seconds": avg_processing_time,
                    "total_size_bytes": total_size,
                    "average_size_bytes": avg_size
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Generated processing report for tenant {tenant_id}: {completed_documents}/{total_documents} completed")
            
            return report
        
    except Exception as exc:
        logger.error(f"Failed to generate processing report for tenant {tenant_id}: {exc}")
        return {
            "tenant_id": tenant_id,
            "status": "error",
            "error": str(exc)
        }
