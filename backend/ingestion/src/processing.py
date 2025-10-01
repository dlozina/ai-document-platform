"""
Processing Pipeline Integration for Ingestion Service

Handles integration with OCR, NER, and Embedding services for document processing.
"""

import logging
import httpx
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

from .config import get_settings
from .models import ProcessingStatus, ProcessingJobResponse
from .database import get_db_manager, ProcessingJob

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Document processing pipeline manager."""
    
    def __init__(self):
        self.settings = get_settings()
        self.db_manager = get_db_manager()
    
    async def process_document(
        self, 
        document_id: str, 
        tenant_id: str,
        file_content: bytes,
        filename: str,
        content_type: str,
        pipeline_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process document through the pipeline.
        
        Args:
            document_id: Document identifier
            tenant_id: Tenant identifier
            file_content: File content as bytes
            filename: Original filename
            content_type: MIME content type
            pipeline_config: Processing configuration
            
        Returns:
            Processing result dictionary
        """
        if not pipeline_config:
            pipeline_config = {
                "enable_ocr": self.settings.enable_ocr_processing,
                "enable_ner": self.settings.enable_ner_processing,
                "enable_embedding": self.settings.enable_embedding_processing,
                "ocr_force": False,
                "priority": 1
            }
        
        results = {
            "document_id": document_id,
            "tenant_id": tenant_id,
            "processing_status": ProcessingStatus.PROCESSING,
            "jobs": [],
            "errors": []
        }
        
        # Start processing jobs
        jobs = []
        
        if pipeline_config.get("enable_ocr", False):
            ocr_job = await self._start_ocr_processing(
                document_id, tenant_id, file_content, filename, content_type,
                pipeline_config.get("ocr_force", False)
            )
            if ocr_job:
                jobs.append(ocr_job)
        
        if pipeline_config.get("enable_ner", False):
            ner_job = await self._start_ner_processing(document_id, tenant_id)
            if ner_job:
                jobs.append(ner_job)
        
        if pipeline_config.get("enable_embedding", False):
            embedding_job = await self._start_embedding_processing(document_id, tenant_id)
            if embedding_job:
                jobs.append(embedding_job)
        
        results["jobs"] = jobs
        
        # Update document status
        with self.db_manager.get_session() as session:
            self.db_manager.update_document(session, document_id, {
                "processing_status": ProcessingStatus.PROCESSING,
                "ocr_status": ProcessingStatus.PENDING if pipeline_config.get("enable_ocr") else None,
                "ner_status": ProcessingStatus.PENDING if pipeline_config.get("enable_ner") else None,
                "embedding_status": ProcessingStatus.PENDING if pipeline_config.get("enable_embedding") else None
            })
        
        return results
    
    async def _start_ocr_processing(
        self, 
        document_id: str, 
        tenant_id: str, 
        file_content: bytes,
        filename: str,
        content_type: str,
        force_ocr: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Start OCR processing job."""
        try:
            # Create processing job record
            job_id = str(uuid.uuid4())
            with self.db_manager.get_session() as session:
                job = self.db_manager.create_processing_job(session, {
                    "id": job_id,
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                    "job_type": "ocr",
                    "status": ProcessingStatus.PENDING,
                    "message": "OCR processing queued"
                })
            
            # Start async OCR processing
            asyncio.create_task(self._process_ocr_async(
                job_id, document_id, tenant_id, file_content, filename, content_type, force_ocr
            ))
            
            return {
                "job_id": job_id,
                "job_type": "ocr",
                "status": ProcessingStatus.PENDING,
                "message": "OCR processing started"
            }
            
        except Exception as e:
            logger.error(f"Failed to start OCR processing for {document_id}: {e}")
            return None
    
    async def _process_ocr_async(
        self, 
        job_id: str, 
        document_id: str, 
        tenant_id: str,
        file_content: bytes,
        filename: str,
        content_type: str,
        force_ocr: bool
    ):
        """Process OCR asynchronously."""
        try:
            # Update job status
            with self.db_manager.get_session() as session:
                self.db_manager.update_processing_job(session, job_id, {
                    "status": ProcessingStatus.PROCESSING,
                    "started_at": datetime.utcnow(),
                    "message": "Processing OCR"
                })
            
            # Call OCR service
            async with httpx.AsyncClient(timeout=300.0) as client:
                files = {"file": (filename, file_content, content_type)}
                params = {"force_ocr": force_ocr}
                headers = {
                    "X-Tenant-ID": tenant_id,
                    "X-Document-ID": document_id
                }
                
                response = await client.post(
                    f"{self.settings.ocr_service_url}/extract",
                    files=files,
                    params=params,
                    headers=headers
                )
                
                if response.status_code == 200:
                    ocr_result = response.json()
                    
                    # Update document with OCR results
                    with self.db_manager.get_session() as session:
                        self.db_manager.update_document(session, document_id, {
                            "ocr_text": ocr_result.get("text", ""),
                            "ocr_status": ProcessingStatus.COMPLETED
                        })
                    
                    # Update job status
                    with self.db_manager.get_session() as session:
                        self.db_manager.update_processing_job(session, job_id, {
                            "status": ProcessingStatus.COMPLETED,
                            "completed_at": datetime.utcnow(),
                            "message": f"OCR completed: {len(ocr_result.get('text', ''))} characters extracted"
                        })
                    
                    logger.info(f"OCR processing completed for {document_id}")
                    
                else:
                    raise Exception(f"OCR service returned status {response.status_code}: {response.text}")
                    
        except Exception as e:
            logger.error(f"OCR processing failed for {document_id}: {e}")
            
            # Update job status
            with self.db_manager.get_session() as session:
                self.db_manager.update_processing_job(session, job_id, {
                    "status": ProcessingStatus.FAILED,
                    "completed_at": datetime.utcnow(),
                    "error_message": str(e)
                })
            
            # Update document status
            with self.db_manager.get_session() as session:
                self.db_manager.update_document(session, document_id, {
                    "ocr_status": ProcessingStatus.FAILED
                })
    
    async def _start_ner_processing(self, document_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Start NER processing job."""
        try:
            # Get document OCR text
            with self.db_manager.get_session() as session:
                document = self.db_manager.get_document(session, document_id)
                if not document or not document.ocr_text:
                    logger.warning(f"No OCR text available for NER processing: {document_id}")
                    return None
            
            # Create processing job record
            job_id = str(uuid.uuid4())
            with self.db_manager.get_session() as session:
                job = self.db_manager.create_processing_job(session, {
                    "id": job_id,
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                    "job_type": "ner",
                    "status": ProcessingStatus.PENDING,
                    "message": "NER processing queued"
                })
            
            # Start async NER processing
            asyncio.create_task(self._process_ner_async(job_id, document_id, tenant_id, document.ocr_text))
            
            return {
                "job_id": job_id,
                "job_type": "ner",
                "status": ProcessingStatus.PENDING,
                "message": "NER processing started"
            }
            
        except Exception as e:
            logger.error(f"Failed to start NER processing for {document_id}: {e}")
            return None
    
    async def _process_ner_async(self, job_id: str, document_id: str, tenant_id: str, text: str):
        """Process NER asynchronously."""
        try:
            # Update job status
            with self.db_manager.get_session() as session:
                self.db_manager.update_processing_job(session, job_id, {
                    "status": ProcessingStatus.PROCESSING,
                    "started_at": datetime.utcnow(),
                    "message": "Processing NER"
                })
            
            # Call NER service
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "text": text,
                    "tenant_id": tenant_id,
                    "document_id": document_id
                }
                
                response = await client.post(
                    f"{self.settings.ner_service_url}/extract",
                    json=payload
                )
                
                if response.status_code == 200:
                    ner_result = response.json()
                    
                    # Update document with NER results
                    with self.db_manager.get_session() as session:
                        self.db_manager.update_document(session, document_id, {
                            "ner_entities": ner_result.get("entities", []),
                            "ner_status": ProcessingStatus.COMPLETED
                        })
                    
                    # Update job status
                    with self.db_manager.get_session() as session:
                        self.db_manager.update_processing_job(session, job_id, {
                            "status": ProcessingStatus.COMPLETED,
                            "completed_at": datetime.utcnow(),
                            "message": f"NER completed: {len(ner_result.get('entities', []))} entities found"
                        })
                    
                    logger.info(f"NER processing completed for {document_id}")
                    
                else:
                    raise Exception(f"NER service returned status {response.status_code}: {response.text}")
                    
        except Exception as e:
            logger.error(f"NER processing failed for {document_id}: {e}")
            
            # Update job status
            with self.db_manager.get_session() as session:
                self.db_manager.update_processing_job(session, job_id, {
                    "status": ProcessingStatus.FAILED,
                    "completed_at": datetime.utcnow(),
                    "error_message": str(e)
                })
            
            # Update document status
            with self.db_manager.get_session() as session:
                self.db_manager.update_document(session, document_id, {
                    "ner_status": ProcessingStatus.FAILED
                })
    
    async def _start_embedding_processing(self, document_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Start embedding processing job."""
        try:
            # Get document OCR text
            with self.db_manager.get_session() as session:
                document = self.db_manager.get_document(session, document_id)
                if not document or not document.ocr_text:
                    logger.warning(f"No OCR text available for embedding processing: {document_id}")
                    return None
            
            # Create processing job record
            job_id = str(uuid.uuid4())
            with self.db_manager.get_session() as session:
                job = self.db_manager.create_processing_job(session, {
                    "id": job_id,
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                    "job_type": "embedding",
                    "status": ProcessingStatus.PENDING,
                    "message": "Embedding processing queued"
                })
            
            # Start async embedding processing
            asyncio.create_task(self._process_embedding_async(job_id, document_id, tenant_id, document.ocr_text))
            
            return {
                "job_id": job_id,
                "job_type": "embedding",
                "status": ProcessingStatus.PENDING,
                "message": "Embedding processing started"
            }
            
        except Exception as e:
            logger.error(f"Failed to start embedding processing for {document_id}: {e}")
            return None
    
    async def _process_embedding_async(self, job_id: str, document_id: str, tenant_id: str, text: str):
        """Process embedding asynchronously."""
        try:
            # Update job status
            with self.db_manager.get_session() as session:
                self.db_manager.update_processing_job(session, job_id, {
                    "status": ProcessingStatus.PROCESSING,
                    "started_at": datetime.utcnow(),
                    "message": "Processing embedding"
                })
            
            # Call embedding service
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "text": text,
                    "document_id": document_id
                }
                
                headers = {
                    "X-Tenant-ID": tenant_id,
                    "X-Document-ID": document_id
                }
                
                response = await client.post(
                    f"{self.settings.embedding_service_url}/embed",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    embedding_result = response.json()
                    
                    # Update document with embedding results
                    with self.db_manager.get_session() as session:
                        self.db_manager.update_document(session, document_id, {
                            "embedding_vector": embedding_result.get("embedding", []),
                            "embedding_status": ProcessingStatus.COMPLETED
                        })
                    
                    # Update job status
                    with self.db_manager.get_session() as session:
                        self.db_manager.update_processing_job(session, job_id, {
                            "status": ProcessingStatus.COMPLETED,
                            "completed_at": datetime.utcnow(),
                            "message": f"Embedding completed: {len(embedding_result.get('embedding', []))} dimensions"
                        })
                    
                    logger.info(f"Embedding processing completed for {document_id}")
                    
                else:
                    raise Exception(f"Embedding service returned status {response.status_code}: {response.text}")
                    
        except Exception as e:
            logger.error(f"Embedding processing failed for {document_id}: {e}")
            
            # Update job status
            with self.db_manager.get_session() as session:
                self.db_manager.update_processing_job(session, job_id, {
                    "status": ProcessingStatus.FAILED,
                    "completed_at": datetime.utcnow(),
                    "error_message": str(e)
                })
            
            # Update document status
            with self.db_manager.get_session() as session:
                self.db_manager.update_document(session, document_id, {
                    "embedding_status": ProcessingStatus.FAILED
                })
    
    def get_processing_status(self, document_id: str) -> Dict[str, Any]:
        """Get processing status for a document."""
        with self.db_manager.get_session() as session:
            document = self.db_manager.get_document(session, document_id)
            if not document:
                return {"error": "Document not found"}
            
            # Get processing jobs
            jobs = session.query(ProcessingJob).filter(
                ProcessingJob.document_id == document_id
            ).all()
            
            return {
                "document_id": document_id,
                "processing_status": document.processing_status,
                "ocr_status": document.ocr_status,
                "ner_status": document.ner_status,
                "embedding_status": document.embedding_status,
                "jobs": [
                    {
                        "job_id": job.id,
                        "job_type": job.job_type,
                        "status": job.status,
                        "progress_percentage": job.progress_percentage,
                        "message": job.message,
                        "error_message": job.error_message,
                        "started_at": job.started_at,
                        "completed_at": job.completed_at
                    }
                    for job in jobs
                ]
            }
    
    def check_pipeline_completion(self, document_id: str) -> bool:
        """Check if all pipeline processing is complete."""
        with self.db_manager.get_session() as session:
            document = self.db_manager.get_document(session, document_id)
            if not document:
                return False
            
            # Check if all enabled processing is complete
            statuses = [document.ocr_status, document.ner_status, document.embedding_status]
            non_none_statuses = [s for s in statuses if s is not None]
            
            if not non_none_statuses:
                return True  # No processing enabled
            
            return all(status == ProcessingStatus.COMPLETED for status in non_none_statuses)


# Global processing pipeline instance
processing_pipeline: Optional[ProcessingPipeline] = None


def get_processing_pipeline() -> ProcessingPipeline:
    """Get processing pipeline instance."""
    global processing_pipeline
    if processing_pipeline is None:
        processing_pipeline = ProcessingPipeline()
    return processing_pipeline
