"""
Ingestion Service API

FastAPI application that handles file uploads, metadata storage, and processing pipeline integration.
"""

import logging
import time
import json
import io
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Query, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from .config import get_settings
from .models import (
    UploadResponse, BatchUploadResponse, DocumentListResponse, 
    HealthResponse, ErrorResponse, DocumentMetadata, DocumentSearchRequest,
    DocumentUpdateRequest, ProcessingJobResponse, TenantQuota,
    ProcessingPipelineConfig, UploadProgress
)
from .database import get_db_manager, get_db_session, ProcessingJob
from .storage import get_minio_manager
from .celery_app import celery_app
from .tasks.ocr_tasks import process_document_ocr
from .utils import (
    FileValidator, TenantQuotaChecker, calculate_file_hash, 
    detect_file_type, generate_storage_path, generate_document_id,
    generate_batch_id, calculate_retention_date, extract_file_metadata,
    log_upload_stats
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
settings = get_settings()
db_manager = get_db_manager()
minio_manager = get_minio_manager()
file_validator = FileValidator(settings.max_file_size_bytes, settings.supported_file_types)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    logger.info("Starting Ingestion Service...")
    
    # Test database connection
    try:
        with db_manager.get_session() as session:
            session.execute(text("SELECT 1"))
        logger.info("Database connection successful")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
    
    # Test MinIO connection
    if minio_manager.test_connection():
        logger.info("MinIO connection successful")
    else:
        logger.error("MinIO connection failed")
    
    # Test Redis connection
    try:
        celery_app.control.inspect().ping()
        logger.info("Redis/Celery connection successful")
    except Exception as e:
        logger.error(f"Redis/Celery connection failed: {e}")
    
    logger.info("Ingestion Service ready")
    
    yield
    
    logger.info("Shutting down Ingestion Service...")


# Create FastAPI app
app = FastAPI(
    title="Ingestion Service",
    description="""
    ## Multi-tenant Document Ingestion Service
    
    Production-ready service for uploading, storing, and processing documents with:
    - **Multi-tenant support** with isolated storage and quotas
    - **PostgreSQL metadata storage** for structured data
    - **MinIO object storage** for file persistence
    - **Processing pipeline integration** with OCR, NER, and Embedding services
    - **File validation and security** features
    - **Batch upload support** for multiple files
    
    ### Features:
    - **File Upload**: Single and batch file uploads
    - **Metadata Management**: Rich document metadata storage
    - **Processing Pipeline**: Automatic OCR, NER, and embedding generation
    - **Multi-tenant**: Tenant isolation and quota management
    - **Search & Discovery**: Document search and filtering
    - **Security**: File validation, virus scanning (optional)
    
    ### Supported File Types:
    - **PDFs**: `.pdf` (both searchable and scanned)
    - **Images**: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.gif`
    - **Documents**: `.doc`, `.docx`
    - **Text**: `.txt`
    
    ### Multi-tenant Headers:
    - `X-Tenant-ID`: Required tenant identifier
    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "Ingestion Service Team",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": "http://localhost:8003",
            "description": "Development server"
        },
        {
            "url": "https://api.example.com",
            "description": "Production server"
        }
    ]
)

# Configure JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

app.json_encoder = DateTimeEncoder


# Dependency functions
def get_tenant_id(tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")) -> str:
    """Extract and validate tenant ID from headers."""
    if settings.require_tenant_header and not tenant_id:
        raise HTTPException(
            status_code=400,
            detail="X-Tenant-ID header is required"
        )
    
    if not tenant_id:
        tenant_id = settings.default_tenant_id or "default"
    
    return tenant_id


def get_db():
    """Get database session."""
    return db_manager.get_session()


def parse_pipeline_config(pipeline_config_json: Optional[str] = None) -> Optional[ProcessingPipelineConfig]:
    """Parse pipeline config from JSON string."""
    if not pipeline_config_json:
        return None
    
    try:
        config_data = json.loads(pipeline_config_json)
        return ProcessingPipelineConfig(**config_data)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pipeline_config JSON: {str(e)}"
        )


# API Endpoints

@app.post("/test-upload")
async def test_upload(
    file: UploadFile = File(..., description="File to upload"),
    tenant_id: str = Depends(get_tenant_id),
    pipeline_config_json: Optional[str] = Form(None, description="Processing pipeline configuration as JSON string"),
):
    """Test endpoint to verify pipeline_config parsing."""
    try:
        # Parse pipeline config
        pipeline_config = parse_pipeline_config(pipeline_config_json)
        
        return {
            "message": "Test successful",
            "filename": file.filename,
            "tenant_id": tenant_id,
            "pipeline_config": pipeline_config.model_dump() if pipeline_config else None,
            "pipeline_config_raw": pipeline_config_json
        }
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and dependencies availability.
    """
    try:
        # Test database connection
        with db_manager.get_session() as session:
            session.execute(text("SELECT 1"))
        database_connected = True
    except Exception:
        database_connected = False
    
    # Test MinIO connection
    minio_connected = minio_manager.test_connection()
    
    # Test Redis connection
    try:
        celery_app.control.inspect().ping()
        redis_connected = True
    except Exception:
        redis_connected = False
    
    dependencies = {
        "database": database_connected,
        "minio": minio_connected,
        "redis": redis_connected,
        "ocr_service": True,  # TODO: Add actual health check
        "ner_service": True,   # TODO: Add actual health check
        "embedding_service": True  # TODO: Add actual health check
    }
    
    overall_status = "healthy" if all(dependencies.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        service="ingestion-service",
        version="1.0.0",
        database_connected=database_connected,
        minio_connected=minio_connected,
        dependencies=dependencies
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(..., description="File to upload"),
    tenant_id: str = Depends(get_tenant_id),
    pipeline_config_json: Optional[str] = Form(None, description="Processing pipeline configuration as JSON string"),
    tags: Optional[List[str]] = Query(None, description="Document tags"),
    description: Optional[str] = Query(None, description="Document description"),
    retention_days: Optional[int] = Query(None, description="Retention period in days"),
    db_session = Depends(get_db)
):
    """
    Upload a single file.
    
    **Parameters:**
    - **file**: The file to upload (required, multipart/form-data)
    - **X-Tenant-ID**: Tenant identifier (required header)
    - **pipeline_config_json**: Processing pipeline configuration as JSON string (optional, form field)
    - **tags**: Document tags (optional, query parameter)
    - **description**: Document description (optional, query parameter)
    - **retention_days**: Retention period in days (optional, query parameter)
    
    **Returns:**
    - Upload response with document metadata
    
    **Note:** The pipeline_config_json should be sent as a form field containing JSON string.
    """
    start_time = time.time()
    
    try:
        # Parse pipeline config
        pipeline_config = parse_pipeline_config(pipeline_config_json)
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        logger.info(
            f"Upload request: {file.filename} "
            f"(size: {file_size / 1024:.2f}KB, tenant: {tenant_id})"
        )
        
        # Validate file
        validation_result = file_validator.validate(file.filename, content)
        if not validation_result["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"File validation failed: {'; '.join(validation_result['errors'])}"
            )
        
        # Check for duplicate files
        file_hash = calculate_file_hash(content)
        existing_doc = db_manager.get_document_by_hash(db_session, file_hash)
        if existing_doc and existing_doc.tenant_id == tenant_id:
            raise HTTPException(
                status_code=409,
                detail="File with identical content already exists"
            )
        
        # Check tenant quota
        quota_checker = TenantQuotaChecker(db_session)
        quota_result = quota_checker.check_quota(tenant_id, file_size)
        if not quota_result["has_quota"]:
            raise HTTPException(
                status_code=413,
                detail="Tenant storage quota exceeded"
            )
        
        # Generate document metadata
        document_id = generate_document_id()
        mime_type, file_type = detect_file_type(content, file.filename)
        storage_path = generate_storage_path(tenant_id, file.filename, file_hash)
        
        # Calculate retention date
        retention_days = retention_days or settings.default_retention_days
        retention_date = calculate_retention_date(retention_days)
        
        # Upload to MinIO
        upload_success = minio_manager.upload_file(
            tenant_id, storage_path, content, mime_type
        )
        
        if not upload_success:
            raise HTTPException(
                status_code=500,
                detail="Failed to upload file to storage"
            )
        
        # Create document record
        document_data = {
            "id": document_id,
            "tenant_id": tenant_id,
            "filename": file.filename,
            "file_size_bytes": file_size,
            "content_type": mime_type,
            "file_type": file_type,
            "file_hash": file_hash,
            "storage_path": storage_path,
            "tags": tags or [],
            "description": description,
            "retention_date": retention_date,
            "processing_status": "pending"
        }
        
        document = db_manager.create_document(db_session, document_data)
        
        # Start processing pipeline with Celery
        if pipeline_config:
            force_ocr = pipeline_config.ocr_force
        else:
            force_ocr = False
        
        # Queue OCR processing task
        ocr_task = process_document_ocr.delay(
            document_id=document_id,
            tenant_id=tenant_id,
            filename=file.filename,
            content_type=mime_type,
            storage_path=storage_path,
            force_ocr=force_ocr
        )
        
        logger.info(f"Queued OCR processing task {ocr_task.id} for document {document_id}")
        
        processing_time_ms = (time.time() - start_time) * 1000
        log_upload_stats(file.filename, file_size, tenant_id, processing_time_ms, True)
        
        logger.info(
            f"Successfully uploaded {file.filename}: "
            f"document_id={document_id}, processing started"
        )
        
        response_data = UploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_size_bytes=file_size,
            content_type=mime_type,
            file_hash=file_hash,
            upload_timestamp=document.upload_timestamp,
            processing_status=document.processing_status,
            storage_path=storage_path,
            message="File uploaded successfully, processing started"
        )
        
        return response_data.model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        log_upload_stats(file.filename, file_size, tenant_id, processing_time_ms, False)
        
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}"
        )


@app.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_files_batch(
    files: List[UploadFile] = File(..., description="Files to upload"),
    tenant_id: str = Depends(get_tenant_id),
    pipeline_config_json: Optional[str] = Form(None, description="Processing pipeline configuration as JSON string"),
    db_session = Depends(get_db)
):
    """
    Upload multiple files in a batch.
    
    **Parameters:**
    - **files**: List of files to upload (required, multipart/form-data)
    - **X-Tenant-ID**: Tenant identifier (required header)
    - **pipeline_config_json**: Processing pipeline configuration as JSON string (optional, form field)
    
    **Returns:**
    - Batch upload response with individual results
    
    **Note:** The pipeline_config_json should be sent as a form field containing JSON string.
    """
    if len(files) > settings.max_files_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum: {settings.max_files_per_request}"
        )
    
    # Parse pipeline config
    pipeline_config = parse_pipeline_config(pipeline_config_json)
    
    batch_id = generate_batch_id()
    successful_uploads = 0
    failed_uploads = 0
    documents = []
    errors = []
    
    for file in files:
        try:
            # Process each file individually
            content = await file.read()
            file_size = len(content)
            
            # Validate file
            validation_result = file_validator.validate(file.filename, content)
            if not validation_result["is_valid"]:
                failed_uploads += 1
                errors.append({
                    "filename": file.filename,
                    "error": f"Validation failed: {'; '.join(validation_result['errors'])}"
                })
                continue
            
            # Check for duplicates
            file_hash = calculate_file_hash(content)
            existing_doc = db_manager.get_document_by_hash(db_session, file_hash)
            if existing_doc and existing_doc.tenant_id == tenant_id:
                failed_uploads += 1
                errors.append({
                    "filename": file.filename,
                    "error": "File with identical content already exists"
                })
                continue
            
            # Generate metadata
            document_id = generate_document_id()
            mime_type, file_type = detect_file_type(content, file.filename)
            storage_path = generate_storage_path(tenant_id, file.filename, file_hash)
            
            # Upload to MinIO
            upload_success = minio_manager.upload_file(
                tenant_id, storage_path, content, mime_type
            )
            
            if not upload_success:
                failed_uploads += 1
                errors.append({
                    "filename": file.filename,
                    "error": "Failed to upload to storage"
                })
                continue
            
            # Create document record
            document_data = {
                "id": document_id,
                "tenant_id": tenant_id,
                "filename": file.filename,
                "file_size_bytes": file_size,
                "content_type": mime_type,
                "file_type": file_type,
                "file_hash": file_hash,
                "storage_path": storage_path,
                "processing_status": "pending"
            }
            
            document = db_manager.create_document(db_session, document_data)
            
            # Start processing with Celery
            if pipeline_config:
                force_ocr = pipeline_config.ocr_force
            else:
                force_ocr = False
            
            # Queue OCR processing task
            ocr_task = process_document_ocr.delay(
                document_id=document_id,
                tenant_id=tenant_id,
                filename=file.filename,
                content_type=mime_type,
                storage_path=storage_path,
                force_ocr=force_ocr
            )
            
            successful_uploads += 1
            documents.append(UploadResponse(
                document_id=document_id,
                filename=file.filename,
                file_size_bytes=file_size,
                content_type=mime_type,
                file_hash=file_hash,
                upload_timestamp=document.upload_timestamp,
                processing_status=document.processing_status,
                storage_path=storage_path,
                message="File uploaded successfully"
            ))
            
        except Exception as e:
            failed_uploads += 1
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
            logger.error(f"Batch upload error for {file.filename}: {e}")
    
    logger.info(
        f"Batch upload completed: {successful_uploads} successful, "
        f"{failed_uploads} failed"
    )
    
    return BatchUploadResponse(
        batch_id=batch_id,
        total_files=len(files),
        successful_uploads=successful_uploads,
        failed_uploads=failed_uploads,
        documents=documents,
        errors=errors
    )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    tenant_id: str = Depends(get_tenant_id),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    processing_status: Optional[str] = Query(None, description="Filter by processing status"),
    db_session = Depends(get_db)
):
    """
    List documents for a tenant.
    
    **Parameters:**
    - **X-Tenant-ID**: Tenant identifier (required header)
    - **page**: Page number (default: 1)
    - **page_size**: Page size (default: 20, max: 100)
    - **file_type**: Filter by file type (optional)
    - **processing_status**: Filter by processing status (optional)
    
    **Returns:**
    - List of documents with pagination info
    """
    documents = db_manager.get_documents_by_tenant(
        db_session, tenant_id, page, page_size, file_type, processing_status
    )
    
    total_count = db_manager.count_documents_by_tenant(
        db_session, tenant_id, file_type, processing_status
    )
    
    return DocumentListResponse(
        documents=documents,
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total_count
    )


@app.get("/documents/{document_id}", response_model=DocumentMetadata)
async def get_document(
    document_id: str,
    tenant_id: str = Depends(get_tenant_id),
    db_session = Depends(get_db)
):
    """
    Get document by ID.
    
    **Parameters:**
    - **document_id**: Document identifier
    - **X-Tenant-ID**: Tenant identifier (required header)
    
    **Returns:**
    - Document metadata
    """
    document = db_manager.get_document(db_session, document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return document


@app.get("/documents/{document_id}/download")
async def download_document(
    document_id: str,
    tenant_id: str = Depends(get_tenant_id),
    db_session = Depends(get_db)
):
    """
    Download original document file.
    
    **Parameters:**
    - **document_id**: Document identifier
    - **X-Tenant-ID**: Tenant identifier (required header)
    
    **Returns:**
    - File content as stream
    """
    document = db_manager.get_document(db_session, document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Download from MinIO
    file_content = minio_manager.download_file(tenant_id, document.storage_path)
    
    if not file_content:
        raise HTTPException(status_code=404, detail="File not found in storage")
    
    return StreamingResponse(
        io.BytesIO(file_content),
        media_type=document.content_type,
        headers={"Content-Disposition": f"attachment; filename={document.filename}"}
    )


@app.get("/documents/{document_id}/processing", response_model=ProcessingJobResponse)
async def get_processing_status(
    document_id: str,
    tenant_id: str = Depends(get_tenant_id),
    db_session = Depends(get_db)
):
    """
    Get processing status for a document.
    
    **Parameters:**
    - **document_id**: Document identifier
    - **X-Tenant-ID**: Tenant identifier (required header)
    
    **Returns:**
    - Processing status and job information
    """
    document = db_manager.get_document(db_session, document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    status_info = get_processing_status(document_id, tenant_id)
    
    return ProcessingJobResponse(
        job_id=document_id,  # Using document_id as job_id for simplicity
        document_id=document_id,
        status=document.processing_status,
        progress_percentage=100 if document.processing_status == "completed" else 0,
        message=f"Processing status: {document.processing_status}",
        started_at=document.created_at,
        completed_at=document.updated_at if document.processing_status == "completed" else None
    )


@app.put("/documents/{document_id}", response_model=DocumentMetadata)
async def update_document(
    document_id: str,
    update_request: DocumentUpdateRequest,
    tenant_id: str = Depends(get_tenant_id),
    db_session = Depends(get_db)
):
    """
    Update document metadata.
    
    **Parameters:**
    - **document_id**: Document identifier
    - **update_request**: Update data
    - **X-Tenant-ID**: Tenant identifier (required header)
    
    **Returns:**
    - Updated document metadata
    """
    document = db_manager.get_document(db_session, document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    update_data = {}
    
    if update_request.tags is not None:
        update_data["tags"] = update_request.tags
    
    if update_request.description is not None:
        update_data["description"] = update_request.description
    
    if update_request.retention_days is not None:
        update_data["retention_date"] = calculate_retention_date(update_request.retention_days)
    
    updated_document = db_manager.update_document(db_session, document_id, update_data)
    
    return updated_document


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    tenant_id: str = Depends(get_tenant_id),
    db_session = Depends(get_db)
):
    """
    Delete a document (soft delete).
    
    **Parameters:**
    - **document_id**: Document identifier
    - **X-Tenant-ID**: Tenant identifier (required header)
    
    **Returns:**
    - Success message
    """
    document = db_manager.get_document(db_session, document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    success = db_manager.soft_delete_document(db_session, document_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete document")
    
    return {"message": "Document deleted successfully"}


@app.get("/tenants/{tenant_id}/quota", response_model=TenantQuota)
async def get_tenant_quota(
    tenant_id: str,
    db_session = Depends(get_db)
):
    """
    Get tenant quota information.
    
    **Parameters:**
    - **tenant_id**: Tenant identifier
    
    **Returns:**
    - Tenant quota information
    """
    quota = db_manager.get_tenant_quota(db_session, tenant_id)
    usage = db_manager.get_tenant_storage_usage(db_session, tenant_id)
    
    if not quota:
        # Create default quota
        quota = db_manager.create_tenant_quota(db_session, tenant_id, {
            "max_storage_bytes": 1073741824,  # 1GB
            "max_files": 1000,
            "max_file_size_bytes": 104857600,  # 100MB
            "retention_days": 365
        })
    
    return TenantQuota(
        tenant_id=tenant_id,
        max_storage_bytes=quota.max_storage_bytes,
        used_storage_bytes=usage["used_storage_bytes"],
        max_files=quota.max_files,
        used_files=usage["used_files"],
        max_file_size_bytes=quota.max_file_size_bytes,
        retention_days=quota.retention_days
    )


# Helper functions
def get_processing_status(document_id: str, tenant_id: str) -> Dict[str, Any]:
    """Get processing status for a document."""
    with db_manager.get_session() as session:
        document = db_manager.get_document(session, document_id)
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


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).model_dump(mode='json')
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).model_dump(mode='json')
    )


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )
