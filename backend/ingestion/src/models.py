"""
Pydantic models for Ingestion Service API
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, UUID4, ConfigDict, field_serializer


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FileType(str, Enum):
    """Supported file types."""
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    DOCUMENT = "document"
    OTHER = "other"


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    id: UUID4 = Field(..., description="Unique document identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    filename: str = Field(..., description="Original filename")
    file_size_bytes: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME content type")
    file_type: FileType = Field(..., description="File type category")
    file_hash: str = Field(..., description="SHA-256 hash of file content")
    storage_path: str = Field(..., description="Path in object storage")
    upload_timestamp: datetime = Field(..., description="Upload timestamp")
    created_by: Optional[str] = Field(None, description="User who uploaded the file")
    
    # Processing status
    processing_status: ProcessingStatus = Field(
        ProcessingStatus.PENDING, 
        description="Current processing status"
    )
    ocr_status: Optional[ProcessingStatus] = Field(None, description="OCR processing status")
    ner_status: Optional[ProcessingStatus] = Field(None, description="NER processing status")
    embedding_status: Optional[ProcessingStatus] = Field(None, description="Embedding processing status")
    
    # Processing results
    ocr_text: Optional[str] = Field(None, description="Extracted text from OCR")
    ner_entities: Optional[List[Dict[str, Any]]] = Field(None, description="Named entities")
    embedding_vector: Optional[List[float]] = Field(None, description="Document embedding vector")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Document tags")
    description: Optional[str] = Field(None, description="Document description")
    retention_date: Optional[datetime] = Field(None, description="Data retention expiration")
    is_deleted: bool = Field(False, description="Soft delete flag")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class UploadResponse(BaseModel):
    """Response model for file upload."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    document_id: UUID4 = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_size_bytes: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME content type")
    file_hash: str = Field(..., description="SHA-256 hash")
    upload_timestamp: datetime = Field(..., description="Upload timestamp")
    processing_status: ProcessingStatus = Field(..., description="Processing status")
    storage_path: str = Field(..., description="Storage path")
    message: str = Field(..., description="Status message")
    
    @field_serializer('upload_timestamp')
    def serialize_upload_timestamp(self, value: datetime) -> str:
        return value.isoformat() if value else None


class BatchUploadResponse(BaseModel):
    """Response model for batch file upload."""
    batch_id: UUID4 = Field(..., description="Batch identifier")
    total_files: int = Field(..., description="Total files in batch")
    successful_uploads: int = Field(..., description="Successfully uploaded files")
    failed_uploads: int = Field(..., description="Failed uploads")
    documents: List[UploadResponse] = Field(..., description="Upload results")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Upload errors")


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    documents: List[DocumentMetadata] = Field(..., description="List of documents")
    total_count: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")


class ProcessingJobResponse(BaseModel):
    """Response model for processing job status."""
    job_id: UUID4 = Field(..., description="Job identifier")
    document_id: UUID4 = Field(..., description="Document identifier")
    status: ProcessingStatus = Field(..., description="Job status")
    progress_percentage: int = Field(..., description="Progress percentage (0-100)")
    message: str = Field(..., description="Status message")
    started_at: Optional[datetime] = Field(None, description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    database_connected: bool = Field(..., description="Database connection status")
    minio_connected: bool = Field(..., description="MinIO connection status")
    dependencies: Dict[str, bool] = Field(..., description="Dependency status")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, value: datetime) -> str:
        """Serialize datetime to ISO string."""
        return value.isoformat()


class FileValidationResult(BaseModel):
    """File validation result."""
    is_valid: bool = Field(..., description="Whether file is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    file_type: Optional[str] = Field(None, description="Detected file type")
    file_size_bytes: int = Field(..., description="File size")


class TenantQuota(BaseModel):
    """Tenant quota information."""
    tenant_id: str = Field(..., description="Tenant identifier")
    max_storage_bytes: int = Field(..., description="Maximum storage in bytes")
    used_storage_bytes: int = Field(..., description="Used storage in bytes")
    max_files: int = Field(..., description="Maximum number of files")
    used_files: int = Field(..., description="Number of files used")
    max_file_size_bytes: int = Field(..., description="Maximum file size")
    retention_days: int = Field(..., description="Data retention period")


class DocumentSearchRequest(BaseModel):
    """Document search request."""
    query: Optional[str] = Field(None, description="Search query")
    file_types: Optional[List[FileType]] = Field(None, description="Filter by file types")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    date_from: Optional[datetime] = Field(None, description="Filter from date")
    date_to: Optional[datetime] = Field(None, description="Filter to date")
    processing_status: Optional[ProcessingStatus] = Field(None, description="Filter by processing status")
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Page size")


class DocumentUpdateRequest(BaseModel):
    """Document update request."""
    tags: Optional[List[str]] = Field(None, description="Document tags")
    description: Optional[str] = Field(None, description="Document description")
    retention_days: Optional[int] = Field(None, ge=1, description="Retention period in days")


class ProcessingPipelineConfig(BaseModel):
    """Processing pipeline configuration."""
    enable_ocr: bool = Field(True, description="Enable OCR processing")
    enable_ner: bool = Field(True, description="Enable NER processing")
    enable_embedding: bool = Field(True, description="Enable embedding processing")
    ocr_force: bool = Field(False, description="Force OCR even for searchable PDFs")
    priority: int = Field(1, ge=1, le=10, description="Processing priority")


class UploadProgress(BaseModel):
    """Upload progress information."""
    document_id: UUID4 = Field(..., description="Document identifier")
    bytes_uploaded: int = Field(..., description="Bytes uploaded")
    total_bytes: int = Field(..., description="Total bytes")
    percentage: float = Field(..., description="Upload percentage")
    status: str = Field(..., description="Upload status")
