"""
Pydantic models for OCR Service API
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class OCRResponse(BaseModel):
    """Response model for OCR extraction."""
    document_id: Optional[str] = Field(None, description="Document identifier")
    text: str = Field(..., description="Extracted text content")
    page_count: int = Field(..., description="Number of pages processed")
    method: str = Field(..., description="Extraction method used")
    confidence: Optional[float] = Field(None, description="OCR confidence score (0-100)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    file_size_bytes: int = Field(..., description="Original file size")
    filename: str = Field(..., description="Original filename")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    tesseract_available: bool = Field(..., description="Whether Tesseract is available")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


class WordInfo(BaseModel):
    """Word information with bounding box."""
    text: str = Field(..., description="Word text")
    confidence: float = Field(..., description="OCR confidence (0-100)")
    left: int = Field(..., description="Left coordinate")
    top: int = Field(..., description="Top coordinate")
    width: int = Field(..., description="Width")
    height: int = Field(..., description="Height")
    page_num: int = Field(..., description="Page number")


class LayoutResponse(BaseModel):
    """Response model for layout extraction."""
    filename: str = Field(..., description="Original filename")
    word_count: int = Field(..., description="Number of words detected")
    words: List[WordInfo] = Field(..., description="List of words with positions")


class AsyncJobResponse(BaseModel):
    """Response model for async processing."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")


class ProcessingStats(BaseModel):
    """Processing statistics for monitoring."""
    filename: str = Field(..., description="Processed filename")
    file_size: str = Field(..., description="Formatted file size")
    processing_time_ms: str = Field(..., description="Processing time")
    method: str = Field(..., description="Processing method")
    text_length: int = Field(..., description="Length of extracted text")
    confidence: Optional[float] = Field(None, description="OCR confidence")
