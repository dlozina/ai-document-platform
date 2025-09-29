"""
Pydantic models for NER Service API
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class EntityInfo(BaseModel):
    """Information about a detected entity."""
    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity type/label")
    start: int = Field(..., description="Start position in text")
    end: int = Field(..., description="End position in text")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")


class NERResponse(BaseModel):
    """Response model for NER extraction."""
    document_id: Optional[str] = Field(None, description="Document identifier")
    text: str = Field(..., description="Input text")
    entities: List[EntityInfo] = Field(..., description="Detected entities")
    entity_count: int = Field(..., description="Total number of entities found")
    model_used: str = Field(..., description="Spacy model used for processing")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    text_length: int = Field(..., description="Length of input text")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    spacy_models_available: Dict[str, bool] = Field(..., description="Available Spacy models")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


class NERRequest(BaseModel):
    """Request model for NER processing."""
    text: str = Field(..., description="Text to process", min_length=1, max_length=1000000)
    entity_types: Optional[List[str]] = Field(None, description="Specific entity types to extract")
    include_confidence: bool = Field(True, description="Include confidence scores")


class BatchNERRequest(BaseModel):
    """Request model for batch NER processing."""
    texts: List[str] = Field(..., description="List of texts to process", min_items=1, max_items=100)
    entity_types: Optional[List[str]] = Field(None, description="Specific entity types to extract")
    include_confidence: bool = Field(True, description="Include confidence scores")


class BatchNERResponse(BaseModel):
    """Response model for batch NER processing."""
    results: List[NERResponse] = Field(..., description="NER results for each text")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    batch_size: int = Field(..., description="Number of texts processed")


class EntityStats(BaseModel):
    """Statistics about detected entities."""
    entity_type: str = Field(..., description="Entity type")
    count: int = Field(..., description="Number of occurrences")
    unique_count: int = Field(..., description="Number of unique entities")
    examples: List[str] = Field(..., description="Example entities of this type")


class NERStatsResponse(BaseModel):
    """Response model for NER statistics."""
    total_entities: int = Field(..., description="Total number of entities")
    entity_types: List[EntityStats] = Field(..., description="Statistics by entity type")
    most_common_entities: List[Dict[str, Any]] = Field(..., description="Most frequently detected entities")


class AsyncJobResponse(BaseModel):
    """Response model for async processing."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")


class ProcessingStats(BaseModel):
    """Processing statistics for monitoring."""
    text_length: int = Field(..., description="Length of processed text")
    processing_time_ms: str = Field(..., description="Processing time")
    model_used: str = Field(..., description="Model used for processing")
    entity_count: int = Field(..., description="Number of entities found")
    entity_types: List[str] = Field(..., description="Types of entities found")
