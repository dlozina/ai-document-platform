"""
Pydantic models for Embedding Service API
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    document_id: Optional[str] = Field(None, description="Document identifier")
    text: str = Field(..., description="Input text content")
    embedding: List[float] = Field(..., description="Generated embedding vector")
    embedding_dimension: int = Field(..., description="Dimension of the embedding vector")
    model_name: str = Field(..., description="Name of the embedding model used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    text_length: int = Field(..., description="Length of input text")
    filename: Optional[str] = Field(None, description="Original filename if from file")


class SearchResult(BaseModel):
    """Individual search result."""
    id: str = Field(..., description="Qdrant point ID (UUID)")
    score: float = Field(..., description="Similarity score")
    text: str = Field(..., description="Text content")
    document_id: Optional[str] = Field(None, description="Original document ID")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")


class SearchResponse(BaseModel):
    """Response model for vector search."""
    query: str = Field(..., description="Search query")
    results: List[SearchResult] = Field(..., description="Search results with scores")
    total_results: int = Field(..., description="Total number of results found")
    search_time_ms: float = Field(..., description="Search time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    qdrant_available: bool = Field(..., description="Whether Qdrant is available")
    embedding_model_loaded: bool = Field(..., description="Whether embedding model is loaded")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    text: str = Field(..., description="Text to embed", min_length=1, max_length=10000)
    document_id: Optional[str] = Field(None, description="Document identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SearchRequest(BaseModel):
    """Request model for vector search."""
    query: str = Field(..., description="Search query", min_length=1, max_length=10000)
    limit: int = Field(10, description="Maximum number of results", ge=1, le=100)
    score_threshold: Optional[float] = Field(None, description="Minimum similarity score", ge=0.0, le=1.0)
    filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filter")


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch embedding generation."""
    texts: List[str] = Field(..., description="List of texts to embed", min_items=1, max_items=100)
    document_ids: Optional[List[str]] = Field(None, description="List of document identifiers")
    metadata: Optional[List[Dict[str, Any]]] = Field(None, description="List of metadata objects")


class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embedding generation."""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    document_ids: Optional[List[str]] = Field(None, description="List of document identifiers")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    batch_size: int = Field(..., description="Number of texts processed")


class CollectionInfo(BaseModel):
    """Information about a Qdrant collection."""
    name: str = Field(..., description="Collection name")
    vector_size: int = Field(..., description="Vector dimension")
    points_count: int = Field(..., description="Number of points in collection")
    status: str = Field(..., description="Collection status")


class AsyncJobResponse(BaseModel):
    """Response model for async processing."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")


class ProcessingStats(BaseModel):
    """Processing statistics for monitoring."""
    filename: Optional[str] = Field(None, description="Processed filename")
    text_length: int = Field(..., description="Length of processed text")
    processing_time_ms: str = Field(..., description="Processing time")
    embedding_dimension: int = Field(..., description="Embedding dimension")
    model_name: str = Field(..., description="Model used")
