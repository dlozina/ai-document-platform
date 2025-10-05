"""
Pydantic models for Embedding Service API
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChunkResult(BaseModel):
    """Chunk result model."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk text content")
    embedding: List[float] = Field(..., description="Chunk embedding vector")
    chunk_index: int = Field(..., description="Chunk index in document")
    start: int = Field(..., description="Start character position")
    end: int = Field(..., description="End character position")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    document_id: Optional[str] = Field(None, description="Document identifier")
    text: str = Field(..., description="Input text content")
    embedding: List[float] = Field(..., description="Generated embedding vector (first chunk or single)")
    embedding_dimension: int = Field(..., description="Dimension of the embedding vector")
    model_name: str = Field(..., description="Name of the embedding model used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    text_length: int = Field(..., description="Length of input text")
    filename: Optional[str] = Field(None, description="Original filename if from file")
    
    # Chunking information
    total_chunks: Optional[int] = Field(None, description="Total number of chunks (if chunked)")
    chunk_results: Optional[List[ChunkResult]] = Field(None, description="All chunk results (if chunked)")
    method: Optional[str] = Field(None, description="Processing method used")


# Search models removed - vector search moved to ingestion service


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


# SearchRequest removed - vector search moved to ingestion service


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch embedding generation."""
    texts: List[str] = Field(..., description="List of texts to embed", min_items=1, max_items=100)
    document_ids: Optional[List[str]] = Field(None, description="List of document identifiers")
    metadata: Optional[List[Dict[str, Any]]] = Field(None, description="List of metadata objects")


# UpdateMetadataRequest removed - Qdrant operations moved to ingestion service


class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embedding generation."""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    document_ids: Optional[List[str]] = Field(None, description="List of document identifiers")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    batch_size: int = Field(..., description="Number of texts processed")


# CollectionInfo removed - Qdrant operations moved to ingestion service


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
