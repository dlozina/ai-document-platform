"""
Pydantic models for Query Service API
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import UUID4, BaseModel, Field, field_serializer


class QueryMode(str, Enum):
    """Query mode enumeration."""

    SEMANTIC_SEARCH = "semantic_search"
    EXTRACTIVE_QA = "extractive_qa"
    RAG = "rag"


class LLMProvider(str, Enum):
    """LLM provider enumeration."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    LOCAL = "local"


class DateRange(BaseModel):
    """Date range filter."""

    start: datetime | None = Field(None, description="Start date")
    end: datetime | None = Field(None, description="End date")


class QueryFilter(BaseModel):
    """Query filter parameters."""

    tenant_id: str | None = Field(None, description="Tenant identifier")
    content_type: str | None = Field(None, description="Content type filter")
    file_type: str | None = Field(None, description="File type filter")
    entity_labels: list[str] | None = Field(
        None, description="NER entity labels to filter by"
    )
    entity_text: list[str] | None = Field(
        None, description="Specific entity text to search for"
    )
    tags: list[str] | None = Field(None, description="Document tags")
    date_range: DateRange | None = Field(None, description="Date range filter")
    processing_status: str | None = Field(
        "completed", description="Processing status filter"
    )
    file_size_min: int | None = Field(None, description="Minimum file size in bytes")
    file_size_max: int | None = Field(None, description="Maximum file size in bytes")
    created_by: str | None = Field(None, description="Created by user filter")


class QueryRequest(BaseModel):
    """Query request model."""

    query: str = Field(
        ..., description="Search query or question", min_length=1, max_length=10000
    )
    mode: QueryMode = Field(QueryMode.SEMANTIC_SEARCH, description="Query mode")
    top_k: int = Field(10, description="Number of documents to retrieve", ge=1, le=100)
    score_threshold: float | None = Field(
        None, description="Minimum similarity score", ge=0.0, le=1.0
    )
    filter: QueryFilter | None = Field(None, description="Filter parameters")
    enable_reranking: bool | None = Field(None, description="Enable result reranking")
    max_context_length: int | None = Field(
        None, description="Maximum context length for LLM"
    )
    llm_provider: LLMProvider | None = Field(None, description="LLM provider override")
    llm_model: str | None = Field(None, description="LLM model override")


class SourceDocument(BaseModel):
    """Source document in query response - only chunked documents are supported."""

    document_id: UUID4 = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    quoted_text: str = Field(..., description="Relevant text excerpt")
    relevance_score: float = Field(..., description="Relevance score")
    page_or_position: str | None = Field(
        None, description="Page or position information"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )

    # Chunking information (required for all documents)
    chunk_id: str = Field(..., description="Chunk identifier")
    chunk_index: int = Field(..., description="Chunk index in document")
    total_chunks: int = Field(..., description="Total number of chunks in document")
    is_chunked: bool = Field(
        True, description="Whether this is a chunked document (always true)"
    )


class DetectedEntity(BaseModel):
    """Detected entity in query response."""

    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity label (PERSON, ORG, GPE, etc.)")
    confidence: float = Field(..., description="Entity confidence score")


class QueryResponse(BaseModel):
    """Query response model."""

    answer: str = Field(..., description="Generated answer or search results")
    confidence_score: float = Field(..., description="Overall confidence score")
    sources: list[SourceDocument] = Field(..., description="Source documents")
    detected_entities: list[DetectedEntity] = Field(
        default_factory=list, description="Detected entities"
    )
    retrieved_documents_count: int = Field(
        ..., description="Number of documents retrieved"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    query_mode: QueryMode = Field(..., description="Query mode used")
    llm_provider: str | None = Field(None, description="LLM provider used")
    llm_model: str | None = Field(None, description="LLM model used")


class SemanticSearchRequest(BaseModel):
    """Semantic search request model."""

    query: str = Field(..., description="Search query", min_length=1, max_length=10000)
    top_k: int = Field(10, description="Number of results", ge=1, le=100)
    score_threshold: float | None = Field(
        None, description="Minimum similarity score", ge=0.0, le=1.0
    )
    filter: QueryFilter | None = Field(None, description="Filter parameters")


class SemanticSearchResponse(BaseModel):
    """Semantic search response model."""

    query: str = Field(..., description="Search query")
    results: list[SourceDocument] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_time_ms: float = Field(..., description="Search time in milliseconds")
    detected_entities: list[DetectedEntity] = Field(
        default_factory=list, description="Detected entities"
    )


class QARequest(BaseModel):
    """Question-answering request model."""

    question: str = Field(
        ..., description="Question to answer", min_length=1, max_length=10000
    )
    mode: QueryMode = Field(QueryMode.RAG, description="QA mode")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    filter: QueryFilter | None = Field(None, description="Filter parameters")
    max_context_length: int | None = Field(None, description="Maximum context length")
    llm_provider: LLMProvider | None = Field(None, description="LLM provider override")
    llm_model: str | None = Field(None, description="LLM model override")


class QAResponse(BaseModel):
    """Question-answering response model."""

    answer: str = Field(..., description="Generated answer")
    confidence_score: float = Field(..., description="Answer confidence score")
    sources: list[SourceDocument] = Field(..., description="Source documents")
    detected_entities: list[DetectedEntity] = Field(
        default_factory=list, description="Detected entities"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    mode: QueryMode = Field(..., description="QA mode used")
    llm_provider: str | None = Field(None, description="LLM provider used")
    llm_model: str | None = Field(None, description="LLM model used")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    qdrant_available: bool = Field(..., description="Whether Qdrant is available")
    database_connected: bool = Field(..., description="Database connection status")
    embedding_model_loaded: bool = Field(
        ..., description="Whether embedding model is loaded"
    )
    llm_available: bool = Field(..., description="Whether LLM is available")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Additional error details")
    error_code: str | None = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        """Serialize datetime to ISO string."""
        return value.isoformat()


class CollectionInfo(BaseModel):
    """Information about a Qdrant collection."""

    name: str = Field(..., description="Collection name")
    vector_size: int = Field(..., description="Vector dimension")
    points_count: int = Field(..., description="Number of points in collection")
    status: str = Field(..., description="Collection status")


class DocumentStats(BaseModel):
    """Document statistics."""

    total_documents: int = Field(..., description="Total number of documents")
    processed_documents: int = Field(..., description="Number of processed documents")
    documents_with_embeddings: int = Field(
        ..., description="Number of documents with embeddings"
    )
    documents_with_ner: int = Field(..., description="Number of documents with NER")
    average_embedding_score: float | None = Field(
        None, description="Average embedding quality score"
    )


class QueryStats(BaseModel):
    """Query statistics."""

    total_queries: int = Field(..., description="Total number of queries")
    semantic_search_queries: int = Field(
        ..., description="Number of semantic search queries"
    )
    qa_queries: int = Field(..., description="Number of QA queries")
    rag_queries: int = Field(..., description="Number of RAG queries")
    average_response_time_ms: float = Field(..., description="Average response time")
    average_confidence_score: float = Field(..., description="Average confidence score")
