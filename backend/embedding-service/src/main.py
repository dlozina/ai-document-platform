"""
Embedding Service API

FastAPI application that exposes embedding generation and vector search functionality via REST endpoints.
"""

import logging
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .enhanced_embedding_processor import EnhancedEmbeddingProcessor
from .config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global embedding processor instance
embedding_processor: Optional[EnhancedEmbeddingProcessor] = None
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global embedding_processor
    
    logger.info("Starting Embedding Service with chunking...")
    embedding_processor = EnhancedEmbeddingProcessor(
        model_name=settings.embedding_model,
        qdrant_host=settings.qdrant_host,
        qdrant_port=settings.qdrant_port,
        qdrant_api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection_name,
        vector_size=settings.qdrant_vector_size,
        chunk_size=1000,  # 1000 characters per chunk
        chunk_overlap=200  # 200 character overlap
    )
    logger.info("Embedding Service ready")
    
    yield
    
    logger.info("Shutting down Embedding Service...")


# Create FastAPI app
app = FastAPI(
    title="Embedding Service",
    description="""
    ## Production-ready Embedding Service
    
    Generate text embeddings and perform vector similarity search using advanced NLP models.
    
    ### Features:
    - **Text embedding generation** using sentence-transformers models
    - **Vector storage and retrieval** with Qdrant vector database
    - **Similarity search** with configurable thresholds and filters
    - **Batch processing** for efficient embedding generation
    - **Multiple text formats** support (TXT, JSON, CSV, Markdown)
    - **Async processing** for large files
    
    ### Supported Formats:
    - **Text files**: `.txt`, `.md`
    - **Structured data**: `.json`, `.csv`
    
    ### Embedding Models:
    - Default: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
    - Configurable via environment variables
    
    ### Vector Operations:
    1. **Generate embeddings** from text
    2. **Store vectors** in Qdrant collection
    3. **Search similar** vectors with cosine similarity
    4. **Batch operations** for multiple texts
    
    ### Performance Tips:
    - Use batch endpoints for multiple texts (more efficient)
    - Configure appropriate batch sizes for your use case
    - Use filters to narrow search results
    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "Embedding Service Team",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": "http://localhost:8002",
            "description": "Development server"
        },
        {
            "url": "https://api.example.com",
            "description": "Production server"
        }
    ]
)


from .models import (
    EmbeddingResponse, SearchResponse, HealthResponse, ErrorResponse,
    EmbeddingRequest, SearchRequest, BatchEmbeddingRequest, BatchEmbeddingResponse,
    CollectionInfo, AsyncJobResponse, UpdateMetadataRequest
)


# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and dependencies availability.
    
    **Returns:**
    - Service status (healthy/degraded)
    - Service information
    - Qdrant availability
    - Embedding model status
    """
    if not embedding_processor:
        return HealthResponse(
            status="unhealthy",
            service="embedding-service",
            version="1.0.0",
            qdrant_available=False,
            embedding_model_loaded=False
        )
    
    try:
        health_status = embedding_processor.health_check()
        
        overall_status = "healthy" if (
            health_status["embedding_model_loaded"] and 
            health_status["qdrant_available"] and 
            health_status["collection_exists"]
        ) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            service="embedding-service",
            version="1.0.0",
            qdrant_available=health_status["qdrant_available"],
            embedding_model_loaded=health_status["embedding_model_loaded"]
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            service="embedding-service",
            version="1.0.0",
            qdrant_available=False,
            embedding_model_loaded=False
        )


@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embedding(
    request: EmbeddingRequest,
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    document_id: Optional[str] = Header(None, alias="X-Document-ID")
):
    """
    Generate embedding for a single text.
    
    **Parameters:**
    - **text**: The text to embed (required)
    - **document_id**: Optional document identifier
    - **metadata**: Optional metadata dictionary
    - **X-Tenant-ID**: Tenant identifier (header)
    - **X-Document-ID**: Document identifier (header)
    
    **Returns:**
    - Generated embedding vector and metadata
    """
    if not embedding_processor:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not initialized"
        )
    
    try:
        logger.info(f"Generating embedding for text (length: {len(request.text)}, tenant: {tenant_id})")
        
        # Process document with chunking
        doc_id = request.document_id or document_id
        metadata = request.metadata or {}
        if tenant_id:
            metadata["tenant_id"] = tenant_id
        
        if doc_id:
            # Use chunked processing for documents
            result = embedding_processor.process_document_with_chunking(
                text=request.text,
                document_id=doc_id,
                filename=metadata.get("filename", "text_input"),
                metadata=metadata
            )
            
            # Return info about the first chunk for backward compatibility
            first_chunk = result["chunk_results"][0] if result["chunk_results"] else None
            embedding = embedding_processor.generate_embedding(request.text[:512])  # Generate embedding for response
            point_id = first_chunk["point_id"] if first_chunk else None
        else:
            # For text-only requests without document_id, use simple embedding
            embedding = embedding_processor.generate_embedding(request.text)
            point_id = None
        
        return EmbeddingResponse(
            document_id=request.document_id or document_id,
            text=request.text,
            embedding=embedding,
            embedding_dimension=len(embedding),
            model_name=settings.embedding_model,
            processing_time_ms=0.0,  # Will be calculated in processor
            text_length=len(request.text),
            filename=None
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Embedding generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embedding: {str(e)}"
        )


@app.post("/embed-batch", response_model=BatchEmbeddingResponse)
async def generate_batch_embeddings(
    request: BatchEmbeddingRequest,
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
):
    """
    Generate embeddings for multiple texts efficiently.
    
    **Parameters:**
    - **texts**: List of texts to embed (required)
    - **document_ids**: Optional list of document identifiers
    - **metadata**: Optional list of metadata dictionaries
    - **X-Tenant-ID**: Tenant identifier (header)
    
    **Returns:**
    - List of embedding vectors and metadata
    """
    if not embedding_processor:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")
    
    if len(request.texts) > 100:
        raise HTTPException(
            status_code=400,
            detail="Too many texts. Maximum batch size: 100"
        )
    
    try:
        logger.info(f"Generating batch embeddings for {len(request.texts)} texts")
        
        start_time = time.time()
        
        # Generate embeddings
        embeddings = embedding_processor.generate_batch_embeddings(request.texts)
        
        # Store in Qdrant if document_ids provided
        if request.document_ids:
            for i, (text, embedding, doc_id) in enumerate(zip(request.texts, embeddings, request.document_ids)):
                metadata = request.metadata[i] if request.metadata and i < len(request.metadata) else {}
                if tenant_id:
                    metadata["tenant_id"] = tenant_id
                
                embedding_processor.store_embedding(
                    text=text,
                    embedding=embedding,
                    document_id=doc_id,
                    metadata=metadata
                )
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchEmbeddingResponse(
            embeddings=embeddings,
            document_ids=request.document_ids,
            processing_time_ms=processing_time,
            batch_size=len(request.texts)
        )
        
    except Exception as e:
        logger.error(f"Batch embedding generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate batch embeddings: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse)
async def search_similar(
    request: SearchRequest,
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
):
    """
    Search for similar embeddings using vector similarity.
    
    **Parameters:**
    - **query**: Search query text (required)
    - **limit**: Maximum number of results (default: 10)
    - **score_threshold**: Minimum similarity score (0-1)
    - **filter**: Optional metadata filter
    - **X-Tenant-ID**: Tenant identifier (header)
    
    **Returns:**
    - List of similar documents with scores
    """
    if not embedding_processor:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")
    
    try:
        logger.info(f"Searching for similar documents to query (length: {len(request.query)})")
        
        start_time = time.time()
        
        # Prepare filter conditions
        filter_conditions = request.filter or {}
        if tenant_id:
            filter_conditions["tenant_id"] = tenant_id
        
        # Perform chunked search
        results = embedding_processor.search_similar_chunks(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold,
            filter_conditions=filter_conditions if filter_conditions else None
        )
        
        search_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@app.post("/embed-file", response_model=EmbeddingResponse)
async def embed_file(
    file: UploadFile = File(..., description="Text file to process"),
    document_id: Optional[str] = Header(None, alias="X-Document-ID"),
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
):
    """
    Generate embedding from uploaded text file.
    
    **Supported formats:**
    - Text files (.txt, .md)
    - Structured data (.json, .csv)
    
    **Parameters:**
    - **file**: The text file to process (required)
    - **X-Document-ID**: Document identifier (header)
    - **X-Tenant-ID**: Tenant identifier (header)
    
    **Returns:**
    - Generated embedding and extracted text
    """
    if not embedding_processor:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")
    
    # Validate file size
    max_size = settings.max_file_size_bytes
    content = await file.read()
    file_size = len(content)
    
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {max_size / 1024 / 1024:.0f}MB"
        )
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    try:
        logger.info(f"Processing file: {file.filename} (size: {file_size / 1024:.2f}KB)")
        
        # Process the file
        result = embedding_processor.process_text_file(
            file_content=content,
            filename=file.filename,
            document_id=document_id,
            metadata={"tenant_id": tenant_id} if tenant_id else None
        )
        
        return EmbeddingResponse(**result)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"File processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {str(e)}"
        )


@app.get("/collection/info", response_model=CollectionInfo)
async def get_collection_info():
    """
    Get information about the Qdrant collection.
    
    **Returns:**
    - Collection metadata and statistics
    """
    if not embedding_processor:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")
    
    try:
        info = embedding_processor.get_collection_info()
        return CollectionInfo(**info)
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/embedding/{point_id}")
async def delete_embedding(point_id: str):
    """
    Delete an embedding from the collection.
    
    **Parameters:**
    - **point_id**: ID of the point to delete
    
    **Returns:**
    - Success confirmation
    """
    if not embedding_processor:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")
    
    try:
        success = embedding_processor.delete_embedding(point_id)
        if success:
            return {"message": f"Embedding {point_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Embedding {point_id} not found")
    except Exception as e:
        logger.error(f"Failed to delete embedding {point_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/update-metadata", response_model=dict)
async def update_metadata(
    request: UpdateMetadataRequest,
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
):
    """
    Update metadata for an existing embedding in Qdrant.
    
    **Parameters:**
    - **document_id**: Document identifier (required)
    - **metadata**: Updated metadata dictionary (required)
    - **X-Tenant-ID**: Tenant identifier (header)
    
    **Returns:**
    - Success confirmation
    """
    if not embedding_processor:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")
    
    try:
        logger.info(f"Updating metadata for document: {request.document_id}")
        
        # Add tenant_id to metadata if provided
        metadata = request.metadata.copy()
        if tenant_id:
            metadata["tenant_id"] = tenant_id
        
        success = embedding_processor.update_metadata(request.document_id, metadata)
        
        if success:
            return {"message": f"Metadata updated successfully for document {request.document_id}"}
        else:
            raise HTTPException(status_code=404, detail=f"Document {request.document_id} not found in Qdrant")
            
    except Exception as e:
        logger.error(f"Failed to update metadata for document {request.document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).model_dump()
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
        ).model_dump()
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
