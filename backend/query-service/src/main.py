"""
Query Service API

FastAPI application that exposes semantic search and question-answering functionality via REST endpoints.
"""

import logging
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .query_processor import QueryProcessor
from .config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global query processor instance
query_processor: Optional[QueryProcessor] = None
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global query_processor
    
    logger.info("Starting Query Service...")
    query_processor = QueryProcessor()
    logger.info("Query Service ready")
    
    yield
    
    logger.info("Shutting down Query Service...")


# Create FastAPI app
app = FastAPI(
    title="Query Service",
    description="""
    ## Production-ready Query Service
    
    Semantic search and question-answering system that leverages OCR, NER, and embeddings to provide intelligent answers.
    
    ### Features:
    - **Semantic Search** using vector similarity and embeddings
    - **Question Answering** with RAG mode
    - **Advanced Filtering** by metadata, entities, dates, and more
    - **Entity Recognition** integration for enhanced search
    - **Result Reranking** for improved relevance
    - **Multiple LLM Support** (OpenAI, Anthropic, Mistral)
    
    ### Query Modes:
    1. **Semantic Search**: Find relevant documents using vector similarity
    2. **Extractive QA**: Extract exact answers from document text (testing only)
    3. **RAG**: Generate comprehensive answers using LLM + retrieved context
    
    ### Filtering Capabilities:
    - **Metadata filters**: tenant_id, content_type, file_type
    - **Entity filters**: Find documents mentioning specific people, organizations, locations
    - **Date ranges**: Filter by upload timestamp
    - **File properties**: Size ranges, processing status
    - **Custom tags**: Document tags and descriptions
    
    ### Use Cases:
    - "What programming languages does Dino know?" → Extract Python, JavaScript, Rust
    - "Where has Dino worked?" → Find Macrometa, identify as US startup
    - "Find documents about system design" → Semantic search on that topic
    - "Show me all documents mentioning Croatian locations" → Entity-filtered search
    
    ### Performance Features:
    - Vector similarity search with Qdrant
    - Result reranking with cross-encoder models
    - Caching support for frequent queries
    - Streaming responses for long LLM outputs
    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "Query Service Team",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": "http://localhost:8004",
            "description": "Development server"
        },
        {
            "url": "https://api.example.com",
            "description": "Production server"
        }
    ]
)


from .models import (
    QueryRequest, QueryResponse, SemanticSearchRequest, SemanticSearchResponse,
    QARequest, QAResponse, HealthResponse, ErrorResponse, CollectionInfo,
    DocumentStats, QueryStats, QueryMode
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
    - Database connection status
    - Embedding model status
    - LLM availability
    """
    if not query_processor:
        return HealthResponse(
            status="unhealthy",
            service="query-service",
            version="1.0.0",
            qdrant_available=False,
            database_connected=False,
            embedding_model_loaded=False,
            llm_available=False
        )
    
    try:
        health_status = query_processor.health_check()
        
        overall_status = "healthy" if (
            health_status["embedding_model_loaded"] and 
            health_status["qdrant_available"] and 
            health_status["database_connected"]
        ) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            service="query-service",
            version="1.0.0",
            qdrant_available=health_status["qdrant_available"],
            database_connected=health_status["database_connected"],
            embedding_model_loaded=health_status["embedding_model_loaded"],
            llm_available=health_status.get("llm_available", False)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            service="query-service",
            version="1.0.0",
            qdrant_available=False,
            database_connected=False,
            embedding_model_loaded=False,
            llm_available=False
        )


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
):
    """
    Main query endpoint supporting all query modes.
    
    **Parameters:**
    - **query**: Search query or question (required)
    - **mode**: Query mode - semantic_search, extractive_qa (testing), or rag
    - **top_k**: Number of documents to retrieve (default: 10)
    - **score_threshold**: Minimum similarity score (0-1)
    - **filter**: Filter parameters (metadata, entities, dates, etc.)
    - **enable_reranking**: Enable result reranking
    - **max_context_length**: Maximum context length for LLM
    - **X-Tenant-ID**: Tenant identifier (header)
    
    **Returns:**
    - Generated answer or search results
    - Source documents with relevance scores
    - Detected entities
    - Processing metadata
    """
    if not query_processor:
        raise HTTPException(
            status_code=503,
            detail="Query service not initialized"
        )
    
    try:
        logger.info(f"Processing query: {request.query[:100]}... (mode: {request.mode})")
        
        start_time = time.time()
        
        # Convert filter to dict if provided
        filter_params = request.filter.dict() if request.filter else None
        
        # Perform semantic search to get relevant documents
        search_results = query_processor.semantic_search(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            filter_params=filter_params,
            tenant_id=tenant_id
        )
        
        # Apply entity filtering if specified
        if request.filter and (request.filter.entity_labels or request.filter.entity_text):
            search_results = query_processor.filter_by_entities(
                search_results,
                entity_labels=request.filter.entity_labels,
                entity_text=request.filter.entity_text
            )
        
        # Rerank results if enabled
        if request.enable_reranking or settings.enable_reranking:
            search_results = query_processor.rerank_results(request.query, search_results)
        
        # Process based on query mode
        if request.mode == QueryMode.SEMANTIC_SEARCH:
            answer = f"Found {len(search_results)} relevant documents for your query."
            confidence_score = 0.8 if search_results else 0.1
            
        elif request.mode == QueryMode.EXTRACTIVE_QA:
            if search_results:
                # Extract answer from top result
                top_doc = search_results[0]
                answer, confidence_score = query_processor.extract_answer_span(
                    request.query, top_doc["text"], top_doc.get("metadata")
                )
            else:
                answer = "No relevant documents found to answer your question."
                confidence_score = 0.1
                
        elif request.mode == QueryMode.RAG:
            if search_results:
                # Generate RAG answer (async)
                answer, confidence_score = await query_processor.generate_rag_answer(
                    request.query,
                    search_results,
                    max_context_length=request.max_context_length
                )
            else:
                answer = "No relevant documents found to generate an answer."
                confidence_score = 0.1
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid query mode: {request.mode}")
        
        # Extract entities from query and results
        query_entities = query_processor.extract_entities(request.query)
        result_entities = query_processor.collect_entities_from_results(search_results)
        
        # Combine and deduplicate entities
        all_entities = query_entities.copy()
        seen_entities = {(e["text"].lower(), e["label"]) for e in query_entities}
        
        for entity in result_entities:
            entity_key = (entity["text"].lower(), entity["label"])
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                all_entities.append(entity)
        
        detected_entities = all_entities
        
        # Format source documents
        sources = []
        for i, result in enumerate(search_results[:5]):  # Limit to top 5 sources
            sources.append({
                "document_id": result.get("document_id", ""),
                "filename": result.get("filename", "Unknown"),
                "quoted_text": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"],
                "relevance_score": result["score"],
                "page_or_position": f"Result {i+1}",
                "metadata": result.get("metadata", {})
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=answer,
            confidence_score=confidence_score,
            sources=sources,
            detected_entities=detected_entities,
            retrieved_documents_count=len(search_results),
            processing_time_ms=processing_time,
            query_mode=request.mode,
            llm_provider=settings.llm_provider if request.mode == QueryMode.RAG else None,
            llm_model=getattr(settings, f"{settings.llm_provider}_model", None) if request.mode == QueryMode.RAG else None
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@app.post("/search", response_model=SemanticSearchResponse)
async def semantic_search(
    request: SemanticSearchRequest,
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
):
    """
    Semantic search endpoint for finding relevant documents.
    
    **Parameters:**
    - **query**: Search query (required)
    - **top_k**: Number of results (default: 10)
    - **score_threshold**: Minimum similarity score (0-1)
    - **filter**: Filter parameters
    - **X-Tenant-ID**: Tenant identifier (header)
    
    **Returns:**
    - List of relevant documents with scores
    """
    if not query_processor:
        raise HTTPException(status_code=503, detail="Query service not initialized")
    
    try:
        logger.info(f"Semantic search: {request.query[:100]}...")
        
        start_time = time.time()
        
        # Convert filter to dict if provided
        filter_params = request.filter.dict() if request.filter else None
        
        # Perform search
        results = query_processor.semantic_search(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            filter_params=filter_params,
            tenant_id=tenant_id
        )
        
        # Apply entity filtering if specified
        if request.filter and (request.filter.entity_labels or request.filter.entity_text):
            results = query_processor.filter_by_entities(
                results,
                entity_labels=request.filter.entity_labels,
                entity_text=request.filter.entity_text
            )
        
        # Rerank results if enabled
        if settings.enable_reranking:
            results = query_processor.rerank_results(request.query, results)
        
        search_time = (time.time() - start_time) * 1000
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "document_id": result.get("document_id", ""),
                "filename": result.get("filename", "Unknown"),
                "quoted_text": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"],
                "relevance_score": result["score"],
                "page_or_position": None,
                "metadata": result.get("metadata", {})
            })
        
        # Extract entities from query and results for semantic search
        query_entities = query_processor.extract_entities(request.query)
        result_entities = query_processor.collect_entities_from_results(results)
        
        # Combine and deduplicate entities
        all_entities = query_entities.copy()
        seen_entities = {(e["text"].lower(), e["label"]) for e in query_entities}
        
        for entity in result_entities:
            entity_key = (entity["text"].lower(), entity["label"])
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                all_entities.append(entity)
        
        return SemanticSearchResponse(
            query=request.query,
            results=formatted_results,
            total_results=len(formatted_results),
            search_time_ms=search_time,
            detected_entities=all_entities
        )
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Semantic search failed: {str(e)}"
        )


@app.post("/qa", response_model=QAResponse)
async def question_answering(
    request: QARequest,
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
):
    """
    Question-answering endpoint with RAG mode.
    
    **Parameters:**
    - **question**: Question to answer (required)
    - **mode**: QA mode - rag (default)
    - **top_k**: Number of documents to retrieve (default: 5)
    - **filter**: Filter parameters
    - **max_context_length**: Maximum context length for LLM
    - **X-Tenant-ID**: Tenant identifier (header)
    
    **Returns:**
    - Generated answer with confidence score
    - Source documents
    - Detected entities
    """
    if not query_processor:
        raise HTTPException(status_code=503, detail="Query service not initialized")
    
    try:
        logger.info(f"QA request: {request.question[:100]}... (mode: {request.mode})")
        
        start_time = time.time()
        
        # Convert filter to dict if provided
        filter_params = request.filter.dict() if request.filter else None
        
        # Perform semantic search to get relevant documents
        search_results = query_processor.semantic_search(
            query=request.question,
            top_k=request.top_k,
            score_threshold=-5.0,  # Lower threshold for QA (negative scores are common)
            filter_params=filter_params,
            tenant_id=tenant_id
        )
        
        # Apply entity filtering if specified
        if request.filter and (request.filter.entity_labels or request.filter.entity_text):
            search_results = query_processor.filter_by_entities(
                search_results,
                entity_labels=request.filter.entity_labels,
                entity_text=request.filter.entity_text
            )
        
        # Process based on QA mode
        if request.mode == QueryMode.RAG:
            if search_results:
                # Generate RAG answer (async)
                answer, confidence_score = await query_processor.generate_rag_answer(
                    request.question,
                    search_results,
                    max_context_length=request.max_context_length
                )
            else:
                answer = "No relevant documents found to generate an answer."
                confidence_score = 0.1
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid QA mode: {request.mode}")
        
        # Extract entities from question and results
        query_entities = query_processor.extract_entities(request.question)
        result_entities = query_processor.collect_entities_from_results(search_results)
        
        # Combine and deduplicate entities
        all_entities = query_entities.copy()
        seen_entities = {(e["text"].lower(), e["label"]) for e in query_entities}
        
        for entity in result_entities:
            entity_key = (entity["text"].lower(), entity["label"])
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                all_entities.append(entity)
        
        detected_entities = all_entities
        
        # Format source documents
        sources = []
        for i, result in enumerate(search_results[:3]):  # Limit to top 3 sources for QA
            sources.append({
                "document_id": result.get("document_id", ""),
                "filename": result.get("filename", "Unknown"),
                "quoted_text": result["text"][:300] + "..." if len(result["text"]) > 300 else result["text"],
                "relevance_score": result["score"],
                "page_or_position": f"Source {i+1}",
                "metadata": result.get("metadata", {})
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return QAResponse(
            answer=answer,
            confidence_score=confidence_score,
            sources=sources,
            detected_entities=detected_entities,
            processing_time_ms=processing_time,
            mode=request.mode,
            llm_provider=settings.llm_provider if request.mode == QueryMode.RAG else None,
            llm_model=getattr(settings, f"{settings.llm_provider}_model", None) if request.mode == QueryMode.RAG else None
        )
        
    except Exception as e:
        logger.error(f"QA processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {str(e)}"
        )


@app.get("/collection/info", response_model=CollectionInfo)
async def get_collection_info():
    """
    Get information about the Qdrant collection.
    
    **Returns:**
    - Collection metadata and statistics
    """
    if not query_processor:
        raise HTTPException(status_code=503, detail="Query service not initialized")
    
    try:
        collection_info = query_processor.qdrant_client.get_collection(
            collection_name=settings.qdrant_collection_name
        )
        
        return CollectionInfo(
            name=settings.qdrant_collection_name,
            vector_size=collection_info.config.params.vectors.size,
            points_count=collection_info.points_count,
            status=collection_info.status
        )
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
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
