"""
NER Service API

FastAPI application that exposes Named Entity Recognition functionality via REST endpoints.
"""

import logging
import time
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from .ner_processor import NERProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global NER processor instance
ner_processor: Optional[NERProcessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global ner_processor
    
    logger.info("Starting NER Service...")
    ner_processor = NERProcessor(
        model_name="en_core_web_sm",
        fallback_model="en_core_web_lg"
    )
    logger.info("NER Service ready")
    
    yield
    
    logger.info("Shutting down NER Service...")


# Create FastAPI app
app = FastAPI(
    title="NER Service",
    description="""
    ## Production-ready Named Entity Recognition Service
    
    Extract named entities from text using advanced NLP technology.
    
    ### Features:
    - **Multiple spaCy models** (small and large)
    - **Entity type filtering** for specific entity types
    - **Batch processing** for multiple texts
    - **Confidence scoring** for entity detection
    - **Entity visualization** with HTML output
    - **Async processing** for large texts
    
    ### Supported Entity Types:
    - **PERSON**: People, including fictional
    - **ORG**: Companies, agencies, institutions
    - **GPE**: Countries, cities, states
    - **MONEY**: Monetary values
    - **PERCENT**: Percentages
    - **DATE**: Absolute or relative dates
    - **TIME**: Times smaller than a day
    - **LOC**: Non-GPE locations
    
    ### Models:
    1. **en_core_web_sm**: Small, fast model (default)
    2. **en_core_web_lg**: Large, accurate model (fallback)
    
    ### Performance Tips:
    - Use small model for speed, large model for accuracy
    - Batch processing is more efficient for multiple texts
    - Filter entity types to reduce processing time
    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "NER Service Team",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": "http://localhost:8001",
            "description": "Development server"
        },
        {
            "url": "https://api.example.com",
            "description": "Production server"
        }
    ]
)


from .models import (
    NERResponse, NERRequest, BatchNERRequest, BatchNERResponse,
    HealthResponse, ErrorResponse, NERStatsResponse, AsyncJobResponse
)


# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and spaCy models availability.
    
    **Returns:**
    - Service status (healthy/degraded)
    - Service information
    - Available spaCy models
    """
    if not ner_processor:
        return HealthResponse(
            status="degraded",
            service="ner-service",
            version="1.0.0",
            spacy_models_available={}
        )
    
    try:
        models_available = ner_processor.get_available_models()
        status = "healthy" if any(models_available.values()) else "degraded"
        
        return HealthResponse(
            status=status,
            service="ner-service",
            version="1.0.0",
            spacy_models_available=models_available
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            service="ner-service",
            version="1.0.0",
            spacy_models_available={}
        )


@app.post("/extract", response_model=NERResponse)
async def extract_entities(
    request: NERRequest,
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    document_id: Optional[str] = Header(None, alias="X-Document-ID")
):
    """
    Extract named entities from text.
    
    **Parameters:**
    - **text**: Text to process (required, max 1M characters)
    - **entity_types**: Specific entity types to extract (optional)
    - **include_confidence**: Include confidence scores (default: true)
    - **X-Tenant-ID**: Tenant identifier (header)
    - **X-Document-ID**: Document identifier (header)
    
    **Returns:**
    - Detected entities with positions and types
    - Processing metadata
    """
    if not ner_processor:
        raise HTTPException(
            status_code=503,
            detail="NER service not initialized"
        )
    
    try:
        logger.info(
            f"Processing text ({len(request.text)} chars) "
            f"for tenant: {tenant_id}"
        )
        
        # Process the text
        result = ner_processor.process_text(
            text=request.text,
            entity_types=request.entity_types,
            include_confidence=request.include_confidence
        )
        
        # Add metadata
        result['document_id'] = document_id
        
        logger.info(
            f"Successfully processed text: "
            f"{result['entity_count']} entities found"
        )
        
        return NERResponse(**result)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process text: {str(e)}"
        )


@app.post("/extract-batch", response_model=BatchNERResponse)
async def extract_entities_batch(
    request: BatchNERRequest,
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
):
    """
    Extract named entities from multiple texts in batch.
    
    **Parameters:**
    - **texts**: List of texts to process (max 100 texts)
    - **entity_types**: Specific entity types to extract (optional)
    - **include_confidence**: Include confidence scores (default: true)
    - **X-Tenant-ID**: Tenant identifier (header)
    
    **Returns:**
    - NER results for each text
    - Batch processing statistics
    """
    if not ner_processor:
        raise HTTPException(
            status_code=503,
            detail="NER service not initialized"
        )
    
    try:
        logger.info(
            f"Processing batch of {len(request.texts)} texts "
            f"for tenant: {tenant_id}"
        )
        
        # Process the batch
        result = ner_processor.process_batch(
            texts=request.texts,
            entity_types=request.entity_types,
            include_confidence=request.include_confidence,
            batch_size=100
        )
        
        logger.info(
            f"Successfully processed batch: "
            f"{result['batch_size']} texts in {result['total_processing_time_ms']:.2f}ms"
        )
        
        return BatchNERResponse(**result)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Batch processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process batch: {str(e)}"
        )


@app.post("/extract-async")
async def extract_entities_async(
    background_tasks: BackgroundTasks,
    request: NERRequest,
    callback_url: Optional[str] = None,
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    document_id: Optional[str] = Header(None, alias="X-Document-ID")
):
    """
    Extract named entities asynchronously (for large texts).
    
    Process the text in the background and optionally POST results to callback_url.
    
    **Returns:**
    - Job ID for status tracking
    """
    import uuid
    
    job_id = str(uuid.uuid4())
    
    # Add background task
    background_tasks.add_task(
        process_text_background,
        job_id=job_id,
        request=request,
        callback_url=callback_url,
        tenant_id=tenant_id,
        document_id=document_id
    )
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Text processing started in background"
    }


async def process_text_background(
    job_id: str,
    request: NERRequest,
    callback_url: Optional[str],
    tenant_id: Optional[str],
    document_id: Optional[str]
):
    """Background task for async processing."""
    try:
        logger.info(f"Background processing job {job_id}")
        
        result = ner_processor.process_text(
            text=request.text,
            entity_types=request.entity_types,
            include_confidence=request.include_confidence
        )
        result['job_id'] = job_id
        result['document_id'] = document_id
        result['tenant_id'] = tenant_id
        
        # If callback URL provided, POST results
        if callback_url:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(
                    callback_url,
                    json=result,
                    timeout=30.0
                )
                logger.info(f"Results posted to {callback_url}")
        
        # TODO: Store results in database or message queue
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Background job {job_id} failed: {e}", exc_info=True)


@app.post("/stats", response_model=NERStatsResponse)
async def get_entity_statistics(
    request: NERRequest,
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
):
    """
    Get statistics about entities in text.
    
    **Parameters:**
    - **text**: Text to analyze
    - **entity_types**: Specific entity types to analyze (optional)
    - **include_confidence**: Include confidence scores (default: true)
    - **X-Tenant-ID**: Tenant identifier (header)
    
    **Returns:**
    - Entity statistics and counts
    - Most common entities
    """
    if not ner_processor:
        raise HTTPException(
            status_code=503,
            detail="NER service not initialized"
        )
    
    try:
        # Process text to get entities
        result = ner_processor.process_text(
            text=request.text,
            entity_types=request.entity_types,
            include_confidence=request.include_confidence
        )
        
        # Generate statistics
        stats = ner_processor.get_entity_statistics(result['entities'])
        
        return NERStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Statistics generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate statistics: {str(e)}"
        )


@app.post("/visualize", response_class=HTMLResponse)
async def visualize_entities(
    request: NERRequest,
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
):
    """
    Generate HTML visualization of entities in text.
    
    **Parameters:**
    - **text**: Text to visualize
    - **entity_types**: Specific entity types to highlight (optional)
    - **include_confidence**: Include confidence scores (default: true)
    - **X-Tenant-ID**: Tenant identifier (header)
    
    **Returns:**
    - HTML page with highlighted entities
    """
    if not ner_processor:
        raise HTTPException(
            status_code=503,
            detail="NER service not initialized"
        )
    
    try:
        # Process text to get entities
        result = ner_processor.process_text(
            text=request.text,
            entity_types=request.entity_types,
            include_confidence=request.include_confidence
        )
        
        # Generate visualization
        html = ner_processor.visualize_entities(request.text, result['entities'])
        
        return HTMLResponse(content=html)
        
    except Exception as e:
        logger.error(f"Visualization error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate visualization: {str(e)}"
        )


@app.get("/models")
async def get_available_models():
    """
    Get information about available spaCy models.
    
    **Returns:**
    - List of available models and their status
    """
    if not ner_processor:
        raise HTTPException(
            status_code=503,
            detail="NER service not initialized"
        )
    
    try:
        models = ner_processor.get_available_models()
        return {
            "available_models": models,
            "primary_model": ner_processor.model_name,
            "fallback_model": ner_processor.fallback_model
        }
    except Exception as e:
        logger.error(f"Model info error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model information: {str(e)}"
        )


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
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
