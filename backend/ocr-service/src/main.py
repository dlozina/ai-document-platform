"""
OCR Service API

FastAPI application that exposes OCR functionality via REST endpoints.
"""

import logging
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from ocr_processor import OCRProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global OCR processor instance
ocr_processor: Optional[OCRProcessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global ocr_processor
    
    logger.info("Starting OCR Service...")
    ocr_processor = OCRProcessor(dpi=300, language='eng')
    logger.info("OCR Service ready")
    
    yield
    
    logger.info("Shutting down OCR Service...")


# Create FastAPI app
app = FastAPI(
    title="OCR Service",
    description="Extract text from PDFs and images",
    version="1.0.0",
    lifespan=lifespan
)


# Pydantic models for request/response
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
    status: str
    service: str
    version: str
    tesseract_available: bool


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None


# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and dependencies availability.
    """
    try:
        import pytesseract
        tesseract_version = pytesseract.get_tesseract_version()
        tesseract_available = True
    except Exception:
        tesseract_available = False
    
    return HealthResponse(
        status="healthy" if tesseract_available else "degraded",
        service="ocr-service",
        version="1.0.0",
        tesseract_available=tesseract_available
    )


@app.post("/extract", response_model=OCRResponse)
async def extract_text(
    file: UploadFile = File(..., description="PDF or image file to process"),
    force_ocr: bool = False,
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    document_id: Optional[str] = Header(None, alias="X-Document-ID")
):
    """
    Extract text from uploaded PDF or image file.
    
    **Supported formats:**
    - PDF (.pdf)
    - Images (.png, .jpg, .jpeg, .tiff, .bmp, .gif)
    
    **Parameters:**
    - **file**: The document file to process (required)
    - **force_ocr**: Force OCR even for searchable PDFs (default: false)
    - **X-Tenant-ID**: Tenant identifier (header)
    - **X-Document-ID**: Document identifier (header)
    
    **Returns:**
    - Extracted text and metadata
    """
    if not ocr_processor:
        raise HTTPException(
            status_code=503,
            detail="OCR service not initialized"
        )
    
    # Validate file size (max 10MB for this endpoint)
    max_size = 10 * 1024 * 1024  # 10MB
    content = await file.read()
    file_size = len(content)
    
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {max_size / 1024 / 1024:.0f}MB"
        )
    
    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded"
        )
    
    try:
        logger.info(
            f"Processing file: {file.filename} "
            f"(size: {file_size / 1024:.2f}KB, tenant: {tenant_id})"
        )
        
        # Process the file
        result = ocr_processor.process_file(
            file_content=content,
            filename=file.filename,
            force_ocr=force_ocr
        )
        
        # Add metadata
        result['file_size_bytes'] = file_size
        result['filename'] = file.filename
        result['document_id'] = document_id
        
        logger.info(
            f"Successfully processed {file.filename}: "
            f"{len(result['text'])} characters extracted"
        )
        
        return OCRResponse(**result)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )


@app.post("/extract-async")
async def extract_text_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    callback_url: Optional[str] = None,
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    document_id: Optional[str] = Header(None, alias="X-Document-ID")
):
    """
    Extract text asynchronously (for large files).
    
    Process the file in the background and optionally POST results to callback_url.
    
    **Returns:**
    - Job ID for status tracking
    """
    import uuid
    
    job_id = str(uuid.uuid4())
    content = await file.read()
    
    # Add background task
    background_tasks.add_task(
        process_file_background,
        job_id=job_id,
        content=content,
        filename=file.filename,
        callback_url=callback_url,
        tenant_id=tenant_id,
        document_id=document_id
    )
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "File processing started in background"
    }


async def process_file_background(
    job_id: str,
    content: bytes,
    filename: str,
    callback_url: Optional[str],
    tenant_id: Optional[str],
    document_id: Optional[str]
):
    """Background task for async processing."""
    try:
        logger.info(f"Background processing job {job_id} for {filename}")
        
        result = ocr_processor.process_file(content, filename)
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
                    timeout=10.0
                )
                logger.info(f"Results posted to {callback_url}")
        
        # TODO: Store results in database or message queue
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Background job {job_id} failed: {e}", exc_info=True)


@app.post("/extract-layout")
async def extract_with_layout(
    file: UploadFile = File(...),
    tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
):
    """
    Extract text with positional information (bounding boxes).
    
    Useful for:
    - Preserving document layout
    - Table extraction
    - Form field detection
    
    **Returns:**
    - List of words with bounding box coordinates
    """
    if not ocr_processor:
        raise HTTPException(status_code=503, detail="OCR service not initialized")
    
    content = await file.read()
    
    try:
        words = ocr_processor.extract_text_with_layout(content, file.filename)
        
        return {
            "filename": file.filename,
            "word_count": len(words),
            "words": words
        }
    
    except Exception as e:
        logger.error(f"Layout extraction error: {e}", exc_info=True)
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
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )