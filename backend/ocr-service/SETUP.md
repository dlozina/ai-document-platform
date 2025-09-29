# OCR Service Setup and Testing Guide

## Overview

Your OCR service is a production-ready FastAPI application that extracts text from PDFs and images using Tesseract OCR. The service supports both native PDF text extraction and OCR for scanned documents.

## Prerequisites

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

**Windows:**
- Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Download Poppler from: http://blog.alivate.com.au/poppler-windows/
- Add to PATH

### Python Dependencies

Install Python 3.12+ (latest stable version) and pip, then:

```bash
cd backend/ocr-service
pip install -r requirements.txt
```

## Running the Service

### Development Mode

```bash
# From the ocr-service directory
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The service will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Production Mode

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Docker

```bash
# Build the image
docker build -t ocr-service .

# Run the container
docker run -p 8000:8000 ocr-service

# Or use docker-compose
docker-compose up -d
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_ocr_processor.py

# Run with verbose output
pytest tests/ -v
```

### API Testing

#### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Extract text from image
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.png"

# Extract text from PDF
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Force OCR on PDF
curl -X POST "http://localhost:8000/extract?force_ocr=true" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Extract with layout information
curl -X POST "http://localhost:8000/extract-layout" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.png"
```

#### Using Python requests

```python
import requests

# Test health endpoint
response = requests.get("http://localhost:8000/health")
print(response.json())

# Test text extraction
with open("sample.png", "rb") as f:
    files = {"file": ("sample.png", f, "image/png")}
    response = requests.post("http://localhost:8000/extract", files=files)
    print(response.json())
```

#### Using the Interactive API Docs

1. Go to http://localhost:8000/docs
2. Click on any endpoint to expand it
3. Click "Try it out"
4. Upload a file and click "Execute"
5. View the response

## API Endpoints

### `GET /health`
Health check endpoint that verifies service status and dependencies.

**Response:**
```json
{
  "status": "healthy",
  "service": "ocr-service",
  "version": "1.0.0",
  "tesseract_available": true
}
```

### `POST /extract`
Extract text from uploaded PDF or image file.

**Parameters:**
- `file`: PDF or image file (required)
- `force_ocr`: Force OCR even for searchable PDFs (optional, default: false)
- `X-Tenant-ID`: Tenant identifier header (optional)
- `X-Document-ID`: Document identifier header (optional)

**Response:**
```json
{
  "document_id": "doc-123",
  "text": "Extracted text content...",
  "page_count": 1,
  "method": "ocr_image",
  "confidence": 85.5,
  "processing_time_ms": 1250.5,
  "file_size_bytes": 1024000,
  "filename": "document.pdf"
}
```

### `POST /extract-async`
Extract text asynchronously for large files.

**Parameters:**
- `file`: PDF or image file (required)
- `callback_url`: URL to POST results when complete (optional)
- `X-Tenant-ID`: Tenant identifier header (optional)
- `X-Document-ID`: Document identifier header (optional)

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "message": "File processing started in background"
}
```

### `POST /extract-layout`
Extract text with positional information (bounding boxes).

**Parameters:**
- `file`: PDF or image file (required)
- `X-Tenant-ID`: Tenant identifier header (optional)

**Response:**
```json
{
  "filename": "document.pdf",
  "word_count": 150,
  "words": [
    {
      "text": "Hello",
      "confidence": 95.0,
      "left": 100,
      "top": 50,
      "width": 40,
      "height": 20,
      "page_num": 1
    }
  ]
}
```

## Configuration

### Environment Variables

Copy `env.example` to `.env` and modify as needed:

```bash
cp env.example .env
```

Key settings:
- `OCR_DPI`: DPI for PDF to image conversion (default: 300)
- `MAX_FILE_SIZE_MB`: Maximum file size in MB (default: 10)
- `TESSERACT_LANG`: OCR language (default: eng)
- `LOG_LEVEL`: Logging level (default: INFO)

### Supported File Formats

**Images:** PNG, JPG, JPEG, TIFF, BMP, GIF
**PDFs:** PDF (both searchable and scanned)

## Troubleshooting

### Common Issues

1. **Tesseract not found**
   ```
   RuntimeError: Tesseract OCR is not installed
   ```
   **Solution:** Install Tesseract OCR system package

2. **Poppler not found (PDF processing)**
   ```
   pdf2image.exceptions.PDFPageCountError
   ```
   **Solution:** Install poppler-utils system package

3. **Permission denied**
   ```
   PermissionError: [Errno 13] Permission denied
   ```
   **Solution:** Check file permissions and Docker user settings

4. **Memory issues with large files**
   ```
   MemoryError
   ```
   **Solution:** Reduce OCR_DPI or increase system memory

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python -m uvicorn src.main:app --reload
```

### Performance Optimization

1. **For faster processing:**
   - Reduce `OCR_DPI` (150-200 for speed, 300+ for quality)
   - Use native PDF extraction when possible
   - Enable caching with Redis

2. **For better accuracy:**
   - Increase `OCR_DPI` (300-600)
   - Use appropriate language packs
   - Preprocess images for better contrast

## Monitoring

### Health Checks

The service provides a health endpoint at `/health` that checks:
- Service status
- Tesseract availability
- Basic functionality

### Logging

Logs include:
- Request processing times
- File sizes and types
- OCR confidence scores
- Error details

### Metrics (Optional)

Enable Prometheus metrics by setting `ENABLE_METRICS=true` in environment.

## Production Deployment

### Docker Deployment

```bash
# Build production image
docker build -t ocr-service:latest .

# Run with proper resource limits
docker run -d \
  --name ocr-service \
  --restart unless-stopped \
  --memory=2g \
  --cpus=2 \
  -p 8000:8000 \
  ocr-service:latest
```

### Kubernetes Deployment

Create deployment and service YAML files for Kubernetes deployment.

### Load Balancing

For high availability, deploy multiple instances behind a load balancer.

## Security Considerations

1. **File Upload Security:**
   - Validate file types and sizes
   - Scan for malware (consider ClamAV)
   - Use temporary storage for processing

2. **API Security:**
   - Implement rate limiting
   - Use HTTPS in production
   - Add authentication if needed
   - Validate tenant headers

3. **Data Privacy:**
   - Don't store processed files permanently
   - Implement data retention policies
   - Use secure logging practices
