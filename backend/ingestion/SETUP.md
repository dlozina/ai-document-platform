# Ingestion Service Setup and Testing Guide

## Overview

The Ingestion Service is a production-ready FastAPI application that handles multi-tenant document uploads, metadata storage in PostgreSQL, and file persistence in MinIO object storage. It uses Redis + Celery for reliable task processing and integrates with OCR, NER, and Embedding services for document processing. Flower provides real-time monitoring of the processing pipeline.

## Prerequisites

### System Dependencies

**PostgreSQL Database:**
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Or use Docker
docker run --name postgres -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres:15-alpine
```

**MinIO Object Storage:**
```bash
# Using Docker (recommended)
docker run -p 9000:9000 -p 9001:9001 --name minio \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"
```

**Redis (for Celery):**
```bash
# Using Docker (recommended)
docker run -p 6379:6379 --name redis redis:7-alpine
```

**Python Dependencies:**
```bash
cd backend/ingestion
pip install -r requirements.txt
```

## Running the Service

### Development Mode

```bash
# From the ingestion-service directory
python -m uvicorn src.main:app --host 0.0.0.0 --port 8003 --reload
```

The service will be available at:
- **API**: http://localhost:8003
- **Interactive Docs**: http://localhost:8003/docs
- **ReDoc**: http://localhost:8003/redoc
- **Flower Monitoring**: http://localhost:5555 (admin/admin)

### Production Mode

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8003 --workers 4
```

### Using Docker Compose

```bash
# Start all services (PostgreSQL, MinIO, Redis, Ingestion Service, Celery Workers, Flower)
docker-compose up -d

# View logs
docker-compose logs -f ingestion-service
docker-compose logs -f celery-worker
docker-compose logs -f flower

# Stop services
docker-compose down
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest tests/ -v
```

### API Testing

#### Using curl

```bash
# Health check
curl http://localhost:8003/health

# Upload single file
curl -X POST "http://localhost:8003/upload" \
  -H "X-Tenant-ID: test-tenant" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.pdf"

# Upload multiple files
curl -X POST "http://localhost:8003/upload/batch" \
  -H "X-Tenant-ID: test-tenant" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@sample1.pdf" \
  -F "files=@sample2.png"

# List documents
curl -X GET "http://localhost:8003/documents" \
  -H "X-Tenant-ID: test-tenant"

# Get document by ID
curl -X GET "http://localhost:8003/documents/{document_id}" \
  -H "X-Tenant-ID: test-tenant"

# Download document
curl -X GET "http://localhost:8003/documents/{document_id}/download" \
  -H "X-Tenant-ID: test-tenant" \
  -o downloaded_file.pdf

# Get processing status
curl -X GET "http://localhost:8003/documents/{document_id}/processing" \
  -H "X-Tenant-ID: test-tenant"

# Update document metadata
curl -X PUT "http://localhost:8003/documents/{document_id}" \
  -H "X-Tenant-ID: test-tenant" \
  -H "Content-Type: application/json" \
  -d '{"tags": ["important"], "description": "Updated description"}'

# Delete document
curl -X DELETE "http://localhost:8003/documents/{document_id}" \
  -H "X-Tenant-ID: test-tenant"

# Get tenant quota
curl -X GET "http://localhost:8003/tenants/test-tenant/quota"
```

#### Using Python requests

```python
import requests

# Test health endpoint
response = requests.get("http://localhost:8003/health")
print(response.json())

# Test file upload
headers = {"X-Tenant-ID": "test-tenant"}
with open("sample.pdf", "rb") as f:
    files = {"file": ("sample.pdf", f, "application/pdf")}
    response = requests.post("http://localhost:8003/upload", files=files, headers=headers)
    print(response.json())

# Test document listing
response = requests.get("http://localhost:8003/documents", headers=headers)
print(response.json())
```

## API Endpoints

### `GET /health`
Health check endpoint that verifies service status and dependencies.

**Response:**
```json
{
  "status": "healthy",
  "service": "ingestion-service",
  "version": "1.0.0",
  "database_connected": true,
  "minio_connected": true,
  "dependencies": {
    "database": true,
    "minio": true,
    "ocr_service": true,
    "ner_service": true,
    "embedding_service": true
  }
}
```

### `POST /upload`
Upload a single file.

**Parameters:**
- `file`: File to upload (required)
- `X-Tenant-ID`: Tenant identifier (required header)
- `tags`: Document tags (optional query parameter)
- `description`: Document description (optional query parameter)
- `retention_days`: Retention period in days (optional query parameter)

**Response:**
```json
{
  "document_id": "uuid-string",
  "filename": "document.pdf",
  "file_size_bytes": 1024000,
  "content_type": "application/pdf",
  "file_hash": "sha256-hash",
  "upload_timestamp": "2024-01-01T12:00:00Z",
  "processing_status": "pending",
  "storage_path": "tenant123/2024/01/01/hash_document.pdf",
  "message": "File uploaded successfully, processing started"
}
```

### `POST /upload/batch`
Upload multiple files in a batch.

**Parameters:**
- `files`: List of files to upload (required)
- `X-Tenant-ID`: Tenant identifier (required header)

**Response:**
```json
{
  "batch_id": "uuid-string",
  "total_files": 3,
  "successful_uploads": 2,
  "failed_uploads": 1,
  "documents": [...],
  "errors": [
    {
      "filename": "invalid.txt",
      "error": "Unsupported file type"
    }
  ]
}
```

### `GET /documents`
List documents for a tenant.

**Parameters:**
- `X-Tenant-ID`: Tenant identifier (required header)
- `page`: Page number (default: 1)
- `page_size`: Page size (default: 20, max: 100)
- `file_type`: Filter by file type (optional)
- `processing_status`: Filter by processing status (optional)

### `GET /documents/{document_id}`
Get document by ID.

**Parameters:**
- `document_id`: Document identifier
- `X-Tenant-ID`: Tenant identifier (required header)

### `GET /documents/{document_id}/download`
Download original document file.

**Parameters:**
- `document_id`: Document identifier
- `X-Tenant-ID`: Tenant identifier (required header)

**Returns:** File content as stream

### `GET /documents/{document_id}/processing`
Get processing status for a document.

**Parameters:**
- `document_id`: Document identifier
- `X-Tenant-ID`: Tenant identifier (required header)

### `PUT /documents/{document_id}`
Update document metadata.

**Parameters:**
- `document_id`: Document identifier
- `X-Tenant-ID`: Tenant identifier (required header)
- Request body: Update data (tags, description, retention_days)

### `DELETE /documents/{document_id}`
Delete a document (soft delete).

**Parameters:**
- `document_id`: Document identifier
- `X-Tenant-ID`: Tenant identifier (required header)

### `GET /tenants/{tenant_id}/quota`
Get tenant quota information.

**Parameters:**
- `tenant_id`: Tenant identifier

## Configuration

### Environment Variables

Copy `env.example` to `.env` and modify as needed:

```bash
cp env.example .env
```

Key settings:
- `DATABASE_URL`: PostgreSQL connection string
- `MINIO_ENDPOINT`: MinIO server endpoint
- `MINIO_ACCESS_KEY`: MinIO access key
- `MINIO_SECRET_KEY`: MinIO secret key
- `MAX_FILE_SIZE_MB`: Maximum file size in MB
- `REQUIRE_TENANT_HEADER`: Whether tenant header is required
- `OCR_SERVICE_URL`: OCR service endpoint
- `NER_SERVICE_URL`: NER service endpoint
- `EMBEDDING_SERVICE_URL`: Embedding service endpoint

### Supported File Types

**Images:** PNG, JPG, JPEG, TIFF, BMP, GIF
**PDFs:** PDF (both searchable and scanned)
**Documents:** DOC, DOCX
**Text:** TXT

## Multi-tenant Features

### Tenant Isolation
- Each tenant has isolated storage buckets in MinIO
- Database records are filtered by tenant_id
- Tenant-specific quotas and limits

### Tenant Headers
- `X-Tenant-ID`: Required tenant identifier header
- All operations are scoped to the specified tenant

### Quota Management
- Storage quota per tenant
- File count limits
- File size limits
- Data retention policies

## Processing Pipeline

The service integrates with other microservices for document processing:

1. **OCR Service**: Extract text from images and PDFs
2. **NER Service**: Extract named entities from text
3. **Embedding Service**: Generate vector embeddings for search

### Processing Flow
1. File uploaded and stored in MinIO
2. Metadata saved to PostgreSQL
3. Processing jobs created for each enabled service
4. Services process documents asynchronously
5. Results stored back in database

## Celery Task Processing

The service uses Redis + Celery for reliable task processing:

### Queue Architecture
- **ocr_queue**: OCR processing (highest priority)
- **ner_queue**: NER processing (medium priority)
- **embedding_queue**: Embedding processing (medium priority)
- **completion_queue**: Final processing steps (low priority)
- **dead_letter_queue**: Failed jobs for manual inspection

### Processing Flow
1. File uploaded and stored in MinIO
2. Metadata saved to PostgreSQL
3. OCR task queued in Redis
4. Celery worker processes OCR task
5. NER and Embedding tasks queued in parallel after OCR
6. Celery workers process NER and Embedding tasks
7. Completion task updates final status
8. Results stored back in database

### Monitoring with Flower
- **Real-time monitoring**: http://localhost:5555
- **Task status**: Track individual task progress
- **Worker health**: Monitor worker status and performance
- **Queue statistics**: View queue depth and processing rates
- **Error tracking**: Monitor failed tasks and retries
- **Performance metrics**: Processing times and throughput

## Troubleshooting

### Common Issues

1. **Database connection failed**
   ```
   sqlalchemy.exc.OperationalError: could not connect to server
   ```
   **Solution:** Check PostgreSQL is running and connection string is correct

2. **MinIO connection failed**
   ```
   minio.error.S3Error: The specified bucket does not exist
   ```
   **Solution:** Check MinIO is running and credentials are correct

3. **Tenant header missing**
   ```
   HTTPException: X-Tenant-ID header is required
   ```
   **Solution:** Include X-Tenant-ID header in requests

4. **File too large**
   ```
   HTTPException: File too large. Maximum size: 100MB
   ```
   **Solution:** Reduce file size or increase MAX_FILE_SIZE_MB setting

5. **Processing service unavailable**
   ```
   httpx.ConnectError: Connection refused
   ```
   **Solution:** Ensure OCR/NER/Embedding services are running

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python -m uvicorn src.main:app --reload
```

### Performance Optimization

1. **For better throughput:**
   - Increase database pool size
   - Use connection pooling
   - Enable Redis caching
   - Increase worker processes

2. **For better reliability:**
   - Use database connection pooling
   - Implement retry logic for external services
   - Add circuit breakers for service calls

## Monitoring

### Health Checks

The service provides a health endpoint at `/health` that checks:
- Service status
- Database connectivity
- MinIO connectivity
- External service availability

### Logging

Logs include:
- Upload processing times
- File sizes and types
- Processing job status
- Error details
- Tenant activity

### Metrics (Optional)

Enable Prometheus metrics by setting `ENABLE_METRICS=true` in environment.

## Production Deployment

### Docker Deployment

```bash
# Build production image
docker build -t ingestion-service:latest .

# Run with proper resource limits
docker run -d \
  --name ingestion-service \
  --restart unless-stopped \
  --memory=2g \
  --cpus=2 \
  -p 8003:8003 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/ingestion \
  -e MINIO_ENDPOINT=minio:9000 \
  ingestion-service:latest
```

### Kubernetes Deployment

Create deployment and service YAML files for Kubernetes deployment.

### Load Balancing

For high availability, deploy multiple instances behind a load balancer.

## Security Considerations

1. **File Upload Security:**
   - Validate file types and sizes
   - Scan for malware (consider ClamAV integration)
   - Use temporary storage for processing
   - Implement rate limiting

2. **API Security:**
   - Use HTTPS in production
   - Implement authentication and authorization
   - Validate tenant headers
   - Add request rate limiting

3. **Data Privacy:**
   - Implement data retention policies
   - Use secure logging practices
   - Encrypt sensitive data
   - Regular security audits

4. **Multi-tenant Security:**
   - Ensure tenant data isolation
   - Validate tenant permissions
   - Monitor cross-tenant access attempts
   - Implement tenant-specific security policies
