# Testing Guide for Embedding Service

This guide provides multiple ways to test all endpoints of the Embedding Service.

## Prerequisites

1. **Start the Service**: Make sure the embedding service is running on port 8002
2. **Start Qdrant**: Ensure Qdrant vector database is running on port 6333

### Quick Start Commands

```bash
# Using Docker Compose (Recommended)
cd backend/embedding-service
docker-compose up -d

# Or manually
python -m uvicorn src.main:app --host 0.0.0.0 --port 8002
```

## Testing Methods

### Method 1: Automated Python Test Script (Recommended)

I've created a comprehensive test script that tests all endpoints automatically:

```bash
cd backend/embedding-service
python test_endpoints.py
```

This script will:
- ✅ Test health endpoint
- ✅ Test single embedding generation
- ✅ Test batch embedding generation  
- ✅ Test similarity search
- ✅ Test file upload (creates test files automatically)
- ✅ Test collection information
- ✅ Test delete functionality
- ✅ Test error handling scenarios

### Method 2: Interactive API Documentation

The easiest way to test endpoints interactively:

1. **Open Browser**: Go to `http://localhost:8002/docs`
2. **Try Endpoints**: Click "Try it out" on any endpoint
3. **Fill Parameters**: Enter test data and click "Execute"
4. **View Results**: See the response and status codes

### Method 3: Curl Commands

Use the provided bash script or run individual curl commands:

```bash
# Make script executable and run
chmod +x test_endpoints.sh
./test_endpoints.sh

# Or run individual commands:
```

#### Health Check
```bash
curl -X GET "http://localhost:8002/health"
```

#### Generate Single Embedding
```bash
curl -X POST "http://localhost:8002/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test sentence for embedding generation.",
    "document_id": "test_doc_001",
    "metadata": {"source": "curl_test", "category": "demo"}
  }'
```

#### Batch Embedding Generation
```bash
curl -X POST "http://localhost:8002/embed-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "First batch test sentence.",
      "Second batch test sentence.",
      "Third batch test sentence."
    ],
    "document_ids": ["batch_doc_1", "batch_doc_2", "batch_doc_3"]
  }'
```

#### Similarity Search
```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test sentence",
    "limit": 5,
    "score_threshold": 0.5
  }'
```

#### File Upload
```bash
# Create a test file first
echo "This is test content for file upload." > test.txt

# Upload the file
curl -X POST "http://localhost:8002/embed-file" \
  -F "file=@test.txt" \
  -H "X-Document-ID: curl_file_test"
```

#### Collection Information
```bash
curl -X GET "http://localhost:8002/collection/info"
```

#### Delete Embedding
```bash
curl -X DELETE "http://localhost:8002/embedding/test_doc_001"
```

### Method 4: Using HTTPie (Alternative to curl)

If you prefer HTTPie over curl:

```bash
# Install HTTPie
pip install httpie

# Test endpoints
http GET localhost:8002/health
http POST localhost:8002/embed text="Test sentence"
http POST localhost:8002/search query="test query" limit:=5
```

### Method 5: Using Postman or Insomnia

1. **Import Collection**: Create a new collection
2. **Add Requests**: Add requests for each endpoint
3. **Set Base URL**: `http://localhost:8002`
4. **Test Endpoints**: Run requests individually or as a collection

## Endpoint Reference

### 1. Health Check
- **URL**: `GET /health`
- **Purpose**: Check service and dependencies status
- **Response**: Service status, Qdrant availability, model status

### 2. Generate Embedding
- **URL**: `POST /embed`
- **Purpose**: Generate embedding for single text
- **Body**: `{"text": "your text", "document_id": "optional", "metadata": {}}`
- **Response**: Embedding vector, metadata, processing time

### 3. Batch Embeddings
- **URL**: `POST /embed-batch`
- **Purpose**: Generate embeddings for multiple texts
- **Body**: `{"texts": ["text1", "text2"], "document_ids": ["id1", "id2"]}`
- **Response**: List of embeddings, batch processing time

### 4. Similarity Search
- **URL**: `POST /search`
- **Purpose**: Find similar texts using vector similarity
- **Body**: `{"query": "search text", "limit": 10, "score_threshold": 0.8}`
- **Response**: Similar texts with scores and metadata

### 5. File Upload
- **URL**: `POST /embed-file`
- **Purpose**: Process uploaded text files
- **Body**: Multipart form with file
- **Headers**: `X-Document-ID` (optional)
- **Response**: Extracted text, embedding, processing info

### 6. Collection Info
- **URL**: `GET /collection/info`
- **Purpose**: Get Qdrant collection information
- **Response**: Collection name, vector size, point count, status

### 7. Delete Embedding
- **URL**: `DELETE /embedding/{point_id}`
- **Purpose**: Delete specific embedding from collection
- **Response**: Success confirmation

## Test Scenarios

### Basic Functionality Tests
1. **Health Check**: Verify service is running
2. **Single Embedding**: Generate embedding for simple text
3. **Batch Embedding**: Process multiple texts
4. **Search**: Find similar texts
5. **File Upload**: Process different file types

### Error Handling Tests
1. **Empty Text**: Send empty string
2. **Text Too Long**: Send text exceeding max length
3. **Invalid JSON**: Send malformed JSON
4. **Missing Fields**: Send requests without required fields
5. **File Too Large**: Upload oversized file

### Performance Tests
1. **Large Batch**: Process 100 texts at once
2. **Long Text**: Process very long text
3. **Concurrent Requests**: Send multiple requests simultaneously
4. **Memory Usage**: Monitor memory consumption

### Integration Tests
1. **End-to-End**: Generate → Store → Search → Delete workflow
2. **File Processing**: Upload → Extract → Embed → Search
3. **Metadata Filtering**: Search with metadata filters
4. **Tenant Isolation**: Test multi-tenant scenarios

## Expected Responses

### Successful Health Check
```json
{
  "status": "healthy",
  "service": "embedding-service",
  "version": "1.0.0",
  "qdrant_available": true,
  "embedding_model_loaded": true
}
```

### Successful Embedding Generation
```json
{
  "document_id": "test_doc_001",
  "text": "This is a test sentence",
  "embedding": [0.1, 0.2, 0.3, ...],
  "embedding_dimension": 384,
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "processing_time_ms": 45.2,
  "text_length": 25,
  "filename": null
}
```

### Successful Search Results
```json
{
  "query": "test query",
  "results": [
    {
      "id": "doc_1",
      "score": 0.95,
      "text": "Similar text content",
      "metadata": {"source": "test"}
    }
  ],
  "total_results": 1,
  "search_time_ms": 12.5
}
```

## Troubleshooting

### Common Issues

#### Service Not Responding
```bash
# Check if service is running
curl http://localhost:8002/health

# Check Docker containers
docker-compose ps

# Check logs
docker-compose logs embedding-service
```

#### Qdrant Connection Issues
```bash
# Check Qdrant status
curl http://localhost:6333/health

# Check Qdrant logs
docker-compose logs qdrant
```

#### Model Loading Issues
- Check available disk space
- Verify internet connection for model download
- Check logs for model loading errors

#### Memory Issues
- Use smaller batch sizes
- Use CPU-only PyTorch installation
- Monitor memory usage during testing

### Debug Mode

Enable debug logging:
```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or in .env file
LOG_LEVEL=DEBUG
```

## Performance Benchmarks

### Expected Performance (CPU)
- **Single Embedding**: 50-100ms
- **Batch (10 texts)**: 200-400ms
- **Search (1000 vectors)**: 10-50ms
- **File Upload**: 100-500ms (depending on file size)

### Expected Performance (GPU)
- **Single Embedding**: 10-20ms
- **Batch (10 texts)**: 50-100ms
- **Search (1000 vectors)**: 5-20ms
- **File Upload**: 50-200ms

## Next Steps

After testing:
1. **Monitor Performance**: Use the health endpoint for monitoring
2. **Scale Testing**: Test with larger datasets
3. **Production Setup**: Configure for production deployment
4. **Integration**: Integrate with your application
5. **Optimization**: Tune parameters for your use case

## Additional Resources

- **API Documentation**: `http://localhost:8002/docs`
- **Qdrant Dashboard**: `http://localhost:6333/dashboard`
- **Service Logs**: `docker-compose logs -f embedding-service`
- **Qdrant Logs**: `docker-compose logs -f qdrant`
