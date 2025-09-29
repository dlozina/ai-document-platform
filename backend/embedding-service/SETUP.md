# Embedding Service Setup Guide

This guide provides detailed instructions for setting up and running the Embedding Service in various environments.

## Prerequisites

### System Requirements
- Python 3.12 or higher
- Docker and Docker Compose (for containerized deployment)
- At least 4GB RAM (8GB+ recommended for larger models)
- 2GB+ free disk space

### External Dependencies
- **Qdrant Vector Database**: Required for vector storage and similarity search
- **Sentence-Transformers Models**: Downloaded automatically on first use

## Installation Methods

### Method 1: Docker Compose (Recommended)

This is the easiest way to get started with both the embedding service and Qdrant.

1. **Clone and Navigate**:
```bash
cd backend/embedding-service
```

2. **Start Services**:
```bash
docker-compose up -d
```

3. **Verify Installation**:
```bash
curl http://localhost:8001/health
```

The service will be available at `http://localhost:8001` and Qdrant at `http://localhost:6333`.

### Method 2: Manual Python Installation

1. **Create Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Start Qdrant** (using Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant:v1.7.0
```

4. **Configure Environment** (optional):
```bash
cp env.example .env
# Edit .env with your settings
```

5. **Run the Service**:
```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8002 --reload
```

### Method 3: Production Deployment

1. **Build Docker Image**:
```bash
docker build -t embedding-service .
```

2. **Run with External Qdrant**:
```bash
docker run -p 8001:8001 \
  -e QDRANT_HOST=your-qdrant-host \
  -e QDRANT_PORT=6333 \
  embedding-service
```

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Application Settings
APP_NAME=Embedding Service
APP_VERSION=1.0.0
LOG_LEVEL=INFO

# Server Settings
HOST=0.0.0.0
PORT=8001
WORKERS=4

# Embedding Model Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
MAX_TEXT_LENGTH=512
BATCH_SIZE=32

# Qdrant Settings
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=embeddings
QDRANT_VECTOR_SIZE=384

# File Processing Settings
MAX_FILE_SIZE_MB=10

# Performance Settings
ENABLE_CACHING=false
CACHE_TTL_SECONDS=3600
REDIS_URL=

# Security Settings
ENABLE_CORS=true
CORS_ORIGINS=["*"]
REQUIRE_TENANT_HEADER=false

# Monitoring Settings
ENABLE_METRICS=true
ENABLE_TRACING=false
```

### Model Configuration

#### Choosing an Embedding Model

The service supports any sentence-transformer model. Choose based on your needs:

**Fast Models (384 dimensions)**:
- `sentence-transformers/all-MiniLM-L6-v2` (default) - Good balance of speed and quality
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` - Multilingual support

**High-Quality Models (768 dimensions)**:
- `sentence-transformers/all-mpnet-base-v2` - Best quality for English
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` - Multilingual, high quality

**Specialized Models**:
- `sentence-transformers/all-distilroberta-v1` - Distilled model, faster
- `sentence-transformers/msmarco-distilbert-base-v4` - Optimized for search

#### Changing the Model

1. **Update Environment Variable**:
```bash
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
QDRANT_VECTOR_SIZE=768  # Must match model dimension
```

2. **Restart the Service**:
```bash
docker-compose restart embedding-service
```

### Qdrant Configuration

#### Local Qdrant Setup

1. **Using Docker**:
```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.7.0
```

2. **Using Docker Compose** (included in our setup):
```yaml
qdrant:
  image: qdrant/qdrant:v1.7.0
  ports:
    - "6333:6333"
    - "6334:6334"
  volumes:
    - qdrant_storage:/qdrant/storage
```

#### External Qdrant Setup

1. **Qdrant Cloud**:
```bash
QDRANT_HOST=your-cluster.qdrant.tech
QDRANT_PORT=6333
QDRANT_API_KEY=your-api-key
```

2. **Self-hosted Qdrant**:
```bash
QDRANT_HOST=your-qdrant-server.com
QDRANT_PORT=6333
QDRANT_API_KEY=optional-api-key
```

## Testing the Installation

### 1. Health Check
```bash
curl http://localhost:8001/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "embedding-service",
  "version": "1.0.0",
  "qdrant_available": true,
  "embedding_model_loaded": true
}
```

### 2. Generate Embedding
```bash
curl -X POST "http://localhost:8001/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test sentence"}'
```

### 3. Search Similar
```bash
curl -X POST "http://localhost:8001/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "limit": 5}'
```

### 4. Upload File
```bash
curl -X POST "http://localhost:8001/embed-file" \
  -F "file=@test.txt"
```

## Performance Tuning

### Model Selection
- **Speed**: Use `all-MiniLM-L6-v2` (384 dims)
- **Quality**: Use `all-mpnet-base-v2` (768 dims)
- **Multilingual**: Use `paraphrase-multilingual-*` models

### Batch Processing
- Use `/embed-batch` for multiple texts (more efficient)
- Optimal batch size: 16-32 texts
- Maximum batch size: 100 texts

### Qdrant Optimization
- Use SSD storage for Qdrant data
- Allocate sufficient RAM (2GB+ recommended)
- Consider Qdrant clustering for large datasets

### Server Configuration
- Increase `WORKERS` for higher concurrency
- Use `uvicorn` with `--workers` for production
- Consider using a reverse proxy (nginx)

## Troubleshooting

### Common Issues

#### 1. Model Download Fails
**Error**: `OSError: [Errno 28] No space left on device`

**Solution**:
```bash
# Check disk space
df -h

# Clear model cache
rm -rf ~/.cache/huggingface/

# Use a smaller model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

#### 2. Qdrant Connection Failed
**Error**: `Failed to connect to Qdrant`

**Solution**:
```bash
# Check if Qdrant is running
curl http://localhost:6333/collections

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant:v1.7.0

# Check network connectivity
telnet localhost 6333
```

#### 3. Out of Memory
**Error**: `CUDA out of memory` or `RuntimeError: [Errno 12] Cannot allocate memory`

**Solution**:
```bash
# Use CPU-only model
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Reduce batch size
BATCH_SIZE=8

# Use smaller model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

#### 4. Slow Performance
**Solutions**:
- Use GPU if available
- Increase batch size
- Use faster model
- Optimize Qdrant configuration

### Logs and Debugging

#### Enable Debug Logging
```bash
LOG_LEVEL=DEBUG
```

#### View Service Logs
```bash
# Docker Compose
docker-compose logs -f embedding-service

# Docker
docker logs -f embedding-service-container

# Manual
tail -f logs/embedding-service.log
```

#### Qdrant Logs
```bash
# Docker
docker logs -f qdrant-container

# Check Qdrant status
curl http://localhost:6333/health
```

## Production Deployment

### Security Considerations
1. **API Keys**: Use Qdrant API keys for authentication
2. **CORS**: Configure appropriate CORS origins
3. **Rate Limiting**: Implement rate limiting
4. **HTTPS**: Use HTTPS in production
5. **Firewall**: Restrict access to necessary ports

### Monitoring
1. **Health Checks**: Monitor `/health` endpoint
2. **Metrics**: Enable Prometheus metrics
3. **Logging**: Centralized logging with structured logs
4. **Alerting**: Set up alerts for service failures

### Scaling
1. **Horizontal Scaling**: Run multiple service instances
2. **Load Balancing**: Use nginx or similar
3. **Qdrant Clustering**: For large datasets
4. **Caching**: Implement Redis caching for frequent queries

### Backup
1. **Qdrant Data**: Regular backups of Qdrant storage
2. **Model Cache**: Backup downloaded models
3. **Configuration**: Version control for configuration files

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs for error messages
3. Test with the health check endpoint
4. Verify all dependencies are properly installed

## Next Steps

After successful setup:
1. Explore the API documentation at `http://localhost:8001/docs`
2. Test with your own data
3. Configure monitoring and alerting
4. Set up production deployment
5. Consider performance optimization based on your use case
