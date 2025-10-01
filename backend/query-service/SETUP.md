# Query Service Setup Guide

This guide will help you set up and run the Query Service for semantic search and question-answering.

## Prerequisites

Before starting, ensure you have:

- **Python 3.12+** installed
- **PostgreSQL** database running
- **Qdrant** vector database running
- **Docker** and **Docker Compose** (optional, for containerized setup)
- **Mistral API Key** (for LLM functionality)

### Getting a Mistral API Key

1. Visit [Mistral AI Console](https://console.mistral.ai/)
2. Sign up or log in to your account
3. Navigate to the API Keys section
4. Create a new API key
5. Copy the key and keep it secure

**Note**: You'll need this API key to use the RAG (Retrieval-Augmented Generation) functionality.

## Quick Setup

### 1. Environment Setup

```bash
# Navigate to query service directory
cd backend/query-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Download Required Models

```bash
# Download spaCy English model
python -m spacy download en_core_web_sm
```

### 3. Configuration

```bash
# Copy environment template
cp env.example .env

# Edit configuration
nano .env
```

**Required Environment Variables**:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/abysalto

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=embeddings

# LLM Provider (choose one)
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here

# OR
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# OR
LLM_PROVIDER=mistral
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 4. Start the Service

```bash
# Run the service
uvicorn src.main:app --host 0.0.0.0 --port 8004 --reload
```

The service will be available at `http://localhost:8004`

## Docker Setup (Recommended)

### Method 1: Run with Main Docker Compose

This is the recommended approach as it integrates with your existing services:

```bash
# Navigate to backend directory
cd backend

# Set your Mistral API key
export MISTRAL_API_KEY=your_actual_mistral_api_key_here

# Start all services including query service
docker-compose up --build

# Or start just the query service and its dependencies
docker-compose up postgres qdrant query-service
```

### Method 2: Run Query Service Standalone

If you want to run just the query service:

```bash
# Navigate to query service directory
cd backend/query-service

# Set your Mistral API key
export MISTRAL_API_KEY=your_actual_mistral_api_key_here

# Run standalone (connects to existing services)
docker-compose up --build
```

### Method 3: Using .env File

Create a `.env` file in the `backend` directory:

```bash
cd backend
echo "MISTRAL_API_KEY=your_actual_mistral_api_key_here" > .env
docker-compose up --build
```

**Security Note**: Never commit your API key to version control! The `.env` file is already in `.gitignore`.

## Verification

### 1. Health Check

```bash
curl http://localhost:8004/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "query-service",
  "version": "1.0.0",
  "qdrant_available": true,
  "database_connected": true,
  "embedding_model_loaded": true,
  "llm_available": true
}
```

### 2. Test Query

```bash
curl -X POST "http://localhost:8004/query" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: test-tenant" \
  -d '{
    "query": "What is Python programming?",
    "mode": "semantic_search",
    "top_k": 5
  }'
```

## Service Dependencies

### PostgreSQL Setup

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb abysalto

# Create user
sudo -u postgres createuser -P user
```

### Qdrant Setup

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant:latest

# Or download binary
wget https://github.com/qdrant/qdrant/releases/latest/download/qdrant
chmod +x qdrant
./qdrant
```

### Collection Setup

The service expects a Qdrant collection named `embeddings` with 384-dimensional vectors. If you're using the embedding service, this should already exist.

## Configuration Details

### Embedding Models

The service uses sentence-transformers models for embedding generation:

- **Default**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Alternative**: `sentence-transformers/all-mpnet-base-v2` (768 dimensions)

To change the model:
```env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
QDRANT_VECTOR_SIZE=768
```

### LLM Providers

#### OpenAI
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo
```

#### Anthropic
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-sonnet-20240229
```

#### Mistral
```env
LLM_PROVIDER=mistral
MISTRAL_API_KEY=...
MISTRAL_MODEL=mistral-small
```

### Performance Tuning

```env
# Query settings
DEFAULT_TOP_K=10
MAX_TOP_K=100
DEFAULT_SCORE_THRESHOLD=0.7
MAX_CONTEXT_LENGTH=4000

# Reranking
ENABLE_RERANKING=true
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Caching
ENABLE_CACHING=false
REDIS_URL=redis://localhost:6379/0
```

## Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

### Test Data

For testing, you'll need documents with embeddings in your Qdrant collection. The ingestion service should populate this data.

## Monitoring

### Health Endpoints

- `GET /health` - Service health status
- `GET /collection/info` - Qdrant collection information

### Logs

```bash
# View logs
tail -f logs/query-service.log

# Docker logs
docker-compose logs -f query-service
```

### Metrics

The service logs query metrics including:
- Processing time
- Confidence scores
- Result counts
- Query patterns

## Troubleshooting

### Common Issues

1. **"spaCy model not found"**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **"Qdrant connection failed"**
   - Check Qdrant is running: `curl http://localhost:6333/collections`
   - Verify host/port in `.env`
   - Check collection exists

3. **"Database connection failed"**
   - Verify PostgreSQL is running
   - Check connection string format
   - Ensure database exists

4. **"LLM API error"**
   - Verify API key is set correctly
   - Check API rate limits
   - Verify model name is correct

5. **"No documents found"**
   - Ensure documents are processed and have embeddings
   - Check tenant_id matches your data
   - Verify collection has points

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=DEBUG uvicorn src.main:app --host 0.0.0.0 --port 8005 --reload
```

### Collection Verification

```bash
# Check Qdrant collection
curl http://localhost:6333/collections/embeddings

# Check collection points
curl http://localhost:6333/collections/embeddings/points/count
```

## Production Deployment

### Environment Variables

```env
# Production settings
LOG_LEVEL=INFO
WORKERS=4
ENABLE_CORS=true
CORS_ORIGINS=["https://yourdomain.com"]

# Security
REQUIRE_TENANT_HEADER=true

# Monitoring
ENABLE_METRICS=true
ENABLE_TRACING=true
```

### Docker Production

```bash
# Build production image
docker build -t query-service:prod .

# Run with production settings
docker run -d \
  --name query-service \
  -p 8005:8005 \
  -e DATABASE_URL=postgresql://... \
  -e QDRANT_HOST=qdrant-host \
  query-service:prod
```

### Load Balancing

For high availability, run multiple instances behind a load balancer:

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  query-service-1:
    image: query-service:prod
    environment:
      - WORKER_ID=1
  
  query-service-2:
    image: query-service:prod
    environment:
      - WORKER_ID=2
  
  nginx:
    image: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

## Support

For issues and questions:

1. Check the logs for error details
2. Verify all dependencies are running
3. Test with simple queries first
4. Check the health endpoint status

## Next Steps

After successful setup:

1. **Test with real data** - Upload documents via ingestion service
2. **Configure LLM** - Set up your preferred LLM provider
3. **Tune parameters** - Adjust top_k, thresholds, and filters
4. **Monitor performance** - Track query times and confidence scores
5. **Scale as needed** - Add more workers or instances
