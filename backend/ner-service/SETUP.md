# NER Service Setup Guide

This guide will help you set up and run the NER Service locally or in production.

## Prerequisites

- Python 3.12 or higher
- Docker and Docker Compose (optional, for containerized deployment)
- Git

## Local Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd backend/ner-service
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

#### Option A: Using Platform-Specific Requirements (Recommended)

**For macOS with Apple Silicon:**
```bash
# Install Python dependencies with Apple Silicon support
pip install -r requirements-macos.txt

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

**For other platforms (Linux, Windows, Intel Mac):**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

#### Option B: Manual Installation (If Option A fails)

**For macOS with Apple Silicon:**
```bash
# Upgrade pip and build tools
pip install -U pip setuptools wheel

# Install spaCy with Apple Silicon support
pip install -U 'spacy[apple]'

# Install other dependencies
pip install fastapi uvicorn[standard] python-multipart
pip install pydantic pydantic-settings httpx
pip install pytest pytest-asyncio python-dotenv

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

**For other platforms:**
```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```


### 4. Configure Environment

```bash
# Copy environment template
cp env.example .env

# Edit configuration as needed
nano .env
```

### 5. Run the Service

```bash
# Development mode with auto-reload
python -m uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload

# Or run directly
python src/main.py
```

### 6. Test the Service

```bash
# Health check
curl http://localhost:8001/health

# Extract entities
curl -X POST "http://localhost:8001/extract" \
     -H "Content-Type: application/json" \
     -d '{"text": "Apple Inc. was founded by Steve Jobs in Cupertino, California."}'
```

## Docker Setup

### 1. Build and Run with Docker

```bash
# Build the image
docker build -t ner-service .

# Run the container
docker run -p 8001:8001 ner-service
```

### 2. Using Docker Compose

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f ner-service

# Stop the service
docker-compose down
```

## Production Deployment

### 1. Environment Configuration

Create a production `.env` file:

```bash
# Production settings
LOG_LEVEL=WARNING
SPACY_MODEL=en_core_web_lg
MAX_TEXT_LENGTH=1000000
ENABLE_CACHING=true
REDIS_URL=redis://redis:6379/0
```

### 2. Docker Compose for Production

```yaml
version: '3.8'
services:
  ner-service:
    build: .
    ports:
      - "8001:8001"
    environment:
      - LOG_LEVEL=WARNING
      - SPACY_MODEL=en_core_web_lg
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### 3. Run Production Setup

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Testing

### 1. Run Unit Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 2. API Testing

```bash
# Test health endpoint
curl http://localhost:8001/health

# Test entity extraction
curl -X POST "http://localhost:8001/extract" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
       "entity_types": ["PERSON", "ORG", "GPE"],
       "include_confidence": true
     }'

# Test batch processing
curl -X POST "http://localhost:8001/extract-batch" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "John Smith works at Microsoft.",
         "The meeting is on March 15, 2024."
       ],
       "entity_types": ["PERSON", "ORG", "DATE"]
     }'
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | NER Service | Application name |
| `LOG_LEVEL` | INFO | Logging level |
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8001 | Server port |
| `SPACY_MODEL` | en_core_web_sm | Primary spaCy model |
| `FALLBACK_MODEL` | en_core_web_lg | Fallback spaCy model |
| `MAX_TEXT_LENGTH` | 1000000 | Maximum text length |
| `BATCH_SIZE` | 100 | Batch processing size |
| `ENABLE_CACHING` | false | Enable Redis caching |
| `REDIS_URL` | | Redis connection URL |

### Model Selection

- **en_core_web_sm**: Small, fast model (~50MB)
  - Good for: Speed, low memory usage
  - Trade-off: Lower accuracy
  
- **en_core_web_lg**: Large, accurate model (~500MB)
  - Good for: High accuracy, comprehensive entity types
  - Trade-off: Higher memory usage, slower processing

## Performance Tuning

### 1. Model Selection

```bash
# For speed (development)
SPACY_MODEL=en_core_web_sm

# For accuracy (production)
SPACY_MODEL=en_core_web_lg
```

### 2. Batch Processing

```bash
# Process multiple texts efficiently
curl -X POST "http://localhost:8001/extract-batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["text1", "text2", "text3"]}'
```

### 3. Entity Type Filtering

```bash
# Only extract specific entity types
curl -X POST "http://localhost:8001/extract" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Your text here",
       "entity_types": ["PERSON", "ORG"]
     }'
```

### 4. Caching (Optional)

```bash
# Enable Redis caching
ENABLE_CACHING=true
REDIS_URL=redis://localhost:6379/0
```

## Monitoring and Logging

### 1. Health Checks

```bash
# Check service health
curl http://localhost:8001/health

# Response example:
{
  "status": "healthy",
  "service": "ner-service",
  "version": "1.0.0",
  "spacy_models_available": {
    "en_core_web_sm": true,
    "en_core_web_lg": true
  }
}
```

### 2. Logging

```bash
# View service logs
docker-compose logs -f ner-service

# Or for local development
tail -f logs/ner-service.log
```

### 3. Metrics

The service logs processing statistics:

```json
{
  "text_length": 1000,
  "processing_time_ms": "45.2",
  "model_used": "en_core_web_sm",
  "entity_count": 5,
  "entity_types": ["PERSON", "ORG", "GPE"]
}
```

## Troubleshooting

### Common Issues

1. **spaCy models not found**
   ```bash
   python -m spacy download en_core_web_sm
   python -m spacy download en_core_web_lg
   ```

2. **Memory issues**
   - Use smaller model (`en_core_web_sm`)
   - Reduce `MAX_TEXT_LENGTH`
   - Increase Docker memory limits

3. **Slow processing**
   - Use batch processing for multiple texts
   - Filter entity types
   - Use smaller model

4. **Port conflicts**
   ```bash
   # Change port in .env
   PORT=8002
   ```

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG

# Run with debug output
python -m uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload --log-level debug
```

## API Documentation

Once the service is running, visit:

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **OpenAPI Schema**: http://localhost:8001/openapi.json

## Next Steps

1. **Integration**: Integrate with your application
2. **Scaling**: Set up load balancing for multiple instances
3. **Monitoring**: Add monitoring and alerting
4. **Customization**: Customize entity types and models
5. **Caching**: Enable Redis caching for better performance
