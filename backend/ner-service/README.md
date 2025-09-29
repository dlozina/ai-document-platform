# NER Service

A production-ready Named Entity Recognition (NER) service built with FastAPI and spaCy.

## Features

- **Multiple spaCy Models**: Support for both small (`en_core_web_sm`) and large (`en_core_web_lg`) models
- **Entity Type Filtering**: Extract specific entity types (PERSON, ORG, GPE, etc.)
- **Batch Processing**: Process multiple texts efficiently
- **Confidence Scoring**: Estimated confidence scores for entity detection
- **Entity Visualization**: HTML visualization of detected entities
- **Async Processing**: Background processing for large texts
- **RESTful API**: Clean, documented API endpoints
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Supported Entity Types

- **PERSON**: People, including fictional characters
- **ORG**: Companies, agencies, institutions
- **GPE**: Countries, cities, states (Geopolitical entities)
- **MONEY**: Monetary values
- **PERCENT**: Percentages
- **DATE**: Absolute or relative dates
- **TIME**: Times smaller than a day
- **LOC**: Non-GPE locations

## Quick Start

### Using Docker (Recommended)

1. **Clone and navigate to the service:**
   ```bash
   cd backend/ner-service
   ```

2. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Test the service:**
   ```bash
   curl -X POST "http://localhost:8001/extract" \
        -H "Content-Type: application/json" \
        -d '{"text": "Apple Inc. was founded by Steve Jobs in Cupertino, California."}'
   ```

### Local Development

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download spaCy models:**
   ```bash
   python -m spacy download en_core_web_sm
   python -m spacy download en_core_web_lg
   ```

3. **Run the service:**
   ```bash
   python -m uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
   ```

## API Endpoints

### Health Check
```http
GET /health
```

### Extract Entities
```http
POST /extract
Content-Type: application/json

{
  "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
  "entity_types": ["PERSON", "ORG", "GPE"],
  "include_confidence": true
}
```

### Batch Processing
```http
POST /extract-batch
Content-Type: application/json

{
  "texts": [
    "John Smith works at Microsoft.",
    "The meeting is on March 15, 2024."
  ],
  "entity_types": ["PERSON", "ORG", "DATE"],
  "include_confidence": true
}
```

### Async Processing
```http
POST /extract-async
Content-Type: application/json

{
  "text": "Very long text...",
  "callback_url": "http://your-service.com/callback"
}
```

### Entity Statistics
```http
POST /stats
Content-Type: application/json

{
  "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California."
}
```

### Entity Visualization
```http
POST /visualize
Content-Type: application/json

{
  "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California."
}
```

### Available Models
```http
GET /models
```

## Configuration

Create a `.env` file based on `env.example`:

```bash
# Application Settings
APP_NAME=NER Service
LOG_LEVEL=INFO

# Server Settings
HOST=0.0.0.0
PORT=8001

# NER Settings
SPACY_MODEL=en_core_web_sm
FALLBACK_MODEL=en_core_web_lg
MAX_TEXT_LENGTH=1000000

# Entity Types (comma-separated)
DEFAULT_ENTITY_TYPES=PERSON,ORG,GPE,MONEY,PERCENT,DATE,TIME,LOC
```

## Response Format

### Entity Extraction Response
```json
{
  "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
  "entities": [
    {
      "text": "Apple Inc.",
      "label": "ORG",
      "start": 0,
      "end": 10,
      "confidence": 0.95
    },
    {
      "text": "Steve Jobs",
      "label": "PERSON",
      "start": 25,
      "end": 35,
      "confidence": 0.92
    },
    {
      "text": "Cupertino, California",
      "label": "GPE",
      "start": 39,
      "end": 60,
      "confidence": 0.88
    }
  ],
  "entity_count": 3,
  "model_used": "en_core_web_sm",
  "processing_time_ms": 45.2,
  "text_length": 61
}
```

## Performance

- **Small Model (`en_core_web_sm`)**: ~50ms for 1000 characters
- **Large Model (`en_core_web_lg`)**: ~200ms for 1000 characters
- **Batch Processing**: More efficient for multiple texts
- **Memory Usage**: ~500MB for small model, ~1GB for large model

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Monitoring

The service includes health checks and metrics:

- **Health Endpoint**: `/health` - Service status and model availability
- **Metrics**: Processing time, entity counts, error rates
- **Logging**: Structured logging with configurable levels

## Deployment

### Docker
```bash
docker build -t ner-service .
docker run -p 8001:8001 ner-service
```

### Docker Compose
```bash
docker-compose up -d
```

### Production Considerations

1. **Model Selection**: Use small model for speed, large model for accuracy
2. **Resource Limits**: Set appropriate memory limits for spaCy models
3. **Load Balancing**: Use multiple instances behind a load balancer
4. **Caching**: Enable Redis caching for repeated requests
5. **Monitoring**: Set up health checks and metrics collection

## Troubleshooting

### Common Issues

1. **spaCy models not found:**
   ```bash
   python -m spacy download en_core_web_sm
   python -m spacy download en_core_web_lg
   ```

2. **Memory issues with large texts:**
   - Reduce `MAX_TEXT_LENGTH` in configuration
   - Use async processing for large texts

3. **Slow processing:**
   - Use small model (`en_core_web_sm`) instead of large model
   - Enable batch processing for multiple texts
   - Filter entity types to reduce processing

### Logs

Check service logs for debugging:
```bash
docker-compose logs ner-service
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
