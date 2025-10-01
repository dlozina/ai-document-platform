# Query Service

Production-ready query service for semantic search and question-answering that leverages OCR, NER, and embeddings to provide intelligent answers to user questions.

## Features

### Core Functionality
- **Semantic Search**: Find relevant documents using vector similarity and embeddings
- **Question Answering**: Extractive QA and RAG (Retrieval-Augmented Generation) modes
- **Advanced Filtering**: Filter by metadata, entities, dates, file properties, and more
- **Entity Recognition**: Integration with NER for enhanced search capabilities
- **Result Reranking**: Improve relevance using cross-encoder models
- **Multiple LLM Support**: OpenAI, Anthropic, Mistral, and local models

### Query Modes

1. **Semantic Search** (`semantic_search`)
   - Find relevant documents using vector similarity
   - Supports advanced filtering and scoring
   - Returns ranked list of documents

2. **Extractive QA** (`extractive_qa`)
   - Extract exact answers from document text
   - Best for factual questions with clear answers
   - Example: "Where is Dino based?" → "Split, Croatia"

3. **RAG** (`rag`)
   - Generate comprehensive answers using LLM + retrieved context
   - Better for complex questions requiring reasoning
   - Example: "What is Dino's technical expertise?" → Generated summary

### Filtering Capabilities

- **Metadata filters**: `tenant_id`, `content_type`, `file_type`, `processing_status`
- **Entity filters**: Find documents mentioning specific people, organizations, locations
- **Date ranges**: Filter by upload timestamp
- **File properties**: Size ranges, processing status, created_by user
- **Custom tags**: Document tags and descriptions

### Use Cases

- "What programming languages does Dino know?" → Extract Python, JavaScript, Rust
- "Where has Dino worked?" → Find Macrometa, identify as US startup
- "Find documents about system design" → Semantic search on that topic
- "Show me all documents mentioning Croatian locations" → Entity-filtered search

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL database
- Qdrant vector database
- spaCy English model: `python -m spacy download en_core_web_sm`
- **Mistral API Key** (get one at [Mistral AI Console](https://console.mistral.ai/))

### Installation

1. **Clone and setup**:
   ```bash
   cd backend/query-service
   pip install -e .
   ```

2. **Configure environment**:
   ```bash
   cp env.example .env
   # Edit .env with your settings
   ```

3. **Start the service**:
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8004
   ```

### Quick Docker Setup

```bash
# Navigate to backend directory
cd backend

# Set your Mistral API key
export MISTRAL_API_KEY=your_actual_mistral_api_key_here

# Start all services
docker-compose up --build
```

### Docker Setup

#### Method 1: Run with Main Docker Compose (Recommended)

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

#### Method 2: Run Query Service Standalone

```bash
# Navigate to query service directory
cd backend/query-service

# Set your Mistral API key
export MISTRAL_API_KEY=your_actual_mistral_api_key_here

# Run standalone (connects to existing services)
docker-compose up --build
```

#### Method 3: Using .env File

Create a `.env` file in the `backend` directory:

```bash
cd backend
echo "MISTRAL_API_KEY=your_actual_mistral_api_key_here" > .env
docker-compose up --build
```

**Important**: Never commit your API key to version control! The `.env` file is already in `.gitignore`.

## API Endpoints

### Main Query Endpoint

```http
POST /query
Content-Type: application/json
X-Tenant-ID: your-tenant-id

{
  "query": "What programming languages does Dino know?",
  "mode": "rag",
  "top_k": 5,
  "score_threshold": 0.7,
  "filter": {
    "tenant_id": "dino",
    "entity_labels": ["PERSON", "ORG"],
    "date_range": {
      "start": "2024-01-01T00:00:00Z",
      "end": "2024-12-31T23:59:59Z"
    }
  }
}
```

**Response**:
```json
{
  "answer": "Dino Lozina knows Python, JavaScript, and Rust programming languages...",
  "confidence_score": 0.87,
  "sources": [
    {
      "document_id": "716f97c5-3a43-46b3-8a71-1b0ceafd0b33",
      "filename": "video-introduction-5.pdf",
      "quoted_text": "I'm Dino Lozina, and I'm based in Split, Croatia",
      "relevance_score": 0.92,
      "page_or_position": "Opening section"
    }
  ],
  "detected_entities": [
    {"text": "Dino Lozina", "label": "PERSON", "confidence": 0.75},
    {"text": "Split", "label": "GPE", "confidence": 0.85},
    {"text": "Croatia", "label": "GPE", "confidence": 0.85}
  ],
  "retrieved_documents_count": 3,
  "processing_time_ms": 245.6,
  "query_mode": "rag",
  "llm_provider": "openai",
  "llm_model": "gpt-3.5-turbo"
}
```

### Semantic Search

```http
POST /search
Content-Type: application/json

{
  "query": "Python programming",
  "top_k": 10,
  "score_threshold": 0.7,
  "filter": {
    "file_type": "pdf",
    "processing_status": "completed"
  }
}
```

### Question Answering

```http
POST /qa
Content-Type: application/json

{
  "question": "What is Dino's leadership style?",
  "mode": "extractive_qa",
  "top_k": 5,
  "max_context_length": 2000
}
```

### Health Check

```http
GET /health
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | Sentence transformer model | `sentence-transformers/all-MiniLM-L6-v2` |
| `QDRANT_HOST` | Qdrant host | `localhost` |
| `QDRANT_PORT` | Qdrant port | `6333` |
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `LLM_PROVIDER` | LLM provider (openai/anthropic/mistral) | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required for Anthropic |
| `ENABLE_RERANKING` | Enable result reranking | `true` |

### LLM Configuration

#### OpenAI
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-3.5-turbo
```

#### Anthropic
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_api_key
ANTHROPIC_MODEL=claude-3-sonnet-20240229
```

#### Mistral
```env
LLM_PROVIDER=mistral
MISTRAL_API_KEY=your_api_key
MISTRAL_MODEL=mistral-small
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Architecture

### Components

1. **QueryProcessor**: Core processing logic
   - Semantic search with Qdrant
   - Entity extraction with spaCy
   - Answer generation (extractive + RAG)
   - Result reranking

2. **FastAPI Application**: REST API endpoints
   - `/query` - Main query endpoint
   - `/search` - Semantic search
   - `/qa` - Question answering
   - `/health` - Health check

3. **Models**: Pydantic models for request/response validation

4. **Utils**: Utility functions for text processing and validation

### Data Flow

1. **Query Input** → Validation → Embedding Generation
2. **Vector Search** → Qdrant → Filtered Results
3. **Entity Filtering** → NER-based filtering
4. **Reranking** → Cross-encoder model (optional)
5. **Answer Generation** → Extractive QA or RAG
6. **Response** → Formatted results with sources

## Performance

### Optimization Tips

- Use appropriate `top_k` values (5-10 for QA, 10-20 for search)
- Enable reranking for better relevance
- Use filters to narrow search scope
- Cache frequent queries with Redis
- Monitor processing times and confidence scores

### Monitoring

- Health check endpoint for service status
- Processing time metrics
- Confidence score tracking
- Query volume and patterns

## Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Qdrant connection failed**
   - Check Qdrant is running
   - Verify host/port settings
   - Check collection exists

3. **Database connection failed**
   - Verify PostgreSQL is running
   - Check connection string
   - Ensure database exists

4. **LLM API errors**
   - Verify API keys are set
   - Check rate limits
   - Verify model names

### Logs

Check logs for detailed error information:
```bash
# Docker logs
docker-compose logs query-service

# Local logs
tail -f logs/query-service.log
```

## License

MIT License - see LICENSE file for details.
