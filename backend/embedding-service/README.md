# Embedding Service

A production-ready embedding service for text vectorization and similarity search using sentence-transformers and Qdrant vector database.

## Features

- **Text Embedding Generation**: Convert text to high-dimensional vectors using state-of-the-art sentence-transformers models
- **Vector Storage**: Store embeddings in Qdrant vector database with efficient indexing
- **Similarity Search**: Find similar texts using cosine similarity with configurable thresholds
- **Batch Processing**: Efficient batch embedding generation for multiple texts
- **Multiple Formats**: Support for TXT, JSON, CSV, and Markdown files
- **REST API**: Clean FastAPI-based REST endpoints with comprehensive documentation
- **Health Monitoring**: Built-in health checks and monitoring capabilities
- **Docker Support**: Containerized deployment with Docker and Docker Compose

## Quick Start

### Using Docker Compose (Recommended)

1. Clone the repository and navigate to the embedding service directory:
```bash
cd backend/embedding-service
```

2. Start the service with Qdrant:
```bash
docker-compose up -d
```

3. The service will be available at `http://localhost:8001`

### Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Qdrant (using Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant:v1.7.0
```

3. Set environment variables (optional):
```bash
cp env.example .env
# Edit .env with your configuration
```

4. Run the service:
```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8001
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Generate Embedding
```bash
POST /embed
Content-Type: application/json

{
  "text": "Your text here",
  "document_id": "optional_doc_id",
  "metadata": {"source": "test"}
}
```

### Batch Embeddings
```bash
POST /embed-batch
Content-Type: application/json

{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "document_ids": ["doc1", "doc2", "doc3"]
}
```

### Similarity Search
```bash
POST /search
Content-Type: application/json

{
  "query": "search query",
  "limit": 10,
  "score_threshold": 0.8,
  "filter": {"category": "test"}
}
```

### Upload File
```bash
POST /embed-file
Content-Type: multipart/form-data

file: [your text file]
```

## Configuration

Key environment variables:

- `EMBEDDING_MODEL`: Sentence-transformer model name (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `QDRANT_HOST`: Qdrant server host (default: `localhost`)
- `QDRANT_PORT`: Qdrant server port (default: `6333`)
- `QDRANT_COLLECTION_NAME`: Collection name (default: `embeddings`)
- `MAX_FILE_SIZE_MB`: Maximum file size (default: `10`)

## Supported Models

The service supports any sentence-transformer model from Hugging Face. Popular options:

- `sentence-transformers/all-MiniLM-L6-v2` (384 dims) - Fast, good quality
- `sentence-transformers/all-mpnet-base-v2` (768 dims) - Higher quality
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims) - Multilingual

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=src tests/
```

## License

MIT License - see LICENSE file for details.
