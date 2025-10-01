# Backend Services Docker Compose

This Docker Compose file orchestrates all backend services in a single deployment.

## Services Included

- **OCR Service** (Port 8000) - Document text extraction
- **NER Service** (Port 8001) - Named Entity Recognition
- **Embedding Service** (Port 8002) - Text embeddings with Qdrant vector database
- **Ingestion Service** (Port 8003) - Main API service with Celery workers
- **PostgreSQL** (Port 5432) - Primary database
- **Redis** (Port 6379) - Cache and message broker
- **MinIO** (Ports 9000, 9001) - Object storage
- **Qdrant** (Ports 6333, 6334) - Vector database
- **Flower** (Port 5555) - Celery monitoring dashboard

## Quick Start

1. **Start all services:**
   ```bash
   cd backend
   docker compose up -d
   ```

2. **View logs:**
   ```bash
   docker compose logs -f
   ```

3. **Stop all services:**
   ```bash
   docker compose down
   ```

4. **Stop and remove volumes:**
   ```bash
   docker compose down -v
   ```

## Service Management

### Start specific services:
```bash
# Start only core services (no workers)
docker compose up -d postgres redis minio qdrant

# Start processing services
docker compose up -d ocr-service ner-service embedding-service

# Start ingestion service and workers
docker compose up -d ingestion-service celery-worker flower
```

### Scale workers:
```bash
# Scale Celery workers
docker compose up -d --scale celery-worker=4
```

### Health Checks:
All services include health checks. Monitor status with:
```bash
docker compose ps
```

## Service URLs

- OCR Service: http://localhost:8000
- NER Service: http://localhost:8001
- Embedding Service: http://localhost:8002
- Ingestion Service: http://localhost:8003
- Flower Dashboard: http://localhost:5555 (admin/admin)
- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)

## Development

### Rebuild services after code changes:
```bash
docker compose build
docker compose up -d
```

### View service logs:
```bash
docker compose logs -f [service-name]
```

### Access service containers:
```bash
docker compose exec [service-name] /bin/bash
```

## Data Persistence

The following data is persisted in Docker volumes:
- `postgres_data` - PostgreSQL database
- `minio_data` - MinIO object storage
- `redis_data` - Redis cache
- `qdrant_storage` - Vector embeddings

## Environment Variables

Key environment variables are configured in the compose file. For production deployments, consider using `.env` files or Docker secrets.

## Troubleshooting

1. **Port conflicts**: Ensure ports 8000-8003, 5432, 6379, 9000-9001, 5555, 6333-6334 are available
2. **Memory issues**: Some services (especially ML models) require significant memory
3. **Service dependencies**: Services start in dependency order, but allow extra time for ML model loading
4. **Health checks**: Services may take 40+ seconds to become healthy due to model initialization
