# Ingestion Service

Multi-tenant document ingestion service with PostgreSQL metadata storage and MinIO object storage integration.

## Structure

ingestion-service/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── models.py            # Pydantic models
│   ├── database.py          # Database models and operations
│   ├── storage.py           # MinIO object storage integration
│   ├── processing.py        # Processing pipeline integration
│   └── utils.py             # Helper functions
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_components.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
