# OCR Service

Production-ready OCR (Optical Character Recognition) service for extracting text from PDFs and images.

## Structure

ocr-service/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── ocr_processor.py     # Core OCR logic
│   ├── models.py            # Pydantic models
│   ├── config.py            # Configuration
│   └── utils.py             # Helper functions
├── tests/
│   ├── __init__.py
│   ├── test_ocr_processor.py
│   └── test_api.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md

## Docker Compose

# Start the service (detached mode)
docker compose up -d

# Start with logs visible
docker compose up

# Stop the service
docker compose down

# Rebuild and start (after code changes)
docker compose up --build -d

# View logs
docker compose logs -f

# Check service status
docker compose ps