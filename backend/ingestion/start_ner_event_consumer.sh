#!/bin/bash

# Start NER Event Consumer
# This script starts the Celery worker for processing NER completion events

echo "Starting NER Event Consumer..."

# Change to the ingestion service directory
cd /Users/dlozina/workspace/assignment/abysalto/backend/ingestion

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start Celery worker for NER event processing
echo "Starting Celery worker for NER events..."
celery -A src.celery_app worker \
    --loglevel=info \
    --queues=ner_event_queue \
    --concurrency=2 \
    --hostname=ner-event-consumer@%h \
    --without-gossip \
    --without-mingle \
    --without-heartbeat

echo "NER Event Consumer started!"
