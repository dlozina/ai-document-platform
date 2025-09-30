#!/bin/bash
# Celery Worker and Flower Startup Script

set -e

# Default values
WORKER_CONCURRENCY=${CELERY_WORKER_CONCURRENCY:-3}
WORKER_LOGLEVEL=${CELERY_WORKER_LOGLEVEL:-info}
FLOWER_PORT=${FLOWER_PORT:-5555}
FLOWER_BASIC_AUTH=${FLOWER_BASIC_AUTH:-admin:admin}

echo "Starting Celery Worker and Flower for Ingestion Service"
echo "Worker Concurrency: $WORKER_CONCURRENCY"
echo "Worker Log Level: $WORKER_LOGLEVEL"
echo "Flower Port: $FLOWER_PORT"

# Start Celery worker in background
echo "Starting Celery worker..."
celery -A src.celery_app worker \
    --loglevel=$WORKER_LOGLEVEL \
    --concurrency=$WORKER_CONCURRENCY \
    --prefetch-multiplier=1 \
    --max-tasks-per-child=1000 \
    --max-memory-per-child=200000 \
    --time-limit=300 \
    --soft-time-limit=240 \
    --queues=ocr_queue,ner_queue,embedding_queue,completion_queue,dead_letter_queue \
    --hostname=worker@%h \
    --detach

# Start Flower monitoring
echo "Starting Flower monitoring..."
celery -A src.celery_app flower \
    --port=$FLOWER_PORT \
    --basic_auth=$FLOWER_BASIC_AUTH \
    --broker_api=redis://redis:6379/0 \
    --persistent=true \
    --db=/tmp/flower.db \
    --max_tasks=10000 \
    --auto_refresh=true \
    --auto_refresh_interval=5 \
    --format_task=true \
    --show_task_events=true \
    --show_task_sent_events=true \
    --show_task_received_events=true \
    --show_task_started_events=true \
    --show_task_succeeded_events=true \
    --show_task_failed_events=true \
    --show_task_retried_events=true \
    --show_task_revoked_events=true \
    --show_task_cancelled_events=true

echo "Celery Worker and Flower started successfully"
echo "Flower monitoring available at: http://localhost:$FLOWER_PORT"
echo "Username: admin, Password: admin"
