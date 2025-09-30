"""
Flower Configuration for Ingestion Service

Production-ready Flower monitoring setup for Celery task monitoring.
"""

import os
from celery import Celery
from .config import get_settings

# Get settings
settings = get_settings()

# Create Celery app for Flower
celery_app = Celery(
    "ingestion-service",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "src.tasks.ocr_tasks",
        "src.tasks.ner_tasks", 
        "src.tasks.embedding_tasks",
        "src.tasks.completion_tasks"
    ]
)

# Flower configuration
FLOWER_CONFIG = {
    "port": 5555,
    "broker_api": settings.celery_broker_url,
    "basic_auth": "admin:admin",  # Change in production
    "url_prefix": "flower",
    "enable_events": True,
    "persistent": True,
    "db": "/tmp/flower.db",
    "max_tasks": 10000,
    "auto_refresh": True,
    "auto_refresh_interval": 5,
    "format_task": True,
    "show_task_events": True,
    "show_task_sent_events": True,
    "show_task_received_events": True,
    "show_task_started_events": True,
    "show_task_succeeded_events": True,
    "show_task_failed_events": True,
    "show_task_retried_events": True,
    "show_task_revoked_events": True,
    "show_task_cancelled_events": True,
    "task_routes": {
        "src.tasks.ocr_tasks.*": {"queue": "ocr_queue"},
        "src.tasks.ner_tasks.*": {"queue": "ner_queue"},
        "src.tasks.embedding_tasks.*": {"queue": "embedding_queue"},
        "src.tasks.completion_tasks.*": {"queue": "completion_queue"},
    },
    "worker_routes": {
        "ocr_queue": {"queue": "ocr_queue"},
        "ner_queue": {"queue": "ner_queue"},
        "embedding_queue": {"queue": "embedding_queue"},
        "completion_queue": {"queue": "completion_queue"},
    }
}

# Production vs Development configuration
if os.getenv("ENVIRONMENT", "development") == "production":
    FLOWER_CONFIG.update({
        "basic_auth": os.getenv("FLOWER_BASIC_AUTH", "admin:admin"),
        "persistent": True,
        "db": "/data/flower.db",
        "max_tasks": 50000,
        "auto_refresh": True,
        "auto_refresh_interval": 10,
    })
else:
    FLOWER_CONFIG.update({
        "basic_auth": "admin:admin",
        "persistent": False,
        "max_tasks": 1000,
        "auto_refresh": True,
        "auto_refresh_interval": 5,
    })

# Export configuration
__all__ = ["celery_app", "FLOWER_CONFIG"]
