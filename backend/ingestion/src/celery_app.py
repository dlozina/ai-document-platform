"""
Celery Configuration for Ingestion Service

Production-ready Celery setup with Redis broker, optimized for 60 files/minute processing.
"""

import os
from celery import Celery
from kombu import Queue
from .config import get_settings

# Get settings
settings = get_settings()

# Create Celery app
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

# Celery Configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        "src.tasks.ocr_tasks.*": {"queue": "ocr_queue"},
        "src.tasks.ner_tasks.*": {"queue": "ner_queue"},
        "src.tasks.embedding_tasks.*": {"queue": "embedding_queue"},
        "src.tasks.completion_tasks.*": {"queue": "completion_queue"},
    },
    
    # Queue configuration
    task_default_queue="default",
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("ocr_queue", routing_key="ocr"),
        Queue("ner_queue", routing_key="ner"),
        Queue("embedding_queue", routing_key="embedding"),
        Queue("completion_queue", routing_key="completion"),
        Queue("dead_letter_queue", routing_key="dead_letter"),
    ),
    
    # Task execution
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task time limits (optimized for 10MB files)
    task_time_limit=300,  # 5 minutes max per task
    task_soft_time_limit=240,  # 4 minutes soft limit
    worker_prefetch_multiplier=1,  # Process one task at a time for FIFO
    
    # Retry configuration
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_reject_on_worker_lost=True,
    
    # Result backend
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Error handling
    task_ignore_result=False,
    task_store_eager_result=True,
    
    # Redis connection
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    
    # Worker configuration
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
    worker_max_memory_per_child=200000,  # 200MB memory limit per worker
)

# Retry policies for different task types
RETRY_POLICIES = {
    "ocr": {
        "max_retries": 3,
        "countdown": 60,  # Start with 1 minute
        "max_countdown": 300,  # Max 5 minutes
        "multiplier": 2,  # Exponential backoff
    },
    "ner": {
        "max_retries": 2,
        "countdown": 30,  # Start with 30 seconds
        "max_countdown": 120,  # Max 2 minutes
        "multiplier": 2,
    },
    "embedding": {
        "max_retries": 1,
        "countdown": 15,  # Start with 15 seconds
        "max_countdown": 60,  # Max 1 minute
        "multiplier": 2,
    }
}

# Task priority levels
TASK_PRIORITIES = {
    "ocr": 9,  # Highest priority
    "ner": 5,  # Medium priority
    "embedding": 5,  # Medium priority
    "completion": 1,  # Lowest priority
}

# Dead letter queue configuration
DEAD_LETTER_QUEUE_CONFIG = {
    "max_retries": 0,  # No more retries
    "queue": "dead_letter_queue",
    "routing_key": "dead_letter",
}

# Monitoring configuration (simplified)
MONITORING_CONFIG = {
    "flower_port": 5555,
    "flower_basic_auth": "admin:admin",  # Change in production
    "flower_url_prefix": "flower",
    "enable_events": True,
    "task_send_sent_event": True,
    "worker_send_task_events": True,
}

# Performance tuning for 60 files/minute
PERFORMANCE_CONFIG = {
    "worker_concurrency": 3,  # 3 workers for 60 files/minute
    "task_always_eager": False,  # Use async processing
    "task_eager_propagates": True,
    "worker_prefetch_multiplier": 1,  # FIFO processing
    "task_acks_late": True,  # Acknowledge after completion
    "worker_disable_rate_limits": False,
}

# Apply performance configuration
celery_app.conf.update(PERFORMANCE_CONFIG)

# Health check configuration
HEALTH_CHECK_CONFIG = {
    "broker_heartbeat": 30,
    "broker_connection_retry": True,
    "broker_connection_retry_on_startup": True,
    "broker_connection_max_retries": 10,
}

# Apply health check configuration
celery_app.conf.update(HEALTH_CHECK_CONFIG)

# Logging configuration
LOGGING_CONFIG = {
    "worker_log_format": "[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    "worker_task_log_format": "[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s",
    "worker_log_color": True,
    "worker_task_log_color": True,
}

# Apply logging configuration
celery_app.conf.update(LOGGING_CONFIG)

# Task result configuration
RESULT_CONFIG = {
    "result_backend": settings.redis_url or "redis://localhost:6379/0",
    "result_expires": 3600,  # 1 hour
    "result_persistent": True,
    "result_compression": "gzip",
    "result_cache_max": 10000,
}

# Apply result configuration
celery_app.conf.update(RESULT_CONFIG)

# Security configuration
SECURITY_CONFIG = {
    "worker_hijack_root_logger": False,
    "worker_log_color": True,
    "task_ignore_result": False,
    "task_store_eager_result": True,
}

# Apply security configuration
celery_app.conf.update(SECURITY_CONFIG)

# Development vs Production configuration
if os.getenv("ENVIRONMENT", "development") == "production":
    # Production-specific settings
    celery_app.conf.update({
        "worker_prefetch_multiplier": 1,  # Conservative prefetch
        "task_acks_late": True,
        "worker_disable_rate_limits": False,
        "task_reject_on_worker_lost": True,
        "worker_max_tasks_per_child": 1000,
        "worker_max_memory_per_child": 200000,
    })
else:
    # Development-specific settings
    celery_app.conf.update({
        "task_always_eager": False,  # Use async in dev too
        "worker_prefetch_multiplier": 1,
        "task_acks_late": True,
    })

# Celery Beat configuration (currently not used)
# celery_app.conf.beat_schedule = {}

# Export the app for use in other modules
__all__ = ["celery_app", "RETRY_POLICIES", "TASK_PRIORITIES", "MONITORING_CONFIG"]
