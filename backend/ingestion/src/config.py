"""
Configuration module for Ingestion Service
"""

from typing import Optional, List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    
    # Application Settings
    app_name: str = "Ingestion Service"
    app_version: str = "1.0.0"
    log_level: str = "INFO"
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8003
    workers: int = 4
    
    # Database Settings (PostgreSQL)
    database_url: str = "postgresql://postgres:password@postgres:5432/ingestion_db"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # MinIO Object Storage Settings
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket_prefix: str = "ingestion"
    
    # File Processing Settings
    max_file_size_mb: int = 100
    max_files_per_request: int = 10
    supported_file_types: List[str] = [
        'application/pdf',
        'image/png',
        'image/jpeg',
        'image/jpg',
        'image/tiff',
        'image/bmp',
        'image/gif',
        'text/plain',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ]
    
    # Multi-tenant Settings
    require_tenant_header: bool = True
    tenant_header_name: str = "X-Tenant-ID"
    default_tenant_id: Optional[str] = None
    
    # Processing Pipeline Settings
    enable_ocr_processing: bool = True
    enable_ner_processing: bool = True
    enable_embedding_processing: bool = True
    ocr_service_url: str = "http://ocr-service:8000"
    ner_service_url: str = "http://ner-service:8001"
    embedding_service_url: str = "http://embedding-service:8002"
    
    # Security Settings
    enable_cors: bool = True
    cors_origins: List[str] = ["*"]
    enable_file_validation: bool = True
    enable_virus_scanning: bool = False
    virus_scan_service_url: Optional[str] = None
    
    # Performance Settings
    enable_caching: bool = False
    cache_ttl_seconds: int = 3600
    redis_url: Optional[str] = "redis://redis:6379/0"
    
    # Redis Event Settings
    redis_host: str = "redis"  # Docker service name
    redis_port: int = 6379
    redis_db: int = 0
    
    # Celery Settings
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/0"
    celery_worker_concurrency: int = 3
    celery_task_time_limit: int = 300
    celery_task_soft_time_limit: int = 240
    celery_worker_prefetch_multiplier: int = 1
    celery_task_acks_late: bool = True
    celery_worker_max_tasks_per_child: int = 1000
    celery_worker_max_memory_per_child: int = 200000
    
    # Monitoring Settings
    enable_metrics: bool = True
    enable_tracing: bool = False
    
    # Data Retention Settings
    default_retention_days: int = 365
    enable_soft_delete: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def supported_extensions(self) -> set:
        """Get supported file extensions."""
        extension_map = {
            'application/pdf': '.pdf',
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg',
            'image/tiff': '.tiff',
            'image/bmp': '.bmp',
            'image/gif': '.gif',
            'text/plain': '.txt',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx'
        }
        return set(extension_map.values())


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to create a singleton settings object.
    """
    return Settings()


# Example usage:
if __name__ == "__main__":
    settings = get_settings()
    print(f"App: {settings.app_name} v{settings.app_version}")
    print(f"Database: {settings.database_url}")
    print(f"MinIO: {settings.minio_endpoint}")
    print(f"Max file size: {settings.max_file_size_mb}MB")
