"""
Configuration module for Embedding Service
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # Application Settings
    app_name: str = "Embedding Service"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # Server Settings
    host: str = "0.0.0.0"  # nosec B104 - Required for containerized microservice
    port: int = 8002
    workers: int = 4

    # Embedding Model Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    max_text_length: int = 512
    batch_size: int = 32

    # Qdrant settings removed - vector storage moved to ingestion service

    # File Processing Settings
    max_file_size_mb: int = 10
    supported_text_formats: list = [".txt", ".md", ".json", ".csv"]
    supported_document_formats: list = [".pdf", ".docx", ".doc"]

    # Performance Settings
    enable_caching: bool = False
    cache_ttl_seconds: int = 3600
    redis_url: str | None = None  # redis://localhost:6379/0

    # Security Settings
    enable_cors: bool = True
    cors_origins: list = ["*"]
    require_tenant_header: bool = False

    # Monitoring Settings
    enable_metrics: bool = True
    enable_tracing: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def all_supported_formats(self) -> set:
        """Get all supported file formats."""
        return set(self.supported_text_formats + self.supported_document_formats)


@lru_cache
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
    print(f"Max file size: {settings.max_file_size_mb}MB")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Qdrant URL: {settings.qdrant_url}")
