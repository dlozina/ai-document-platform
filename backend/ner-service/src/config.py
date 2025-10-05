"""
Configuration module for NER Service
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # Application Settings
    app_name: str = "NER Service"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # Server Settings
    host: str = "0.0.0.0"  # nosec B104 - Required for containerized microservice
    port: int = 8001
    workers: int = 4

    # NER Settings
    spacy_model: str = "en_core_web_sm"  # Default to small model
    fallback_model: str = "en_core_web_lg"  # Fallback to large model
    max_text_length: int = 1000000  # 1M characters
    batch_size: int = 100

    # Entity Types to Extract
    default_entity_types: list[str] = [
        "PERSON",
        "ORG",
        "GPE",
        "MONEY",
        "PERCENT",
        "DATE",
        "TIME",
        "LOC",
    ]

    # Performance Settings
    enable_caching: bool = False
    cache_ttl_seconds: int = 3600
    redis_url: str | None = None  # redis://localhost:6379/0

    # Security Settings
    enable_cors: bool = True
    cors_origins: list[str] = ["*"]
    require_tenant_header: bool = False

    # Monitoring Settings
    enable_metrics: bool = True
    enable_tracing: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def max_text_length_bytes(self) -> int:
        """Get max text length in bytes."""
        return self.max_text_length * 4  # Rough estimate for UTF-8


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
    print(f"Spacy Model: {settings.spacy_model}")
    print(f"Max Text Length: {settings.max_text_length}")
    print(f"Default Entity Types: {settings.default_entity_types}")
