"""
Configuration module for Query Service
"""

from typing import Optional, List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    
    # Application Settings
    app_name: str = "Query Service"
    app_version: str = "1.0.0"
    log_level: str = "INFO"
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8004
    workers: int = 4
    
    # Embedding Model Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    max_text_length: int = 512
    batch_size: int = 32
    
    # Qdrant Settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "embeddings"
    qdrant_vector_size: int = 384
    
    # Database Settings
    database_url: str = "postgresql://user:password@localhost:5432/abysalto"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # LLM Settings
    llm_provider: str = "mistral"  # openai, anthropic, mistral, local
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-sonnet-20240229"
    mistral_api_key: Optional[str] = None
    mistral_model: str = "mistral-large-2411"
    
    # Query Settings
    default_top_k: int = 10
    max_top_k: int = 100
    default_score_threshold: float = 0.7
    max_context_length: int = 4000
    enable_reranking: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Performance Settings
    enable_caching: bool = False
    cache_ttl_seconds: int = 3600
    redis_url: Optional[str] = None  # redis://localhost:6379/0
    
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
    def qdrant_url(self) -> str:
        """Get Qdrant connection URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


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
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Qdrant URL: {settings.qdrant_url}")
    print(f"LLM Provider: {settings.llm_provider}")
