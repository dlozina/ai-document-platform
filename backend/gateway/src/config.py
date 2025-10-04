"""
Configuration module for API Gateway Service
"""

from typing import Optional, List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    
    # Application Settings
    app_name: str = "API Gateway Service"
    app_version: str = "1.0.0"
    log_level: str = "INFO"
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8005
    workers: int = 4
    
    # Database Settings
    database_url: str = "postgresql://postgres:password@postgres:5432/ingestion_db"
    
    # JWT Settings
    jwt_secret_key: str = "your-super-secret-jwt-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    
    # Rate Limiting Settings
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_requests_per_hour: int = 1000
    rate_limit_requests_per_day: int = 10000
    rate_limit_burst_size: int = 10
    
    # Redis Settings
    redis_url: str = "redis://localhost:6379/1"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1
    
    # Backend Services
    ingestion_service_url: str = "http://localhost:8003"
    ocr_service_url: str = "http://localhost:8000"
    ner_service_url: str = "http://localhost:8001"
    embedding_service_url: str = "http://localhost:8002"
    query_service_url: str = "http://localhost:8005"
    
    # Security Settings
    enable_cors: bool = True
    cors_origins: List[str] = ["*"]
    allowed_hosts: List[str] = ["localhost", "127.0.0.1", "0.0.0.0"]
    
    # Monitoring Settings
    enable_metrics: bool = True
    enable_tracing: bool = False
    
    # User Management
    default_user_role: str = "user"
    admin_user_role: str = "admin"
    enable_user_registration: bool = True
    require_email_verification: bool = False
    
    # Password Settings
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special_chars: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


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
    print(f"JWT Secret: {settings.jwt_secret_key[:10]}...")
    print(f"Rate Limiting: {settings.rate_limit_enabled}")
    print(f"Redis: {settings.redis_url}")
