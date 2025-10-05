"""
Configuration module for OCR Service
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # Application Settings
    app_name: str = "OCR Service"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # Server Settings
    host: str = "0.0.0.0"  # nosec B104 - Required for containerized microservice
    port: int = 8000
    workers: int = 4

    # OCR Settings
    tesseract_cmd: str | None = None  # Auto-detect if None
    tesseract_lang: str = "eng+hrv"  # Support English and Croatian
    ocr_dpi: int = 300
    enable_language_detection: bool = True  # Enable automatic language detection

    # File Processing Settings
    max_file_size_mb: int = 20
    supported_image_formats: list = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]
    supported_pdf_format: str = ".pdf"

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
        return set(self.supported_image_formats + [self.supported_pdf_format])


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
    print(f"OCR Language: {settings.tesseract_lang}")
    print(f"DPI: {settings.ocr_dpi}")
