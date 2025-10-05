"""
Utility functions for Embedding Service
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def calculate_file_hash(content: bytes) -> str:
    """
    Calculate SHA-256 hash of file content.

    Args:
        content: File content as bytes

    Returns:
        SHA-256 hash as hexadecimal string
    """
    return hashlib.sha256(content).hexdigest()


def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
    """
    Validate file extension against allowed extensions.

    Args:
        filename: Name of the file
        allowed_extensions: Set of allowed extensions (e.g., {'.txt', '.json'})

    Returns:
        True if extension is allowed, False otherwise
    """
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_extensions


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)

    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1

    return f"{size:.1f} {size_names[i]}"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing or replacing unsafe characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove path separators and other unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    sanitized = filename

    for char in unsafe_chars:
        sanitized = sanitized.replace(char, "_")

    # Limit length
    if len(sanitized) > 255:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        sanitized = name[: 255 - len(ext)] + ext

    return sanitized


def create_error_response(error: str, detail: str | None = None) -> dict[str, Any]:
    """
    Create standardized error response.

    Args:
        error: Error message
        detail: Additional error details

    Returns:
        Error response dictionary
    """
    response = {"error": error}
    if detail:
        response["detail"] = detail
    return response


def log_processing_stats(
    filename: str | None,
    text_length: int,
    processing_time_ms: float,
    embedding_dimension: int,
    model_name: str,
) -> None:
    """
    Log processing statistics for monitoring.

    Args:
        filename: Processed filename (if applicable)
        text_length: Length of processed text
        processing_time_ms: Processing time in milliseconds
        embedding_dimension: Dimension of generated embedding
        model_name: Name of the embedding model used
    """
    stats = {
        "filename": filename,
        "text_length": text_length,
        "processing_time_ms": f"{processing_time_ms:.2f}",
        "embedding_dimension": embedding_dimension,
        "model_name": model_name,
    }

    logger.info(f"Processing stats: {stats}")


def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to maximum length with ellipsis.

    Args:
        text: Input text
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def validate_text_length(text: str, max_length: int = 10000) -> bool:
    """
    Validate text length against maximum allowed length.

    Args:
        text: Input text
        max_length: Maximum allowed length

    Returns:
        True if text is within limits, False otherwise
    """
    return len(text) <= max_length


def normalize_text(text: str) -> str:
    """
    Normalize text for embedding generation.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove control characters
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")

    return text.strip()


class FileValidator:
    """Utility class for file validation."""

    def __init__(self, max_size_bytes: int, allowed_extensions: set):
        self.max_size_bytes = max_size_bytes
        self.allowed_extensions = allowed_extensions

    def validate(self, filename: str, content: bytes) -> dict[str, Any]:
        """
        Validate file against constraints.

        Args:
            filename: Name of the file
            content: File content as bytes

        Returns:
            Validation result with 'valid' boolean and 'error' message if invalid
        """
        # Check file size
        if len(content) > self.max_size_bytes:
            return {
                "valid": False,
                "error": f"File too large. Maximum size: {format_file_size(self.max_size_bytes)}",
            }

        # Check if file is empty
        if len(content) == 0:
            return {"valid": False, "error": "Empty file uploaded"}

        # Check file extension
        if not validate_file_extension(filename, self.allowed_extensions):
            return {
                "valid": False,
                "error": f"Unsupported file type. Allowed: {', '.join(self.allowed_extensions)}",
            }

        return {"valid": True, "error": None}


class TextValidator:
    """Utility class for text validation."""

    def __init__(self, max_length: int = 10000, min_length: int = 1):
        self.max_length = max_length
        self.min_length = min_length

    def validate(self, text: str) -> dict[str, Any]:
        """
        Validate text against constraints.

        Args:
            text: Input text

        Returns:
            Validation result with 'valid' boolean and 'error' message if invalid
        """
        # Check minimum length
        if len(text.strip()) < self.min_length:
            return {
                "valid": False,
                "error": f"Text too short. Minimum length: {self.min_length} characters",
            }

        # Check maximum length
        if len(text) > self.max_length:
            return {
                "valid": False,
                "error": f"Text too long. Maximum length: {self.max_length} characters",
            }

        return {"valid": True, "error": None}
