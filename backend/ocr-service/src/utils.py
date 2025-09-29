"""
Utility functions for OCR Service
"""

import hashlib
import logging
from typing import Optional, Dict, Any
from pathlib import Path

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
        allowed_extensions: Set of allowed extensions (e.g., {'.pdf', '.png'})
        
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
        sanitized = sanitized.replace(char, '_')
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        sanitized = name[:255-len(ext)] + ext
    
    return sanitized


def create_error_response(error: str, detail: Optional[str] = None) -> Dict[str, Any]:
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
    filename: str,
    file_size: int,
    processing_time_ms: float,
    method: str,
    text_length: int,
    confidence: Optional[float] = None
) -> None:
    """
    Log processing statistics for monitoring.
    
    Args:
        filename: Processed filename
        file_size: File size in bytes
        processing_time_ms: Processing time in milliseconds
        method: Processing method used
        text_length: Length of extracted text
        confidence: OCR confidence score (if applicable)
    """
    stats = {
        "filename": filename,
        "file_size": format_file_size(file_size),
        "processing_time_ms": f"{processing_time_ms:.2f}",
        "method": method,
        "text_length": text_length,
        "confidence": confidence
    }
    
    logger.info(f"Processing stats: {stats}")


class FileValidator:
    """Utility class for file validation."""
    
    def __init__(self, max_size_bytes: int, allowed_extensions: set):
        self.max_size_bytes = max_size_bytes
        self.allowed_extensions = allowed_extensions
    
    def validate(self, filename: str, content: bytes) -> Dict[str, Any]:
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
                "error": f"File too large. Maximum size: {format_file_size(self.max_size_bytes)}"
            }
        
        # Check if file is empty
        if len(content) == 0:
            return {
                "valid": False,
                "error": "Empty file uploaded"
            }
        
        # Check file extension
        if not validate_file_extension(filename, self.allowed_extensions):
            return {
                "valid": False,
                "error": f"Unsupported file type. Allowed: {', '.join(self.allowed_extensions)}"
            }
        
        return {"valid": True, "error": None}
