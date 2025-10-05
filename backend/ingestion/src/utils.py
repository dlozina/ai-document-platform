"""
Utility functions for Ingestion Service
"""

import hashlib
import io
import logging
import mimetypes
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import magic
from PIL import Image

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


def detect_file_type(content: bytes, filename: str) -> tuple[str, str]:
    """
    Detect file type using both MIME type detection and file extension.

    Args:
        content: File content as bytes
        filename: Original filename

    Returns:
        Tuple of (mime_type, file_category)
    """
    # Try to detect MIME type from content
    try:
        mime_type = magic.from_buffer(content, mime=True)
    except Exception:
        # Fallback to extension-based detection
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            mime_type = "application/octet-stream"

    # Map MIME types to file categories
    category_map = {
        "application/pdf": "pdf",
        "image/png": "image",
        "image/jpeg": "image",
        "image/jpg": "image",
        "image/tiff": "image",
        "image/bmp": "image",
        "image/gif": "image",
        "text/plain": "text",
        "application/msword": "document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document",
    }

    file_category = category_map.get(mime_type, "other")

    return mime_type, file_category


def validate_file_content(
    content: bytes, filename: str, mime_type: str
) -> dict[str, Any]:
    """
    Validate file content against its declared type.

    Args:
        content: File content as bytes
        filename: Original filename
        mime_type: Detected MIME type

    Returns:
        Validation result dictionary
    """
    errors = []
    warnings = []

    try:
        if mime_type.startswith("image/"):
            # Validate image file
            Image.open(io.BytesIO(content))

        elif mime_type == "application/pdf":
            # Validate PDF file
            doc = fitz.open(stream=content, filetype="pdf")
            if len(doc) == 0:
                errors.append("PDF file contains no pages")
            doc.close()

        elif mime_type == "text/plain":
            # Validate text file
            try:
                content.decode("utf-8")
            except UnicodeDecodeError:
                warnings.append("Text file may not be UTF-8 encoded")

    except Exception as e:
        errors.append(f"File content validation failed: {str(e)}")

    return {"is_valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def generate_storage_path(tenant_id: str, filename: str, file_hash: str) -> str:
    """
    Generate a unique storage path for a file.

    Args:
        tenant_id: Tenant identifier
        filename: Original filename
        file_hash: File content hash

    Returns:
        Storage path string
    """
    # Sanitize filename
    safe_filename = sanitize_filename(filename)

    # Create path: tenant_id/year/month/day/hash_filename
    now = datetime.utcnow()
    path_parts = [
        tenant_id,
        str(now.year),
        f"{now.month:02d}",
        f"{now.day:02d}",
        f"{file_hash[:8]}_{safe_filename}",
    ]

    return "/".join(path_parts)


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


def calculate_retention_date(retention_days: int) -> datetime:
    """
    Calculate retention expiration date.

    Args:
        retention_days: Number of days to retain

    Returns:
        Expiration datetime
    """
    return datetime.utcnow() + timedelta(days=retention_days)


def generate_document_id() -> str:
    """
    Generate a unique document identifier.

    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def generate_batch_id() -> str:
    """
    Generate a unique batch identifier.

    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def create_error_response(
    error: str, detail: str | None = None, error_code: str | None = None
) -> dict[str, Any]:
    """
    Create standardized error response.

    Args:
        error: Error message
        detail: Additional error details
        error_code: Error code

    Returns:
        Error response dictionary
    """
    response = {"error": error, "timestamp": datetime.utcnow().isoformat()}
    if detail:
        response["detail"] = detail
    if error_code:
        response["error_code"] = error_code
    return response


def log_upload_stats(
    filename: str,
    file_size: int,
    tenant_id: str,
    processing_time_ms: float,
    success: bool,
) -> None:
    """
    Log upload statistics for monitoring.

    Args:
        filename: Uploaded filename
        file_size: File size in bytes
        tenant_id: Tenant identifier
        processing_time_ms: Processing time in milliseconds
        success: Whether upload was successful
    """
    stats = {
        "filename": filename,
        "file_size": format_file_size(file_size),
        "tenant_id": tenant_id,
        "processing_time_ms": f"{processing_time_ms:.2f}",
        "success": success,
    }

    logger.info(f"Upload stats: {stats}")


class FileValidator:
    """Utility class for file validation."""

    def __init__(self, max_size_bytes: int, allowed_mime_types: list[str]):
        self.max_size_bytes = max_size_bytes
        self.allowed_mime_types = allowed_mime_types

    def validate(self, filename: str, content: bytes) -> dict[str, Any]:
        """
        Validate file against constraints.

        Args:
            filename: Name of the file
            content: File content as bytes

        Returns:
            Validation result with 'is_valid' boolean and 'errors' list
        """
        errors = []
        warnings = []

        # Check file size
        if len(content) > self.max_size_bytes:
            errors.append(
                f"File too large. Maximum size: {format_file_size(self.max_size_bytes)}"
            )

        # Check if file is empty
        if len(content) == 0:
            errors.append("Empty file uploaded")
            return {"is_valid": False, "errors": errors, "warnings": warnings}

        # Detect file type
        mime_type, file_category = detect_file_type(content, filename)

        # Check MIME type
        if mime_type not in self.allowed_mime_types:
            errors.append(
                f"Unsupported file type: {mime_type}. "
                f"Allowed types: {', '.join(self.allowed_mime_types)}"
            )

        # Validate content
        content_validation = validate_file_content(content, filename, mime_type)
        errors.extend(content_validation["errors"])
        warnings.extend(content_validation["warnings"])

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "file_type": mime_type,
            "file_size_bytes": len(content),
        }


class TenantQuotaChecker:
    """Utility class for tenant quota validation."""

    def __init__(self, db_session):
        self.db_session = db_session

    def check_quota(self, tenant_id: str, file_size_bytes: int) -> dict[str, Any]:
        """
        Check if tenant has quota for file upload.

        Args:
            tenant_id: Tenant identifier
            file_size_bytes: File size in bytes

        Returns:
            Quota check result
        """
        # TODO: Implement actual quota checking against database
        # This is a placeholder implementation

        return {
            "has_quota": True,
            "current_usage_bytes": 0,
            "max_storage_bytes": 1024 * 1024 * 1024,  # 1GB default
            "remaining_bytes": 1024 * 1024 * 1024,
        }


def extract_file_metadata(content: bytes, filename: str) -> dict[str, Any]:
    """
    Extract metadata from file content.

    Args:
        content: File content as bytes
        filename: Original filename

    Returns:
        Metadata dictionary
    """
    metadata = {
        "filename": filename,
        "file_size_bytes": len(content),
        "file_hash": calculate_file_hash(content),
        "upload_timestamp": datetime.utcnow().isoformat(),
    }

    # Detect file type
    mime_type, file_category = detect_file_type(content, filename)
    metadata["content_type"] = mime_type
    metadata["file_type"] = file_category

    # Extract additional metadata based on file type
    try:
        if mime_type.startswith("image/"):
            image = Image.open(io.BytesIO(content))
            metadata["image_width"] = image.width
            metadata["image_height"] = image.height
            metadata["image_mode"] = image.mode

        elif mime_type == "application/pdf":
            doc = fitz.open(stream=content, filetype="pdf")
            metadata["pdf_page_count"] = len(doc)
            metadata["pdf_title"] = doc.metadata.get("title", "")
            metadata["pdf_author"] = doc.metadata.get("author", "")
            metadata["pdf_creation_date"] = doc.metadata.get("creationDate", "")
            doc.close()

    except Exception as e:
        logger.warning(f"Failed to extract metadata from {filename}: {e}")

    return metadata


def convert_document_to_metadata(document) -> dict[str, Any]:
    """
    Convert a Document database object to DocumentMetadata dictionary.

    Args:
        document: Document database object

    Returns:
        Dictionary compatible with DocumentMetadata Pydantic model
    """
    return {
        "id": document.id,
        "tenant_id": document.tenant_id,
        "filename": document.filename,
        "file_size_bytes": document.file_size_bytes,
        "content_type": document.content_type,
        "file_type": document.file_type,
        "file_hash": document.file_hash,
        "storage_path": document.storage_path,
        "upload_timestamp": document.upload_timestamp,
        "created_by": document.created_by,
        "processing_status": document.processing_status,
        "ocr_status": document.ocr_status,
        "ner_status": document.ner_status,
        "embedding_status": document.embedding_status,
        "ocr_text": document.ocr_text,
        "ner_entities": document.ner_entities,
        "embedding_vector": document.embedding_vector,
        "tags": document.tags or [],
        "description": document.description,
        "retention_date": document.retention_date,
        "is_deleted": document.is_deleted,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
    }
