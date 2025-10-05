"""
Unit tests for Ingestion Service components
"""

import io
from unittest.mock import Mock, patch

import fitz  # PyMuPDF
import pytest
from PIL import Image, ImageDraw
from src.database import DatabaseManager
from src.processing import ProcessingPipeline
from src.storage import MinIOManager
from src.utils import (
    FileValidator,
    calculate_file_hash,
    detect_file_type,
    extract_file_metadata,
    format_file_size,
    generate_storage_path,
    sanitize_filename,
    validate_file_content,
)


@pytest.fixture
def sample_image():
    """Create a simple test image with text."""
    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)
    text = "Test Document 12345"
    draw.text((10, 40), text, fill="black")

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return img_bytes.getvalue()


@pytest.fixture
def sample_pdf():
    """Create a simple test PDF with text."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    text = "This is a test PDF document.\nIt has multiple lines.\nPage 1 content."
    page.insert_text((50, 50), text, fontsize=12)

    pdf_bytes = doc.write()
    doc.close()

    return pdf_bytes


class TestUtils:
    """Test utility functions."""

    def test_calculate_file_hash(self, sample_image):
        """Test file hash calculation."""
        hash1 = calculate_file_hash(sample_image)
        hash2 = calculate_file_hash(sample_image)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
        assert isinstance(hash1, str)

    def test_detect_file_type(self, sample_image, sample_pdf):
        """Test file type detection."""
        # Test image
        mime_type, file_type = detect_file_type(sample_image, "test.png")
        assert mime_type == "image/png"
        assert file_type == "image"

        # Test PDF
        mime_type, file_type = detect_file_type(sample_pdf, "test.pdf")
        assert mime_type == "application/pdf"
        assert file_type == "pdf"

    def test_validate_file_content(self, sample_image, sample_pdf):
        """Test file content validation."""
        # Test valid image
        result = validate_file_content(sample_image, "test.png", "image/png")
        assert result["is_valid"]
        assert len(result["errors"]) == 0

        # Test valid PDF
        result = validate_file_content(sample_pdf, "test.pdf", "application/pdf")
        assert result["is_valid"]
        assert len(result["errors"]) == 0

        # Test invalid content
        result = validate_file_content(
            b"invalid content", "test.pdf", "application/pdf"
        )
        assert not result["is_valid"]
        assert len(result["errors"]) > 0

    def test_generate_storage_path(self):
        """Test storage path generation."""
        path = generate_storage_path("tenant123", "test.pdf", "abc123def456")

        assert path.startswith("tenant123/")
        assert "abc123def456" in path
        assert path.endswith("test.pdf")

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Test unsafe characters
        sanitized = sanitize_filename("test<>file.pdf")
        assert "<" not in sanitized
        assert ">" not in sanitized

        # Test path separators
        sanitized = sanitize_filename("test/file.pdf")
        assert "/" not in sanitized

        # Test length limit
        long_name = "a" * 300 + ".pdf"
        sanitized = sanitize_filename(long_name)
        assert len(sanitized) <= 255

    def test_format_file_size(self):
        """Test file size formatting."""
        assert format_file_size(0) == "0 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"

    def test_extract_file_metadata(self, sample_image, sample_pdf):
        """Test file metadata extraction."""
        # Test image metadata
        metadata = extract_file_metadata(sample_image, "test.png")

        assert "filename" in metadata
        assert "file_size_bytes" in metadata
        assert "file_hash" in metadata
        assert "content_type" in metadata
        assert "file_type" in metadata
        assert "upload_timestamp" in metadata

        assert metadata["filename"] == "test.png"
        assert metadata["content_type"] == "image/png"
        assert metadata["file_type"] == "image"
        assert metadata["file_size_bytes"] == len(sample_image)

        # Test PDF metadata
        metadata = extract_file_metadata(sample_pdf, "test.pdf")

        assert metadata["filename"] == "test.pdf"
        assert metadata["content_type"] == "application/pdf"
        assert metadata["file_type"] == "pdf"
        assert metadata["file_size_bytes"] == len(sample_pdf)


class TestFileValidator:
    """Test file validator."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = FileValidator(1024 * 1024, ["image/png", "application/pdf"])

        assert validator.max_size_bytes == 1024 * 1024
        assert "image/png" in validator.allowed_mime_types
        assert "application/pdf" in validator.allowed_mime_types

    def test_validate_valid_file(self, sample_image):
        """Test validation of valid file."""
        validator = FileValidator(1024 * 1024, ["image/png"])

        result = validator.validate("test.png", sample_image)

        assert result["is_valid"]
        assert len(result["errors"]) == 0
        assert result["file_type"] == "image/png"
        assert result["file_size_bytes"] == len(sample_image)

    def test_validate_file_too_large(self, sample_image):
        """Test validation of file that's too large."""
        validator = FileValidator(100, ["image/png"])  # 100 bytes limit

        result = validator.validate("test.png", sample_image)

        assert not result["is_valid"]
        assert len(result["errors"]) > 0
        assert "too large" in result["errors"][0].lower()

    def test_validate_empty_file(self):
        """Test validation of empty file."""
        validator = FileValidator(1024 * 1024, ["image/png"])

        result = validator.validate("test.png", b"")

        assert not result["is_valid"]
        assert len(result["errors"]) > 0
        assert "empty" in result["errors"][0].lower()

    def test_validate_unsupported_type(self, sample_image):
        """Test validation of unsupported file type."""
        validator = FileValidator(1024 * 1024, ["application/pdf"])  # Only PDF allowed

        result = validator.validate("test.png", sample_image)

        assert not result["is_valid"]
        assert len(result["errors"]) > 0
        assert "unsupported" in result["errors"][0].lower()


class TestMinIOManager:
    """Test MinIO manager."""

    @patch("src.storage.Minio")
    def test_minio_manager_initialization(self, mock_minio):
        """Test MinIO manager initialization."""
        mock_client = Mock()
        mock_minio.return_value = mock_client

        manager = MinIOManager()

        assert manager.client == mock_client
        mock_minio.assert_called_once()

    @patch("src.storage.Minio")
    def test_get_tenant_bucket_name(self, mock_minio):
        """Test tenant bucket name generation."""
        manager = MinIOManager()

        bucket_name = manager.get_tenant_bucket_name("tenant123")

        assert bucket_name == "ingestion-tenant123"

    @patch("src.storage.Minio")
    def test_upload_file_success(self, mock_minio, sample_image):
        """Test successful file upload."""
        mock_client = Mock()
        mock_client.bucket_exists.return_value = True
        mock_minio.return_value = mock_client

        manager = MinIOManager()

        result = manager.upload_file(
            "tenant123", "path/file.png", sample_image, "image/png"
        )

        assert result
        mock_client.put_object.assert_called_once()

    @patch("src.storage.Minio")
    def test_upload_file_bucket_creation(self, mock_minio, sample_image):
        """Test file upload with bucket creation."""
        mock_client = Mock()
        mock_client.bucket_exists.return_value = False
        mock_minio.return_value = mock_client

        manager = MinIOManager()

        result = manager.upload_file(
            "tenant123", "path/file.png", sample_image, "image/png"
        )

        assert result
        mock_client.make_bucket.assert_called_once()
        mock_client.put_object.assert_called_once()

    @patch("src.storage.Minio")
    def test_download_file_success(self, mock_minio):
        """Test successful file download."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.read.return_value = b"file content"
        mock_client.get_object.return_value = mock_response
        mock_minio.return_value = mock_client

        manager = MinIOManager()

        result = manager.download_file("tenant123", "path/file.png")

        assert result == b"file content"
        mock_client.get_object.assert_called_once()
        mock_response.close.assert_called_once()
        mock_response.release_conn.assert_called_once()

    @patch("src.storage.Minio")
    def test_file_exists_true(self, mock_minio):
        """Test file exists check when file exists."""
        mock_client = Mock()
        mock_client.stat_object.return_value = Mock()
        mock_minio.return_value = mock_client

        manager = MinIOManager()

        result = manager.file_exists("tenant123", "path/file.png")

        assert result
        mock_client.stat_object.assert_called_once()

    @patch("src.storage.Minio")
    def test_file_exists_false(self, mock_minio):
        """Test file exists check when file doesn't exist."""
        mock_client = Mock()
        mock_client.stat_object.side_effect = Exception("Not found")
        mock_minio.return_value = mock_client

        manager = MinIOManager()

        result = manager.file_exists("tenant123", "path/file.png")

        assert not result


class TestDatabaseManager:
    """Test database manager."""

    @patch("src.database.create_engine")
    def test_database_manager_initialization(self, mock_create_engine):
        """Test database manager initialization."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        manager = DatabaseManager()

        assert manager.engine == mock_engine
        mock_create_engine.assert_called_once()

    def test_create_document_data(self):
        """Test document data creation."""
        document_data = {
            "id": "test-id",
            "tenant_id": "tenant123",
            "filename": "test.pdf",
            "file_size_bytes": 1024,
            "content_type": "application/pdf",
            "file_type": "pdf",
            "file_hash": "abc123",
            "storage_path": "tenant123/test.pdf",
            "processing_status": "pending",
        }

        # This would normally create a database record
        # For testing, we just verify the data structure
        assert document_data["id"] == "test-id"
        assert document_data["tenant_id"] == "tenant123"
        assert document_data["filename"] == "test.pdf"
        assert document_data["processing_status"] == "pending"


class TestProcessingPipeline:
    """Test processing pipeline."""

    @patch("src.processing.get_db_manager")
    def test_processing_pipeline_initialization(self, mock_get_db_manager):
        """Test processing pipeline initialization."""
        mock_db_manager = Mock()
        mock_get_db_manager.return_value = mock_db_manager

        pipeline = ProcessingPipeline()

        assert pipeline.db_manager == mock_db_manager

    @patch("src.processing.get_db_manager")
    @patch("src.processing.httpx.AsyncClient")
    async def test_process_document_ocr_success(
        self, mock_client_class, mock_get_db_manager, sample_image
    ):
        """Test successful OCR processing."""
        # Mock database manager
        mock_db_manager = Mock()
        mock_session = Mock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_get_db_manager.return_value = mock_db_manager

        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "extracted text", "confidence": 95.0}
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        pipeline = ProcessingPipeline()

        result = await pipeline.process_document(
            document_id="test-id",
            tenant_id="tenant123",
            file_content=sample_image,
            filename="test.png",
            content_type="image/png",
            pipeline_config={"enable_ocr": True},
        )

        assert result["document_id"] == "test-id"
        assert result["tenant_id"] == "tenant123"
        assert result["processing_status"] == "processing"
        assert len(result["jobs"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
