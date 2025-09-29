"""
API Tests for OCR Service
"""

import pytest
import io
from fastapi.testclient import TestClient
from PIL import Image
from PIL import ImageDraw
import fitz  # PyMuPDF

from src.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create sample image for testing."""
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), "Test Document 12345", fill='black')
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()


@pytest.fixture
def sample_pdf_bytes():
    """Create sample PDF for testing."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 50), "This is a test PDF document.\nIt has multiple lines.", fontsize=12)
    
    pdf_bytes = doc.write()
    doc.close()
    
    return pdf_bytes


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health endpoint returns correct status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "tesseract_available" in data
        assert data["service"] == "ocr-service"
        assert data["version"] == "1.0.0"


class TestExtractEndpoint:
    """Test text extraction endpoint."""
    
    def test_extract_image(self, client, sample_image_bytes):
        """Test extracting text from image."""
        files = {"file": ("test.png", sample_image_bytes, "image/png")}
        
        response = client.post("/extract", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "text" in data
        assert "page_count" in data
        assert "method" in data
        assert "confidence" in data
        assert "processing_time_ms" in data
        assert "file_size_bytes" in data
        assert "filename" in data
        
        # Verify content
        assert data["page_count"] == 1
        assert data["method"] == "ocr_image"
        assert isinstance(data["confidence"], float)
        assert data["filename"] == "test.png"
        assert data["file_size_bytes"] == len(sample_image_bytes)
    
    def test_extract_pdf(self, client, sample_pdf_bytes):
        """Test extracting text from PDF."""
        files = {"file": ("test.pdf", sample_pdf_bytes, "application/pdf")}
        
        response = client.post("/extract", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["method"] == "native_pdf"
        assert data["confidence"] is None  # Native extraction
        assert data["page_count"] == 1
        assert "test PDF document" in data["text"].lower()
    
    def test_extract_with_headers(self, client, sample_image_bytes):
        """Test extraction with custom headers."""
        headers = {
            "X-Tenant-ID": "test-tenant",
            "X-Document-ID": "doc-123"
        }
        files = {"file": ("test.png", sample_image_bytes, "image/png")}
        
        response = client.post("/extract", files=files, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "doc-123"
    
    def test_extract_force_ocr(self, client, sample_pdf_bytes):
        """Test forcing OCR on PDF."""
        files = {"file": ("test.pdf", sample_pdf_bytes, "application/pdf")}
        params = {"force_ocr": True}
        
        response = client.post("/extract", files=files, params=params)
        
        assert response.status_code == 200
        data = response.json()
        assert data["method"] == "ocr_pdf"
        assert isinstance(data["confidence"], float)
    
    def test_extract_empty_file(self, client):
        """Test handling empty file."""
        files = {"file": ("empty.txt", b"", "text/plain")}
        
        response = client.post("/extract", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_extract_unsupported_format(self, client):
        """Test handling unsupported file format."""
        files = {"file": ("test.txt", b"some text", "text/plain")}
        
        response = client.post("/extract", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_extract_large_file(self, client):
        """Test handling large file."""
        # Create a large file (11MB)
        large_content = b"x" * (11 * 1024 * 1024)
        files = {"file": ("large.pdf", large_content, "application/pdf")}
        
        response = client.post("/extract", files=files)
        
        assert response.status_code == 413
        data = response.json()
        assert "error" in data


class TestExtractAsyncEndpoint:
    """Test async extraction endpoint."""
    
    def test_extract_async(self, client, sample_image_bytes):
        """Test async extraction."""
        files = {"file": ("test.png", sample_image_bytes, "image/png")}
        
        response = client.post("/extract-async", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "job_id" in data
        assert "status" in data
        assert "message" in data
        assert data["status"] == "processing"
    
    def test_extract_async_with_callback(self, client, sample_image_bytes):
        """Test async extraction with callback URL."""
        files = {"file": ("test.png", sample_image_bytes, "image/png")}
        params = {"callback_url": "http://example.com/callback"}
        
        response = client.post("/extract-async", files=files, params=params)
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data


class TestExtractLayoutEndpoint:
    """Test layout extraction endpoint."""
    
    def test_extract_layout(self, client, sample_image_bytes):
        """Test extracting text with layout information."""
        files = {"file": ("test.png", sample_image_bytes, "image/png")}
        
        response = client.post("/extract-layout", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "filename" in data
        assert "word_count" in data
        assert "words" in data
        assert isinstance(data["words"], list)
        
        if data["words"]:  # If words were detected
            word = data["words"][0]
            assert "text" in word
            assert "confidence" in word
            assert "left" in word
            assert "top" in word
            assert "width" in word
            assert "height" in word
            assert "page_num" in word


class TestErrorHandling:
    """Test error handling."""
    
    def test_missing_file(self, client):
        """Test request without file."""
        response = client.post("/extract")
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_file_parameter(self, client):
        """Test with invalid file parameter."""
        response = client.post("/extract", files={"invalid": ("test.txt", b"content", "text/plain")})
        
        assert response.status_code == 422


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "OCR Service"
    
    def test_docs_endpoint(self, client):
        """Test docs endpoint is accessible."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint(self, client):
        """Test ReDoc endpoint is accessible."""
        response = client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
