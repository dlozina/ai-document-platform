"""
Tests for Ingestion Service
"""

import pytest
import io
from fastapi.testclient import TestClient
from PIL import Image
from PIL import ImageDraw
import fitz  # PyMuPDF

from src.main import app

client = TestClient(app)


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
    
    def test_health_check(self):
        """Test health endpoint returns correct status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "database_connected" in data
        assert "minio_connected" in data
        assert "dependencies" in data
        assert data["service"] == "ingestion-service"
        assert data["version"] == "1.0.0"


class TestUploadEndpoint:
    """Test file upload endpoint."""
    
    def test_upload_image(self, sample_image_bytes):
        """Test uploading an image file."""
        headers = {"X-Tenant-ID": "test-tenant"}
        files = {"file": ("test.png", sample_image_bytes, "image/png")}
        
        response = client.post("/upload", files=files, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "document_id" in data
        assert "filename" in data
        assert "file_size_bytes" in data
        assert "content_type" in data
        assert "file_hash" in data
        assert "upload_timestamp" in data
        assert "processing_status" in data
        assert "storage_path" in data
        assert "message" in data
        
        # Verify content
        assert data["filename"] == "test.png"
        assert data["content_type"] == "image/png"
        assert data["file_size_bytes"] == len(sample_image_bytes)
        assert data["processing_status"] == "pending"
    
    def test_upload_pdf(self, sample_pdf_bytes):
        """Test uploading a PDF file."""
        headers = {"X-Tenant-ID": "test-tenant"}
        files = {"file": ("test.pdf", sample_pdf_bytes, "application/pdf")}
        
        response = client.post("/upload", files=files, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["filename"] == "test.pdf"
        assert data["content_type"] == "application/pdf"
        assert data["file_size_bytes"] == len(sample_pdf_bytes)
    
    def test_upload_without_tenant_header(self, sample_image_bytes):
        """Test upload without tenant header."""
        files = {"file": ("test.png", sample_image_bytes, "image/png")}
        
        response = client.post("/upload", files=files)
        
        # Should fail if tenant header is required
        assert response.status_code == 400
    
    def test_upload_empty_file(self):
        """Test handling empty file."""
        headers = {"X-Tenant-ID": "test-tenant"}
        files = {"file": ("empty.txt", b"", "text/plain")}
        
        response = client.post("/upload", files=files, headers=headers)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_upload_unsupported_format(self, sample_image_bytes):
        """Test handling unsupported file format."""
        headers = {"X-Tenant-ID": "test-tenant"}
        files = {"file": ("test.txt", b"some text", "text/plain")}
        
        response = client.post("/upload", files=files, headers=headers)
        
        # Should fail for unsupported format
        assert response.status_code == 400


class TestBatchUploadEndpoint:
    """Test batch upload endpoint."""
    
    def test_batch_upload(self, sample_image_bytes, sample_pdf_bytes):
        """Test batch upload of multiple files."""
        headers = {"X-Tenant-ID": "test-tenant"}
        files = [
            ("files", ("test1.png", sample_image_bytes, "image/png")),
            ("files", ("test2.pdf", sample_pdf_bytes, "application/pdf"))
        ]
        
        response = client.post("/upload/batch", files=files, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "batch_id" in data
        assert "total_files" in data
        assert "successful_uploads" in data
        assert "failed_uploads" in data
        assert "documents" in data
        assert "errors" in data
        
        assert data["total_files"] == 2
        assert data["successful_uploads"] >= 0
        assert data["failed_uploads"] >= 0
    
    def test_batch_upload_too_many_files(self, sample_image_bytes):
        """Test batch upload with too many files."""
        headers = {"X-Tenant-ID": "test-tenant"}
        files = []
        
        # Create more than max_files_per_request files
        for i in range(15):  # Assuming max is 10
            files.append(("files", (f"test{i}.png", sample_image_bytes, "image/png")))
        
        response = client.post("/upload/batch", files=files, headers=headers)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data


class TestDocumentEndpoints:
    """Test document management endpoints."""
    
    def test_list_documents(self):
        """Test listing documents."""
        headers = {"X-Tenant-ID": "test-tenant"}
        
        response = client.get("/documents", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "documents" in data
        assert "total_count" in data
        assert "page" in data
        assert "page_size" in data
        assert "has_next" in data
    
    def test_get_document_not_found(self):
        """Test getting non-existent document."""
        headers = {"X-Tenant-ID": "test-tenant"}
        
        response = client.get("/documents/non-existent-id", headers=headers)
        
        assert response.status_code == 404
    
    def test_get_processing_status_not_found(self):
        """Test getting processing status for non-existent document."""
        headers = {"X-Tenant-ID": "test-tenant"}
        
        response = client.get("/documents/non-existent-id/processing", headers=headers)
        
        assert response.status_code == 404
    
    def test_update_document_not_found(self):
        """Test updating non-existent document."""
        headers = {"X-Tenant-ID": "test-tenant"}
        update_data = {"tags": ["test"]}
        
        response = client.put("/documents/non-existent-id", json=update_data, headers=headers)
        
        assert response.status_code == 404
    
    def test_delete_document_not_found(self):
        """Test deleting non-existent document."""
        headers = {"X-Tenant-ID": "test-tenant"}
        
        response = client.delete("/documents/non-existent-id", headers=headers)
        
        assert response.status_code == 404


class TestTenantQuotaEndpoint:
    """Test tenant quota endpoint."""
    
    def test_get_tenant_quota(self):
        """Test getting tenant quota."""
        response = client.get("/tenants/test-tenant/quota")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "tenant_id" in data
        assert "max_storage_bytes" in data
        assert "used_storage_bytes" in data
        assert "max_files" in data
        assert "used_files" in data
        assert "max_file_size_bytes" in data
        assert "retention_days" in data


class TestErrorHandling:
    """Test error handling."""
    
    def test_missing_file_parameter(self):
        """Test request without file parameter."""
        headers = {"X-Tenant-ID": "test-tenant"}
        
        response = client.post("/upload", headers=headers)
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_file_parameter(self):
        """Test with invalid file parameter."""
        headers = {"X-Tenant-ID": "test-tenant"}
        
        response = client.post("/upload", files={"invalid": ("test.txt", b"content", "text/plain")}, headers=headers)
        
        assert response.status_code == 422


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_schema(self):
        """Test OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "Ingestion Service"
    
    def test_docs_endpoint(self):
        """Test docs endpoint is accessible."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint(self):
        """Test ReDoc endpoint is accessible."""
        response = client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
