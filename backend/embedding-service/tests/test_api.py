"""
Tests for Embedding Service API endpoints
"""

from unittest.mock import patch

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_success(self):
        """Test successful health check."""
        with patch("src.main.embedding_processor") as mock_processor:
            mock_processor.health_check.return_value = {
                "embedding_model_loaded": True,
                "qdrant_available": True,
                "collection_exists": True,
            }

            response = client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "healthy"
            assert data["service"] == "embedding-service"
            assert data["version"] == "1.0.0"
            assert data["qdrant_available"] is True
            assert data["embedding_model_loaded"] is True

    def test_health_check_degraded(self):
        """Test degraded health check."""
        with patch("src.main.embedding_processor") as mock_processor:
            mock_processor.health_check.return_value = {
                "embedding_model_loaded": True,
                "qdrant_available": False,
                "collection_exists": False,
            }

            response = client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "degraded"


class TestEmbeddingEndpoints:
    """Test embedding generation endpoints."""

    def test_generate_embedding_success(self):
        """Test successful embedding generation."""
        with patch("src.main.embedding_processor") as mock_processor:
            mock_embedding = [0.1, 0.2, 0.3, 0.4]
            mock_processor.generate_embedding.return_value = mock_embedding

            request_data = {
                "text": "This is a test text",
                "document_id": "test_doc_123",
            }

            response = client.post("/embed", json=request_data)
            assert response.status_code == 200

            data = response.json()
            assert data["text"] == "This is a test text"
            assert data["embedding"] == mock_embedding
            assert data["embedding_dimension"] == len(mock_embedding)
            assert data["document_id"] == "test_doc_123"

    def test_generate_embedding_empty_text(self):
        """Test embedding generation with empty text."""
        request_data = {"text": "", "document_id": "test_doc_123"}

        response = client.post("/embed", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_generate_embedding_long_text(self):
        """Test embedding generation with text too long."""
        long_text = "a" * 10001  # Exceeds max length

        request_data = {"text": long_text, "document_id": "test_doc_123"}

        response = client.post("/embed", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_batch_embedding_success(self):
        """Test successful batch embedding generation."""
        with patch("src.main.embedding_processor") as mock_processor:
            mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
            mock_processor.generate_batch_embeddings.return_value = mock_embeddings

            request_data = {
                "texts": ["Text 1", "Text 2", "Text 3"],
                "document_ids": ["doc1", "doc2", "doc3"],
            }

            response = client.post("/embed-batch", json=request_data)
            assert response.status_code == 200

            data = response.json()
            assert len(data["embeddings"]) == 3
            assert data["batch_size"] == 3
            assert data["document_ids"] == ["doc1", "doc2", "doc3"]

    def test_batch_embedding_too_many_texts(self):
        """Test batch embedding with too many texts."""
        texts = ["text"] * 101  # Exceeds max batch size

        request_data = {"texts": texts}

        response = client.post("/embed-batch", json=request_data)
        assert response.status_code == 400


class TestSearchEndpoints:
    """Test search endpoints."""

    def test_search_similar_success(self):
        """Test successful similarity search."""
        with patch("src.main.embedding_processor") as mock_processor:
            mock_results = [
                {
                    "id": "doc1",
                    "score": 0.95,
                    "text": "Similar text",
                    "metadata": {"source": "test"},
                },
                {
                    "id": "doc2",
                    "score": 0.87,
                    "text": "Another similar text",
                    "metadata": {"source": "test"},
                },
            ]
            mock_processor.search_similar.return_value = mock_results

            request_data = {"query": "test query", "limit": 10}

            response = client.post("/search", json=request_data)
            assert response.status_code == 200

            data = response.json()
            assert data["query"] == "test query"
            assert len(data["results"]) == 2
            assert data["total_results"] == 2
            assert data["results"][0]["score"] == 0.95

    def test_search_with_filters(self):
        """Test search with metadata filters."""
        with patch("src.main.embedding_processor") as mock_processor:
            mock_processor.search_similar.return_value = []

            request_data = {
                "query": "test query",
                "limit": 5,
                "score_threshold": 0.8,
                "filter": {"category": "test"},
            }

            response = client.post("/search", json=request_data)
            assert response.status_code == 200

            # Verify that search_similar was called with correct parameters
            mock_processor.search_similar.assert_called_once()
            call_args = mock_processor.search_similar.call_args
            assert call_args[1]["query"] == "test query"
            assert call_args[1]["limit"] == 5
            assert call_args[1]["score_threshold"] == 0.8
            assert call_args[1]["filter_conditions"] == {"category": "test"}


class TestFileEndpoints:
    """Test file processing endpoints."""

    def test_embed_file_success(self):
        """Test successful file embedding."""
        with patch("src.main.embedding_processor") as mock_processor:
            mock_result = {
                "text": "File content",
                "embedding": [0.1, 0.2, 0.3],
                "embedding_dimension": 3,
                "model_name": "test-model",
                "processing_time_ms": 100.0,
                "text_length": 12,
                "filename": "test.txt",
                "point_id": "test_point",
                "method": "embed_text_txt",
            }
            mock_processor.process_text_file.return_value = mock_result

            file_content = b"This is test file content"
            files = {"file": ("test.txt", file_content, "text/plain")}

            response = client.post("/embed-file", files=files)
            assert response.status_code == 200

            data = response.json()
            assert data["text"] == "File content"
            assert data["filename"] == "test.txt"
            assert data["embedding_dimension"] == 3

    def test_embed_file_empty(self):
        """Test embedding empty file."""
        files = {"file": ("empty.txt", b"", "text/plain")}

        response = client.post("/embed-file", files=files)
        assert response.status_code == 400
        assert "Empty file uploaded" in response.json()["detail"]

    def test_embed_file_unsupported_format(self):
        """Test embedding unsupported file format."""
        files = {"file": ("test.exe", b"binary content", "application/octet-stream")}

        response = client.post("/embed-file", files=files)
        assert response.status_code == 400


class TestCollectionEndpoints:
    """Test collection management endpoints."""

    def test_get_collection_info_success(self):
        """Test successful collection info retrieval."""
        with patch("src.main.embedding_processor") as mock_processor:
            mock_info = {
                "name": "embeddings",
                "vector_size": 384,
                "points_count": 1000,
                "status": "green",
            }
            mock_processor.get_collection_info.return_value = mock_info

            response = client.get("/collection/info")
            assert response.status_code == 200

            data = response.json()
            assert data["name"] == "embeddings"
            assert data["vector_size"] == 384
            assert data["points_count"] == 1000

    def test_delete_embedding_success(self):
        """Test successful embedding deletion."""
        with patch("src.main.embedding_processor") as mock_processor:
            mock_processor.delete_embedding.return_value = True

            response = client.delete("/embedding/test_point_id")
            assert response.status_code == 200

            data = response.json()
            assert "deleted successfully" in data["message"]

    def test_delete_embedding_not_found(self):
        """Test deletion of non-existent embedding."""
        with patch("src.main.embedding_processor") as mock_processor:
            mock_processor.delete_embedding.return_value = False

            response = client.delete("/embedding/nonexistent_point")
            assert response.status_code == 404


class TestErrorHandling:
    """Test error handling."""

    def test_service_not_initialized(self):
        """Test error when service is not initialized."""
        with patch("src.main.embedding_processor", None):
            response = client.post("/embed", json={"text": "test"})
            assert response.status_code == 503
            assert "not initialized" in response.json()["detail"]

    def test_internal_server_error(self):
        """Test internal server error handling."""
        with patch("src.main.embedding_processor") as mock_processor:
            mock_processor.generate_embedding.side_effect = Exception("Test error")

            response = client.post("/embed", json={"text": "test"})
            assert response.status_code == 500
            assert "Failed to generate embedding" in response.json()["detail"]
