"""
API Tests for NER Service
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_text():
    """Create sample text for testing."""
    return """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
    The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
    Apple's revenue for 2023 was $394.3 billion, representing a 2.8% increase from the previous year.
    The company employs over 164,000 people worldwide and operates in more than 50 countries.
    """


@pytest.fixture
def sample_texts():
    """Create sample texts for batch testing."""
    return [
        "John Smith works at Microsoft Corporation in Seattle, Washington.",
        "The meeting is scheduled for March 15, 2024 at 2:30 PM.",
        "Apple Inc. reported $100 billion in revenue last quarter.",
        "Dr. Sarah Johnson is a professor at Stanford University.",
        "The conference will be held in New York City next month.",
    ]


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
        assert "spacy_models_available" in data
        assert data["service"] == "ner-service"
        assert data["version"] == "1.0.0"


class TestExtractEndpoint:
    """Test entity extraction endpoint."""

    def test_extract_entities(self, client, sample_text):
        """Test extracting entities from text."""
        request_data = {"text": sample_text, "include_confidence": True}

        response = client.post("/extract", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "text" in data
        assert "entities" in data
        assert "entity_count" in data
        assert "model_used" in data
        assert "processing_time_ms" in data
        assert "text_length" in data

        # Verify content
        assert isinstance(data["entities"], list)
        assert data["text_length"] == len(sample_text)
        assert data["processing_time_ms"] > 0

        # Check if entities were found
        if data["entity_count"] > 0:
            entity = data["entities"][0]
            assert "text" in entity
            assert "label" in entity
            assert "start" in entity
            assert "end" in entity
            assert "confidence" in entity

    def test_extract_with_entity_types(self, client, sample_text):
        """Test extracting specific entity types."""
        request_data = {
            "text": sample_text,
            "entity_types": ["PERSON", "ORG"],
            "include_confidence": True,
        }

        response = client.post("/extract", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Check that only specified entity types are returned
        for entity in data["entities"]:
            assert entity["label"] in ["PERSON", "ORG"]

    def test_extract_with_headers(self, client, sample_text):
        """Test extraction with custom headers."""
        headers = {"X-Tenant-ID": "test-tenant", "X-Document-ID": "doc-123"}
        request_data = {"text": sample_text, "include_confidence": True}

        response = client.post("/extract", json=request_data, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "doc-123"

    def test_extract_empty_text(self, client):
        """Test handling empty text."""
        request_data = {"text": "", "include_confidence": True}

        response = client.post("/extract", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_extract_long_text(self, client):
        """Test handling very long text."""
        long_text = "A" * 2000000  # 2M characters
        request_data = {"text": long_text, "include_confidence": True}

        response = client.post("/extract", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "error" in data


class TestExtractBatchEndpoint:
    """Test batch extraction endpoint."""

    def test_extract_batch(self, client, sample_texts):
        """Test batch entity extraction."""
        request_data = {"texts": sample_texts, "include_confidence": True}

        response = client.post("/extract-batch", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "results" in data
        assert "total_processing_time_ms" in data
        assert "batch_size" in data

        # Verify content
        assert len(data["results"]) == len(sample_texts)
        assert data["batch_size"] == len(sample_texts)
        assert data["total_processing_time_ms"] > 0

        # Check individual results
        for result in data["results"]:
            assert "text" in result
            assert "entities" in result
            assert "entity_count" in result

    def test_extract_batch_with_entity_types(self, client, sample_texts):
        """Test batch extraction with specific entity types."""
        request_data = {
            "texts": sample_texts,
            "entity_types": ["PERSON", "ORG"],
            "include_confidence": True,
        }

        response = client.post("/extract-batch", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Check that only specified entity types are returned
        for result in data["results"]:
            for entity in result["entities"]:
                assert entity["label"] in ["PERSON", "ORG"]

    def test_extract_batch_empty_list(self, client):
        """Test handling empty text list."""
        request_data = {"texts": [], "include_confidence": True}

        response = client.post("/extract-batch", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_extract_batch_too_many_texts(self, client):
        """Test handling too many texts."""
        texts = ["Sample text"] * 150  # More than 100
        request_data = {"texts": texts, "include_confidence": True}

        response = client.post("/extract-batch", json=request_data)

        assert response.status_code == 422  # Validation error


class TestExtractAsyncEndpoint:
    """Test async extraction endpoint."""

    def test_extract_async(self, client, sample_text):
        """Test async extraction."""
        request_data = {"text": sample_text, "include_confidence": True}

        response = client.post("/extract-async", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "job_id" in data
        assert "status" in data
        assert "message" in data
        assert data["status"] == "processing"

    def test_extract_async_with_callback(self, client, sample_text):
        """Test async extraction with callback URL."""
        request_data = {"text": sample_text, "include_confidence": True}
        params = {"callback_url": "http://example.com/callback"}

        response = client.post("/extract-async", json=request_data, params=params)

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data


class TestStatsEndpoint:
    """Test statistics endpoint."""

    def test_get_stats(self, client, sample_text):
        """Test getting entity statistics."""
        request_data = {"text": sample_text, "include_confidence": True}

        response = client.post("/stats", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "total_entities" in data
        assert "entity_types" in data
        assert "most_common_entities" in data

        assert isinstance(data["total_entities"], int)
        assert isinstance(data["entity_types"], list)
        assert isinstance(data["most_common_entities"], list)


class TestVisualizeEndpoint:
    """Test visualization endpoint."""

    def test_visualize_entities(self, client, sample_text):
        """Test generating entity visualization."""
        request_data = {"text": sample_text, "include_confidence": True}

        response = client.post("/visualize", json=request_data)

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Check that HTML content is returned
        html_content = response.text
        assert "<html" in html_content.lower()
        assert "<body" in html_content.lower()


class TestModelsEndpoint:
    """Test models information endpoint."""

    def test_get_models(self, client):
        """Test getting available models."""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()

        assert "available_models" in data
        assert "primary_model" in data
        assert "fallback_model" in data

        assert isinstance(data["available_models"], dict)
        assert isinstance(data["primary_model"], str)
        assert isinstance(data["fallback_model"], str)


class TestErrorHandling:
    """Test error handling."""

    def test_missing_text(self, client):
        """Test request without text."""
        request_data = {"include_confidence": True}

        response = client.post("/extract", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_invalid_entity_types(self, client, sample_text):
        """Test with invalid entity types."""
        request_data = {
            "text": sample_text,
            "entity_types": ["INVALID_TYPE"],
            "include_confidence": True,
        }

        response = client.post("/extract", json=request_data)

        # Should still work, just return no entities
        assert response.status_code == 200
        data = response.json()
        assert data["entity_count"] == 0


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is accessible."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()

        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "NER Service"

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
