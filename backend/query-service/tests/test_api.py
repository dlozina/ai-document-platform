"""
Tests for Query Service API
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.main import app
from src.models import QueryRequest, QueryMode, QueryFilter, DateRange
from src.query_processor import QueryProcessor
from src.config import get_settings

# Test client
client = TestClient(app)


@pytest.fixture
def mock_query_processor():
    """Mock query processor for testing."""
    with patch('src.main.query_processor') as mock:
        processor = Mock(spec=QueryProcessor)
        processor.health_check.return_value = {
            "embedding_model_loaded": True,
            "qdrant_available": True,
            "database_connected": True,
            "nlp_loaded": True,
            "rerank_model_loaded": True
        }
        processor.semantic_search.return_value = [
            {
                "id": "test-id-1",
                "score": 0.95,
                "text": "This is a test document about Python programming.",
                "document_id": "doc-1",
                "filename": "test.pdf",
                "metadata": {"tenant_id": "test-tenant"}
            }
        ]
        processor.extract_entities.return_value = [
            {"text": "Python", "label": "ORG", "confidence": 0.8}
        ]
        processor.extract_answer_span.return_value = ("Python is a programming language.", 0.9)
        processor.generate_rag_answer.return_value = ("Based on the documents, Python is a programming language.", 0.85)
        processor.filter_by_entities.return_value = []
        processor.rerank_results.return_value = []
        mock.return_value = processor
        yield processor


@pytest.fixture
def sample_query_request():
    """Sample query request for testing."""
    return {
        "query": "What programming languages does Dino know?",
        "mode": "semantic_search",
        "top_k": 10,
        "score_threshold": 0.7,
        "filter": {
            "tenant_id": "test-tenant",
            "content_type": "application/pdf"
        }
    }


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_success(self, mock_query_processor):
        """Test successful health check."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "query-service"
        assert data["version"] == "1.0.0"
        assert data["qdrant_available"] is True
        assert data["database_connected"] is True
        assert data["embedding_model_loaded"] is True
    
    def test_health_check_degraded(self, mock_query_processor):
        """Test degraded health check."""
        mock_query_processor.health_check.return_value = {
            "embedding_model_loaded": True,
            "qdrant_available": False,
            "database_connected": True,
            "nlp_loaded": True,
            "rerank_model_loaded": True
        }
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "degraded"
        assert data["qdrant_available"] is False


class TestQueryEndpoint:
    """Test main query endpoint."""
    
    def test_semantic_search_query(self, mock_query_processor, sample_query_request):
        """Test semantic search query."""
        response = client.post("/query", json=sample_query_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "confidence_score" in data
        assert "sources" in data
        assert "detected_entities" in data
        assert data["query_mode"] == "semantic_search"
        assert data["retrieved_documents_count"] >= 0
    
    def test_extractive_qa_query(self, mock_query_processor):
        """Test extractive QA query."""
        request = {
            "query": "What is Python?",
            "mode": "extractive_qa",
            "top_k": 5
        }
        
        response = client.post("/query", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["query_mode"] == "extractive_qa"
        assert "answer" in data
        assert data["confidence_score"] > 0
    
    def test_rag_query(self, mock_query_processor):
        """Test RAG query."""
        request = {
            "query": "Explain Python programming",
            "mode": "rag",
            "top_k": 3
        }
        
        response = client.post("/query", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["query_mode"] == "rag"
        assert "answer" in data
        assert data["confidence_score"] > 0
    
    def test_query_with_tenant_header(self, mock_query_processor, sample_query_request):
        """Test query with tenant header."""
        headers = {"X-Tenant-ID": "test-tenant"}
        response = client.post("/query", json=sample_query_request, headers=headers)
        assert response.status_code == 200
    
    def test_query_validation_error(self, mock_query_processor):
        """Test query validation error."""
        invalid_request = {
            "query": "",  # Empty query should fail
            "mode": "semantic_search"
        }
        
        response = client.post("/query", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_query_with_entity_filter(self, mock_query_processor):
        """Test query with entity filter."""
        request = {
            "query": "Find documents about Python",
            "mode": "semantic_search",
            "filter": {
                "entity_labels": ["ORG", "PERSON"],
                "entity_text": ["Python", "Programming"]
            }
        }
        
        response = client.post("/query", json=request)
        assert response.status_code == 200


class TestSemanticSearchEndpoint:
    """Test semantic search endpoint."""
    
    def test_semantic_search_success(self, mock_query_processor):
        """Test successful semantic search."""
        request = {
            "query": "Python programming",
            "top_k": 10,
            "score_threshold": 0.7
        }
        
        response = client.post("/search", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["query"] == "Python programming"
        assert "results" in data
        assert "total_results" in data
        assert "search_time_ms" in data
    
    def test_semantic_search_with_filter(self, mock_query_processor):
        """Test semantic search with filter."""
        request = {
            "query": "Python programming",
            "top_k": 5,
            "filter": {
                "tenant_id": "test-tenant",
                "file_type": "pdf"
            }
        }
        
        response = client.post("/search", json=request)
        assert response.status_code == 200


class TestQAEndpoint:
    """Test question-answering endpoint."""
    
    def test_extractive_qa_success(self, mock_query_processor):
        """Test successful extractive QA."""
        request = {
            "question": "What is Python?",
            "mode": "extractive_qa",
            "top_k": 5
        }
        
        response = client.post("/qa", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "confidence_score" in data
        assert "sources" in data
        assert data["mode"] == "extractive_qa"
    
    def test_rag_success(self, mock_query_processor):
        """Test successful RAG."""
        request = {
            "question": "Explain Python programming",
            "mode": "rag",
            "top_k": 3,
            "max_context_length": 2000
        }
        
        response = client.post("/qa", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["mode"] == "rag"
        assert "answer" in data
        assert "llm_provider" in data


class TestCollectionInfoEndpoint:
    """Test collection info endpoint."""
    
    def test_collection_info_success(self, mock_query_processor):
        """Test successful collection info retrieval."""
        # Mock Qdrant collection info
        mock_collection = Mock()
        mock_collection.config.params.vectors.size = 384
        mock_collection.points_count = 1000
        mock_collection.status = "green"
        
        mock_query_processor.qdrant_client.get_collection.return_value = mock_collection
        
        response = client.get("/collection/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "vector_size" in data
        assert "points_count" in data
        assert "status" in data


class TestErrorHandling:
    """Test error handling."""
    
    def test_service_not_initialized(self):
        """Test error when service is not initialized."""
        with patch('src.main.query_processor', None):
            response = client.post("/query", json={"query": "test"})
            assert response.status_code == 503
    
    def test_invalid_query_mode(self, mock_query_processor):
        """Test invalid query mode."""
        request = {
            "query": "test query",
            "mode": "invalid_mode"
        }
        
        response = client.post("/query", json=request)
        assert response.status_code == 400
    
    def test_query_processor_exception(self, mock_query_processor):
        """Test query processor exception."""
        mock_query_processor.semantic_search.side_effect = Exception("Test error")
        
        request = {
            "query": "test query",
            "mode": "semantic_search"
        }
        
        response = client.post("/query", json=request)
        assert response.status_code == 500


class TestQueryProcessor:
    """Test QueryProcessor class."""
    
    @patch('src.query_processor.SentenceTransformer')
    @patch('src.query_processor.QdrantClient')
    @patch('src.query_processor.create_engine')
    @patch('src.query_processor.spacy.load')
    def test_query_processor_initialization(self, mock_spacy, mock_engine, mock_qdrant, mock_sentence_transformer):
        """Test QueryProcessor initialization."""
        processor = QueryProcessor()
        
        # Verify all components are initialized
        assert processor.embedding_model is not None
        assert processor.qdrant_client is not None
        assert processor.db_engine is not None
        assert processor.nlp is not None
    
    def test_generate_query_embedding(self):
        """Test query embedding generation."""
        with patch('src.query_processor.SentenceTransformer') as mock_model:
            mock_embedding = Mock()
            mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
            mock_model.return_value.encode.return_value = mock_embedding
            
            processor = QueryProcessor()
            embedding = processor.generate_query_embedding("test query")
            
            assert isinstance(embedding, list)
            assert len(embedding) == 3
    
    def test_extract_entities(self):
        """Test entity extraction."""
        with patch('src.query_processor.spacy.load') as mock_spacy:
            mock_doc = Mock()
            mock_ent = Mock()
            mock_ent.text = "Python"
            mock_ent.label_ = "ORG"
            mock_ent.start_char = 0
            mock_ent.end_char = 6
            mock_doc.ents = [mock_ent]
            
            mock_nlp = Mock()
            mock_nlp.return_value = mock_doc
            mock_spacy.return_value = mock_nlp
            
            processor = QueryProcessor()
            entities = processor.extract_entities("Python is great")
            
            assert len(entities) == 1
            assert entities[0]["text"] == "Python"
            assert entities[0]["label"] == "ORG"


if __name__ == "__main__":
    pytest.main([__file__])
