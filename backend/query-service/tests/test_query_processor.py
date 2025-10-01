"""
Tests for Query Processor
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.query_processor import QueryProcessor
from src.models import QueryFilter, DateRange


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch('src.query_processor.get_settings') as mock:
        settings = Mock()
        settings.embedding_model = "test-model"
        settings.qdrant_host = "localhost"
        settings.qdrant_port = 6333
        settings.qdrant_collection_name = "test-collection"
        settings.enable_reranking = True
        settings.rerank_model = "test-rerank-model"
        settings.max_context_length = 4000
        mock.return_value = settings
        yield settings


@pytest.fixture
def query_processor(mock_settings):
    """Create QueryProcessor instance with mocked dependencies."""
    with patch('src.query_processor.SentenceTransformer'), \
         patch('src.query_processor.QdrantClient'), \
         patch('src.query_processor.create_engine'), \
         patch('src.query_processor.spacy.load'), \
         patch('src.query_processor.CrossEncoder'):
        
        processor = QueryProcessor()
        return processor


class TestQueryProcessorInitialization:
    """Test QueryProcessor initialization."""
    
    def test_initialization_success(self, mock_settings):
        """Test successful initialization."""
        with patch('src.query_processor.SentenceTransformer') as mock_st, \
             patch('src.query_processor.QdrantClient') as mock_qdrant, \
             patch('src.query_processor.create_engine') as mock_engine, \
             patch('src.query_processor.spacy.load') as mock_spacy, \
             patch('src.query_processor.CrossEncoder') as mock_cross:
            
            processor = QueryProcessor()
            
            # Verify all components are initialized
            assert processor.embedding_model is not None
            assert processor.qdrant_client is not None
            assert processor.db_engine is not None
            assert processor.nlp is not None
            assert processor.rerank_model is not None
    
    def test_initialization_without_spacy(self, mock_settings):
        """Test initialization when spaCy fails to load."""
        with patch('src.query_processor.SentenceTransformer'), \
             patch('src.query_processor.QdrantClient'), \
             patch('src.query_processor.create_engine'), \
             patch('src.query_processor.spacy.load', side_effect=Exception("spaCy not available")), \
             patch('src.query_processor.CrossEncoder'):
            
            processor = QueryProcessor()
            assert processor.nlp is None
    
    def test_initialization_without_rerank(self, mock_settings):
        """Test initialization when rerank model fails to load."""
        mock_settings.enable_reranking = True
        
        with patch('src.query_processor.SentenceTransformer'), \
             patch('src.query_processor.QdrantClient'), \
             patch('src.query_processor.create_engine'), \
             patch('src.query_processor.spacy.load'), \
             patch('src.query_processor.CrossEncoder', side_effect=Exception("Rerank model not available")):
            
            processor = QueryProcessor()
            assert processor.rerank_model is None


class TestEmbeddingGeneration:
    """Test embedding generation."""
    
    def test_generate_query_embedding_success(self, query_processor):
        """Test successful query embedding generation."""
        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        query_processor.embedding_model.encode.return_value = mock_embedding
        
        embedding = query_processor.generate_query_embedding("test query")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 4
        assert embedding == [0.1, 0.2, 0.3, 0.4]
        query_processor.embedding_model.encode.assert_called_once_with("test query", convert_to_tensor=False)
    
    def test_generate_query_embedding_failure(self, query_processor):
        """Test query embedding generation failure."""
        query_processor.embedding_model.encode.side_effect = Exception("Model error")
        
        with pytest.raises(Exception, match="Model error"):
            query_processor.generate_query_embedding("test query")


class TestQdrantFilterBuilding:
    """Test Qdrant filter building."""
    
    def test_build_qdrant_filter_with_tenant(self, query_processor):
        """Test building Qdrant filter with tenant ID."""
        filter_params = {"content_type": "application/pdf"}
        tenant_id = "test-tenant"
        
        filter_obj = query_processor._build_qdrant_filter(filter_params, tenant_id)
        
        assert filter_obj is not None
        assert len(filter_obj.must) == 2  # tenant_id + content_type
    
    def test_build_qdrant_filter_without_params(self, query_processor):
        """Test building Qdrant filter without parameters."""
        filter_obj = query_processor._build_qdrant_filter(None, None)
        assert filter_obj is None
    
    def test_build_qdrant_filter_with_date_range(self, query_processor):
        """Test building Qdrant filter with date range."""
        date_range = DateRange(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 12, 31)
        )
        filter_params = {"date_range": date_range}
        
        filter_obj = query_processor._build_qdrant_filter(filter_params, None)
        
        assert filter_obj is not None
        assert len(filter_obj.must) == 2  # start and end date


class TestSemanticSearch:
    """Test semantic search functionality."""
    
    def test_semantic_search_success(self, query_processor):
        """Test successful semantic search."""
        # Mock Qdrant search results
        mock_result = Mock()
        mock_result.id = "test-id"
        mock_result.score = 0.95
        mock_result.payload = {
            "text": "Test document content",
            "document_id": "doc-1",
            "filename": "test.pdf",
            "tenant_id": "test-tenant"
        }
        
        query_processor.qdrant_client.search.return_value = [mock_result]
        
        results = query_processor.semantic_search(
            query="test query",
            top_k=10,
            score_threshold=0.7,
            tenant_id="test-tenant"
        )
        
        assert len(results) == 1
        assert results[0]["id"] == "test-id"
        assert results[0]["score"] == 0.95
        assert results[0]["text"] == "Test document content"
    
    def test_semantic_search_with_filter(self, query_processor):
        """Test semantic search with filter."""
        filter_params = {
            "content_type": "application/pdf",
            "file_type": "pdf"
        }
        
        query_processor.qdrant_client.search.return_value = []
        
        results = query_processor.semantic_search(
            query="test query",
            filter_params=filter_params,
            tenant_id="test-tenant"
        )
        
        # Verify search was called with filter
        query_processor.qdrant_client.search.assert_called_once()
        call_args = query_processor.qdrant_client.search.call_args
        assert call_args[1]["query_filter"] is not None


class TestReranking:
    """Test result reranking."""
    
    def test_rerank_results_success(self, query_processor):
        """Test successful result reranking."""
        query = "test query"
        results = [
            {"text": "First document", "score": 0.8},
            {"text": "Second document", "score": 0.9}
        ]
        
        # Mock rerank scores
        query_processor.rerank_model.predict.return_value = [0.9, 0.8]
        
        reranked = query_processor.rerank_results(query, results)
        
        assert len(reranked) == 2
        assert reranked[0]["score"] == 0.9  # Should be reordered
        assert reranked[1]["score"] == 0.8
        assert "rerank_score" in reranked[0]
        assert "original_score" in reranked[0]
    
    def test_rerank_results_no_model(self, query_processor):
        """Test reranking when no rerank model is available."""
        query_processor.rerank_model = None
        
        results = [{"text": "test", "score": 0.8}]
        reranked = query_processor.rerank_results("query", results)
        
        assert reranked == results  # Should return original results
    
    def test_rerank_results_empty(self, query_processor):
        """Test reranking with empty results."""
        reranked = query_processor.rerank_results("query", [])
        assert reranked == []


class TestEntityExtraction:
    """Test entity extraction functionality."""
    
    def test_extract_entities_success(self, query_processor):
        """Test successful entity extraction."""
        # Mock spaCy document
        mock_ent = Mock()
        mock_ent.text = "Python"
        mock_ent.label_ = "ORG"
        mock_ent.start_char = 0
        mock_ent.end_char = 6
        
        mock_doc = Mock()
        mock_doc.ents = [mock_ent]
        
        query_processor.nlp.return_value = mock_doc
        
        entities = query_processor.extract_entities("Python is great")
        
        assert len(entities) == 1
        assert entities[0]["text"] == "Python"
        assert entities[0]["label"] == "ORG"
        assert entities[0]["confidence"] == 0.8
    
    def test_extract_entities_no_nlp(self, query_processor):
        """Test entity extraction when NLP model is not available."""
        query_processor.nlp = None
        
        entities = query_processor.extract_entities("test text")
        assert entities == []
    
    def test_extract_entities_failure(self, query_processor):
        """Test entity extraction failure."""
        query_processor.nlp.side_effect = Exception("NLP error")
        
        entities = query_processor.extract_entities("test text")
        assert entities == []


class TestEntityFiltering:
    """Test entity-based filtering."""
    
    def test_filter_by_entities_with_labels(self, query_processor):
        """Test filtering by entity labels."""
        results = [
            {
                "text": "Document 1",
                "metadata": {
                    "ner_entities": [
                        {"text": "Python", "label": "ORG"},
                        {"text": "John", "label": "PERSON"}
                    ]
                }
            },
            {
                "text": "Document 2",
                "metadata": {
                    "ner_entities": [
                        {"text": "Java", "label": "ORG"}
                    ]
                }
            }
        ]
        
        filtered = query_processor.filter_by_entities(
            results,
            entity_labels=["PERSON"]
        )
        
        assert len(filtered) == 1
        assert filtered[0]["text"] == "Document 1"
    
    def test_filter_by_entities_with_text(self, query_processor):
        """Test filtering by entity text."""
        results = [
            {
                "text": "Document 1",
                "metadata": {
                    "ner_entities": [
                        {"text": "Python", "label": "ORG"}
                    ]
                }
            }
        ]
        
        filtered = query_processor.filter_by_entities(
            results,
            entity_text=["Python"]
        )
        
        assert len(filtered) == 1
    
    def test_filter_by_entities_no_entities(self, query_processor):
        """Test filtering when documents have no entities."""
        results = [
            {
                "text": "Document 1",
                "metadata": {}
            }
        ]
        
        filtered = query_processor.filter_by_entities(
            results,
            entity_labels=["PERSON"]
        )
        
        assert len(filtered) == 0


class TestAnswerExtraction:
    """Test answer span extraction."""
    
    def test_extract_answer_span_success(self, query_processor):
        """Test successful answer span extraction."""
        question = "What is Python?"
        document_text = "Python is a programming language. It is widely used for web development."
        
        answer, confidence = query_processor.extract_answer_span(question, document_text)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_extract_answer_span_no_keywords(self, query_processor):
        """Test answer extraction when no keywords match."""
        question = "What is XYZ?"
        document_text = "This document is about Python programming and web development."
        
        answer, confidence = query_processor.extract_answer_span(question, document_text)
        
        assert isinstance(answer, str)
        assert confidence < 0.5  # Should have lower confidence


class TestRAGGeneration:
    """Test RAG answer generation."""
    
    def test_generate_rag_answer_success(self, query_processor):
        """Test successful RAG answer generation."""
        question = "What is Python?"
        context_documents = [
            {
                "text": "Python is a programming language",
                "filename": "doc1.pdf"
            },
            {
                "text": "Python is used for web development",
                "filename": "doc2.pdf"
            }
        ]
        
        answer, confidence = query_processor.generate_rag_answer(
            question, context_documents
        )
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert "Python" in answer
    
    def test_generate_rag_answer_empty_context(self, query_processor):
        """Test RAG with empty context."""
        answer, confidence = query_processor.generate_rag_answer("test", [])
        
        assert isinstance(answer, str)
        assert confidence < 0.5


class TestHealthCheck:
    """Test health check functionality."""
    
    def test_health_check_all_healthy(self, query_processor):
        """Test health check when all components are healthy."""
        query_processor.qdrant_client.get_collections.return_value = []
        
        with patch('src.query_processor.text') as mock_text:
            mock_conn = Mock()
            query_processor.db_engine.connect.return_value.__enter__.return_value = mock_conn
            
            health = query_processor.health_check()
            
            assert health["embedding_model_loaded"] is True
            assert health["qdrant_available"] is True
            assert health["database_connected"] is True
    
    def test_health_check_qdrant_unavailable(self, query_processor):
        """Test health check when Qdrant is unavailable."""
        query_processor.qdrant_client.get_collections.side_effect = Exception("Connection failed")
        
        health = query_processor.health_check()
        
        assert health["qdrant_available"] is False
    
    def test_health_check_database_unavailable(self, query_processor):
        """Test health check when database is unavailable."""
        query_processor.db_engine.connect.side_effect = Exception("Connection failed")
        
        health = query_processor.health_check()
        
        assert health["database_connected"] is False


if __name__ == "__main__":
    pytest.main([__file__])
