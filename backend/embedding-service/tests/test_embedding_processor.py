"""
Tests for Embedding Processor
"""

import json
from unittest.mock import Mock, patch

import numpy as np
import pytest
from src.embedding_processor import EmbeddingProcessor


class TestEmbeddingProcessor:
    """Test EmbeddingProcessor class."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock embedding processor for testing."""
        with (
            patch("src.embedding_processor.SentenceTransformer") as mock_model,
            patch("src.embedding_processor.QdrantClient") as mock_qdrant,
        ):
            # Mock the model
            mock_model_instance = Mock()
            mock_model_instance.encode.return_value = np.array([0.1, 0.2, 0.3])
            mock_model_instance.get_sentence_embedding_dimension.return_value = 384
            mock_model.return_value = mock_model_instance

            # Mock Qdrant client
            mock_qdrant_instance = Mock()
            mock_qdrant_instance.get_collections.return_value = Mock(collections=[])
            mock_qdrant_instance.create_collection.return_value = None
            mock_qdrant.return_value = mock_qdrant_instance

            processor = EmbeddingProcessor(
                model_name="test-model",
                qdrant_host="localhost",
                qdrant_port=6333,
                collection_name="test_collection",
            )

            processor.model = mock_model_instance
            processor.qdrant_client = mock_qdrant_instance

            return processor

    def test_generate_embedding(self, mock_processor):
        """Test single embedding generation."""
        text = "This is a test text"
        expected_embedding = [0.1, 0.2, 0.3]

        result = mock_processor.generate_embedding(text)

        assert result == expected_embedding
        mock_processor.model.encode.assert_called_once_with(
            text, convert_to_tensor=False
        )

    def test_generate_batch_embeddings(self, mock_processor):
        """Test batch embedding generation."""
        texts = ["Text 1", "Text 2", "Text 3"]
        expected_embeddings = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]

        # Mock batch encoding
        mock_processor.model.encode.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        )

        result = mock_processor.generate_batch_embeddings(texts)

        assert len(result) == 3
        assert result == expected_embeddings
        mock_processor.model.encode.assert_called_once_with(
            texts, convert_to_tensor=False
        )

    def test_store_embedding(self, mock_processor):
        """Test storing embedding in Qdrant."""
        text = "Test text"
        embedding = [0.1, 0.2, 0.3]
        document_id = "test_doc_123"
        metadata = {"source": "test"}

        result = mock_processor.store_embedding(
            text=text, embedding=embedding, document_id=document_id, metadata=metadata
        )

        assert result == document_id
        mock_processor.qdrant_client.upsert.assert_called_once()

        # Verify the call arguments
        call_args = mock_processor.qdrant_client.upsert.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        assert len(call_args[1]["points"]) == 1

        point = call_args[1]["points"][0]
        assert point.id == document_id
        assert point.vector == embedding
        assert point.payload["text"] == text
        assert point.payload["source"] == "test"

    def test_search_similar(self, mock_processor):
        """Test similarity search."""
        query = "test query"
        mock_results = [
            Mock(
                id="doc1",
                score=0.95,
                payload={"text": "Similar text", "source": "test"},
            ),
            Mock(
                id="doc2",
                score=0.87,
                payload={"text": "Another text", "source": "test"},
            ),
        ]

        mock_processor.qdrant_client.search.return_value = mock_results

        result = mock_processor.search_similar(query, limit=5)

        assert len(result) == 2
        assert result[0]["id"] == "doc1"
        assert result[0]["score"] == 0.95
        assert result[0]["text"] == "Similar text"
        assert result[0]["metadata"]["source"] == "test"

        # Verify search was called correctly
        mock_processor.qdrant_client.search.assert_called_once()
        call_args = mock_processor.qdrant_client.search.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        assert call_args[1]["limit"] == 5

    def test_search_similar_with_filters(self, mock_processor):
        """Test similarity search with filters."""
        query = "test query"
        filter_conditions = {"category": "test"}

        mock_processor.qdrant_client.search.return_value = []

        mock_processor.search_similar(
            query=query,
            limit=10,
            score_threshold=0.8,
            filter_conditions=filter_conditions,
        )

        # Verify search was called with filters
        call_args = mock_processor.qdrant_client.search.call_args
        assert call_args[1]["score_threshold"] == 0.8
        assert "query_filter" in call_args[1]

    def test_process_text_file_txt(self, mock_processor):
        """Test processing TXT file."""
        file_content = b"This is a test text file"
        filename = "test.txt"

        # Mock the embedding generation and storage
        mock_processor.generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_processor.store_embedding.return_value = "test_point_id"

        result = mock_processor.process_text_file(
            file_content=file_content, filename=filename, document_id="test_doc"
        )

        assert result["text"] == "This is a test text file"
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["filename"] == filename
        assert result["method"] == "embed_text_txt"
        assert result["point_id"] == "test_point_id"

    def test_process_text_file_json(self, mock_processor):
        """Test processing JSON file."""
        json_data = {"text": "This is JSON content", "metadata": {"source": "test"}}
        file_content = json.dumps(json_data).encode("utf-8")
        filename = "test.json"

        mock_processor.generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_processor.store_embedding.return_value = "test_point_id"

        result = mock_processor.process_text_file(
            file_content=file_content, filename=filename
        )

        assert result["text"] == "This is JSON content"
        assert result["method"] == "embed_text_json"

    def test_process_text_file_csv(self, mock_processor):
        """Test processing CSV file."""
        csv_content = b"name,description\nItem1,Description1\nItem2,Description2"
        filename = "test.csv"

        mock_processor.generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_processor.store_embedding.return_value = "test_point_id"

        result = mock_processor.process_text_file(
            file_content=csv_content, filename=filename
        )

        assert "Item1" in result["text"]
        assert "Item2" in result["text"]
        assert result["method"] == "embed_text_csv"

    def test_process_text_file_unsupported_format(self, mock_processor):
        """Test processing unsupported file format."""
        file_content = b"Some binary content"
        filename = "test.exe"

        with pytest.raises(ValueError, match="Unsupported text file format"):
            mock_processor.process_text_file(
                file_content=file_content, filename=filename
            )

    def test_get_collection_info(self, mock_processor):
        """Test getting collection information."""
        mock_collection_info = Mock()
        mock_collection_info.config.params.vectors.size = 384
        mock_collection_info.points_count = 1000
        mock_collection_info.status = "green"

        mock_processor.qdrant_client.get_collection.return_value = mock_collection_info

        result = mock_processor.get_collection_info()

        assert result["name"] == "test_collection"
        assert result["vector_size"] == 384
        assert result["points_count"] == 1000
        assert result["status"] == "green"

    def test_delete_embedding(self, mock_processor):
        """Test deleting an embedding."""
        point_id = "test_point_id"

        result = mock_processor.delete_embedding(point_id)

        assert result is True
        mock_processor.qdrant_client.delete.assert_called_once_with(
            collection_name="test_collection", points_selector=[point_id]
        )

    def test_delete_embedding_error(self, mock_processor):
        """Test deleting an embedding with error."""
        point_id = "nonexistent_point"
        mock_processor.qdrant_client.delete.side_effect = Exception("Point not found")

        result = mock_processor.delete_embedding(point_id)

        assert result is False

    def test_health_check(self, mock_processor):
        """Test health check functionality."""
        # Mock successful health checks
        mock_processor.model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_processor.qdrant_client.get_collections.return_value = Mock(
            collections=[Mock(name="test_collection")]
        )

        result = mock_processor.health_check()

        assert result["embedding_model_loaded"] is True
        assert result["qdrant_available"] is True
        assert result["collection_exists"] is True

    def test_health_check_model_failure(self, mock_processor):
        """Test health check with model failure."""
        mock_processor.model.encode.side_effect = Exception("Model error")
        mock_processor.qdrant_client.get_collections.return_value = Mock(collections=[])

        result = mock_processor.health_check()

        assert result["embedding_model_loaded"] is False
        assert result["qdrant_available"] is True
        assert result["collection_exists"] is False

    def test_health_check_qdrant_failure(self, mock_processor):
        """Test health check with Qdrant failure."""
        mock_processor.model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_processor.qdrant_client.get_collections.side_effect = Exception(
            "Qdrant error"
        )

        result = mock_processor.health_check()

        assert result["embedding_model_loaded"] is True
        assert result["qdrant_available"] is False
        assert result["collection_exists"] is False

    def test_extract_text_from_json_dict(self, mock_processor):
        """Test extracting text from JSON dict."""
        json_data = {"text": "This is the content", "other": "ignored"}
        result = mock_processor._extract_text_from_json(json_data)
        assert result == "This is the content"

    def test_extract_text_from_json_list(self, mock_processor):
        """Test extracting text from JSON list."""
        json_data = ["Item 1", "Item 2", "Item 3"]
        result = mock_processor._extract_text_from_json(json_data)
        assert result == "Item 1 Item 2 Item 3"

    def test_extract_text_from_json_string(self, mock_processor):
        """Test extracting text from JSON string."""
        json_data = "Simple string content"
        result = mock_processor._extract_text_from_json(json_data)
        assert result == "Simple string content"

    def test_extract_text_from_csv(self, mock_processor):
        """Test extracting text from CSV."""
        csv_content = b"name,description\nItem1,Description1\nItem2,Description2"
        result = mock_processor._extract_text_from_csv(csv_content)

        assert "Item1" in result
        assert "Item2" in result
        assert "Description1" in result
        assert "Description2" in result
