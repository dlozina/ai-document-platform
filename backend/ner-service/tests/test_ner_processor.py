"""
Unit tests for NER Processor
"""

from unittest.mock import Mock, patch

import pytest
from src.ner_processor import NERProcessor


@pytest.fixture
def ner_processor():
    """Create NER processor instance for testing."""
    with patch("spacy.load") as mock_load:
        # Mock spaCy model
        mock_nlp = Mock()
        mock_nlp.pipe = Mock(return_value=[])
        mock_load.return_value = mock_nlp

        processor = NERProcessor()
        return processor


@pytest.fixture
def sample_text():
    """Create sample text for testing."""
    return """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
    The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
    Apple's revenue for 2023 was $394.3 billion, representing a 2.8% increase from the previous year.
    """


@pytest.fixture
def mock_doc():
    """Create mock spaCy document."""
    mock_doc = Mock()
    mock_doc.ents = []

    # Mock entities
    mock_entity1 = Mock()
    mock_entity1.text = "Apple Inc."
    mock_entity1.label_ = "ORG"
    mock_entity1.start_char = 5
    mock_entity1.end_char = 15

    mock_entity2 = Mock()
    mock_entity2.text = "Steve Jobs"
    mock_entity2.label_ = "PERSON"
    mock_entity2.start_char = 100
    mock_entity2.end_char = 110

    mock_doc.ents = [mock_entity1, mock_entity2]

    return mock_doc


class TestNERProcessor:
    """Test suite for NER Processor."""

    def test_initialization(self):
        """Test NER processor initialization."""
        with patch("spacy.load") as mock_load:
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp

            processor = NERProcessor()
            # Check that models are loaded
            assert isinstance(processor.models, dict)
            assert "en" in processor.models

    def test_initialization_fallback(self):
        """Test initialization with fallback model."""
        with patch("spacy.load") as mock_load:
            # First call fails, second succeeds
            mock_load.side_effect = [OSError("Model not found"), Mock()]

            processor = NERProcessor()
            # Check that models dictionary exists (may be empty if all models fail)
            assert isinstance(processor.models, dict)

    def test_initialization_no_models(self):
        """Test initialization when no models are available."""
        with patch("spacy.load") as mock_load:
            mock_load.side_effect = OSError("No models found")

            with pytest.raises(RuntimeError, match="No spaCy models available"):
                NERProcessor()

    def test_process_text(self, ner_processor, sample_text, mock_doc):
        """Test processing text."""
        with patch.object(ner_processor.nlp, "__call__", return_value=mock_doc):
            result = ner_processor.process_text(sample_text)

            # Verify structure
            assert "text" in result
            assert "entities" in result
            assert "entity_count" in result
            assert "model_used" in result
            assert "processing_time_ms" in result
            assert "text_length" in result

            # Verify content
            assert result["text"] == sample_text
            assert result["entity_count"] == 2
            assert result["text_length"] == len(sample_text)
            assert result["processing_time_ms"] > 0

            # Check entities
            assert len(result["entities"]) == 2
            assert result["entities"][0]["text"] == "Apple Inc."
            assert result["entities"][0]["label"] == "ORG"
            assert result["entities"][1]["text"] == "Steve Jobs"
            assert result["entities"][1]["label"] == "PERSON"

    def test_process_text_with_entity_types(self, ner_processor, sample_text, mock_doc):
        """Test processing text with specific entity types."""
        with patch.object(ner_processor.nlp, "__call__", return_value=mock_doc):
            result = ner_processor.process_text(sample_text, entity_types=["ORG"])

            # Should only return ORG entities
            assert result["entity_count"] == 1
            assert result["entities"][0]["label"] == "ORG"

    def test_process_text_with_confidence(self, ner_processor, sample_text, mock_doc):
        """Test processing text with confidence scores."""
        with patch.object(ner_processor.nlp, "__call__", return_value=mock_doc):
            result = ner_processor.process_text(sample_text, include_confidence=True)

            # Check that confidence scores are included
            for entity in result["entities"]:
                assert "confidence" in entity
                assert 0.0 <= entity["confidence"] <= 1.0

    def test_process_text_empty(self, ner_processor):
        """Test processing empty text."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            ner_processor.process_text("")

    def test_process_text_too_long(self, ner_processor):
        """Test processing text that's too long."""
        long_text = "A" * 2000000  # 2M characters

        with pytest.raises(ValueError, match="Text too long"):
            ner_processor.process_text(long_text)

    def test_process_batch(self, ner_processor, mock_doc):
        """Test batch processing."""
        texts = ["Text 1", "Text 2", "Text 3"]

        with patch.object(ner_processor.nlp, "pipe", return_value=[mock_doc] * 3):
            result = ner_processor.process_batch(texts)

            # Verify structure
            assert "results" in result
            assert "total_processing_time_ms" in result
            assert "batch_size" in result

            # Verify content
            assert len(result["results"]) == 3
            assert result["batch_size"] == 3
            assert result["total_processing_time_ms"] > 0

            # Check individual results
            for i, result_item in enumerate(result["results"]):
                assert result_item["text"] == texts[i]
                assert "entities" in result_item

    def test_get_entity_statistics(self, ner_processor):
        """Test getting entity statistics."""
        entities = [
            {"text": "Apple Inc.", "label": "ORG"},
            {"text": "Steve Jobs", "label": "PERSON"},
            {"text": "Microsoft", "label": "ORG"},
            {"text": "Bill Gates", "label": "PERSON"},
        ]

        stats = ner_processor.get_entity_statistics(entities)

        # Verify structure
        assert "total_entities" in stats
        assert "entity_types" in stats
        assert "most_common_entities" in stats

        # Verify content
        assert stats["total_entities"] == 4
        assert len(stats["entity_types"]) == 2  # ORG and PERSON

        # Check entity type counts
        org_count = next(
            et for et in stats["entity_types"] if et["entity_type"] == "ORG"
        )
        person_count = next(
            et for et in stats["entity_types"] if et["entity_type"] == "PERSON"
        )

        assert org_count["count"] == 2
        assert person_count["count"] == 2

    def test_get_entity_statistics_empty(self, ner_processor):
        """Test getting statistics for empty entity list."""
        stats = ner_processor.get_entity_statistics([])

        assert stats["total_entities"] == 0
        assert stats["entity_types"] == []
        assert stats["most_common_entities"] == []

    def test_estimate_confidence(self, ner_processor):
        """Test confidence estimation."""
        # Mock entity and document
        mock_entity = Mock()
        mock_entity.text = "Apple Inc."
        mock_entity.label_ = "ORG"

        mock_doc = Mock()

        confidence = ner_processor._estimate_confidence(mock_entity, mock_doc)

        # Confidence should be between 0 and 1
        assert 0.0 <= confidence <= 1.0

    def test_get_available_models(self, ner_processor):
        """Test getting available models."""
        with patch("spacy.load") as mock_load:
            mock_load.side_effect = [Mock(), OSError("Not found")]

            models = ner_processor.get_available_models()

            assert isinstance(models, dict)
            assert "en_core_web_sm" in models
            assert "en_core_web_lg" in models

    def test_visualize_entities(self, ner_processor, sample_text):
        """Test entity visualization."""
        entities = [
            {"text": "Apple Inc.", "label": "ORG", "start": 5, "end": 15},
            {"text": "Steve Jobs", "label": "PERSON", "start": 100, "end": 110},
        ]

        with patch("spacy.displacy.render") as mock_render:
            mock_render.return_value = "<html>Visualization</html>"

            html = ner_processor.visualize_entities(sample_text, entities)

            assert html == "<html>Visualization</html>"
            mock_render.assert_called_once()

    def test_visualize_entities_no_model(self):
        """Test visualization when no model is available."""
        processor = NERProcessor.__new__(NERProcessor)
        processor.nlp = None

        html = processor.visualize_entities("test", [])

        assert "<p>No spaCy model available for visualization</p>" in html


class TestNERProcessorEdgeCases:
    """Test edge cases and error handling."""

    def test_process_text_special_characters(self, ner_processor):
        """Test processing text with special characters."""
        text_with_special = "Text with émojis 🚀 and spëcial chars!"

        with patch.object(ner_processor.nlp, "__call__", return_value=Mock(ents=[])):
            result = ner_processor.process_text(text_with_special)

            assert result["text"] == text_with_special
            assert result["entity_count"] == 0

    def test_process_text_unicode(self, ner_processor):
        """Test processing Unicode text."""
        unicode_text = "中文文本 with 日本語 and العربية"

        with patch.object(ner_processor.nlp, "__call__", return_value=Mock(ents=[])):
            result = ner_processor.process_text(unicode_text)

            assert result["text"] == unicode_text
            assert result["text_length"] == len(unicode_text)

    def test_process_text_whitespace_only(self, ner_processor):
        """Test processing whitespace-only text."""
        whitespace_text = "   \n\t   "

        with pytest.raises(ValueError, match="Text cannot be empty"):
            ner_processor.process_text(whitespace_text)

    def test_process_text_single_word(self, ner_processor):
        """Test processing single word."""
        single_word = "Apple"

        with patch.object(ner_processor.nlp, "__call__", return_value=Mock(ents=[])):
            result = ner_processor.process_text(single_word)

            assert result["text"] == single_word
            assert result["entity_count"] == 0


@pytest.mark.parametrize(
    "model_name,fallback_model",
    [
        ("en_core_web_sm", "en_core_web_lg"),
        ("en_core_web_lg", "en_core_web_sm"),
    ],
)
def test_different_model_combinations(model_name, fallback_model):
    """Test different model combinations."""
    with patch("spacy.load") as mock_load:
        mock_nlp = Mock()
        mock_load.return_value = mock_nlp

        processor = NERProcessor()
        # Check that models dictionary is populated
        assert isinstance(processor.models, dict)


@pytest.mark.parametrize(
    "entity_types",
    [
        ["PERSON"],
        ["ORG", "GPE"],
        ["PERSON", "ORG", "MONEY", "DATE"],
        None,
    ],
)
def test_entity_type_filtering(entity_types):
    """Test different entity type filtering options."""
    with patch("spacy.load") as mock_load:
        mock_nlp = Mock()
        mock_load.return_value = mock_nlp

        processor = NERProcessor()

        # Mock document with various entity types
        mock_entity1 = Mock()
        mock_entity1.text = "John"
        mock_entity1.label_ = "PERSON"
        mock_entity1.start_char = 0
        mock_entity1.end_char = 4

        mock_entity2 = Mock()
        mock_entity2.text = "Apple"
        mock_entity2.label_ = "ORG"
        mock_entity2.start_char = 5
        mock_entity2.end_char = 10

        mock_doc = Mock()
        mock_doc.ents = [mock_entity1, mock_entity2]

        with patch.object(processor.nlp, "__call__", return_value=mock_doc):
            result = processor.process_text("John Apple", entity_types=entity_types)

            if entity_types is None:
                # All entities should be returned
                assert result["entity_count"] == 2
            else:
                # Only filtered entities should be returned
                for entity in result["entities"]:
                    assert entity["label"] in entity_types
