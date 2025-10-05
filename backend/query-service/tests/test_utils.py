"""
Tests for Query Service utilities
"""

from datetime import datetime

import pytest
from src.utils import (
    calculate_confidence_score,
    calculate_text_hash,
    create_error_response,
    extract_keywords,
    format_search_result,
    format_timestamp,
    log_query_metrics,
    parse_date_range,
    sanitize_query_text,
    truncate_text,
    validate_query_params,
)


class TestTextHash:
    """Test text hash calculation."""

    def test_calculate_text_hash(self):
        """Test hash calculation."""
        text = "Hello, world!"
        hash1 = calculate_text_hash(text)
        hash2 = calculate_text_hash(text)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64-character hex string

    def test_calculate_text_hash_different_texts(self):
        """Test hash calculation with different texts."""
        hash1 = calculate_text_hash("Hello")
        hash2 = calculate_text_hash("World")

        assert hash1 != hash2


class TestTimestampFormatting:
    """Test timestamp formatting."""

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        timestamp = datetime(2024, 1, 15, 10, 30, 45)
        formatted = format_timestamp(timestamp)

        assert isinstance(formatted, str)
        assert "2024-01-15T10:30:45" in formatted


class TestTextTruncation:
    """Test text truncation."""

    def test_truncate_text_short(self):
        """Test truncation of short text."""
        text = "Short text"
        truncated = truncate_text(text, max_length=20)

        assert truncated == text

    def test_truncate_text_long(self):
        """Test truncation of long text."""
        text = "This is a very long text that should be truncated"
        truncated = truncate_text(text, max_length=20)

        assert len(truncated) == 20
        assert truncated.endswith("...")
        assert truncated.startswith("This is a very")

    def test_truncate_text_custom_suffix(self):
        """Test truncation with custom suffix."""
        text = "Long text"
        truncated = truncate_text(text, max_length=5, suffix="---")

        assert len(truncated) == 5
        assert truncated.endswith("---")


class TestKeywordExtraction:
    """Test keyword extraction."""

    def test_extract_keywords(self):
        """Test keyword extraction."""
        text = "Python is a programming language. Python is widely used for web development."
        keywords = extract_keywords(text, max_keywords=5)

        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        assert "Python" in keywords
        assert "programming" in keywords

    def test_extract_keywords_empty_text(self):
        """Test keyword extraction with empty text."""
        keywords = extract_keywords("")
        assert keywords == []

    def test_extract_keywords_short_text(self):
        """Test keyword extraction with short text."""
        keywords = extract_keywords("Hi there")
        assert isinstance(keywords, list)


class TestQueryParamsValidation:
    """Test query parameter validation."""

    def test_validate_query_params_valid(self):
        """Test validation of valid parameters."""
        params = {"top_k": 10, "score_threshold": 0.7, "tenant_id": "test-tenant"}

        validated = validate_query_params(params)

        assert validated["top_k"] == 10
        assert validated["score_threshold"] == 0.7
        assert validated["tenant_id"] == "test-tenant"

    def test_validate_query_params_invalid_top_k(self):
        """Test validation with invalid top_k."""
        params = {"top_k": 150, "score_threshold": 0.7}  # Too high

        validated = validate_query_params(params)

        assert validated["top_k"] == 10  # Should default to 10

    def test_validate_query_params_invalid_threshold(self):
        """Test validation with invalid score threshold."""
        params = {"top_k": 10, "score_threshold": 1.5}  # Too high

        validated = validate_query_params(params)

        assert "score_threshold" not in validated  # Should be filtered out


class TestSearchResultFormatting:
    """Test search result formatting."""

    def test_format_search_result(self):
        """Test search result formatting."""
        result = {
            "document_id": "doc-1",
            "filename": "test.pdf",
            "score": 0.95,
            "text": "This is a test document with some content",
            "metadata": {"tenant_id": "test"},
        }

        formatted = format_search_result(result)

        assert formatted["document_id"] == "doc-1"
        assert formatted["filename"] == "test.pdf"
        assert formatted["relevance_score"] == 0.95
        assert "quoted_text" in formatted
        assert "metadata" in formatted

    def test_format_search_result_long_text(self):
        """Test formatting with long text."""
        long_text = "A" * 1000
        result = {
            "document_id": "doc-1",
            "filename": "test.pdf",
            "score": 0.95,
            "text": long_text,
            "metadata": {},
        }

        formatted = format_search_result(result, max_text_length=100)

        assert len(formatted["quoted_text"]) <= 100
        assert formatted["quoted_text"].endswith("...")


class TestConfidenceScoreCalculation:
    """Test confidence score calculation."""

    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        similarity_scores = [0.8, 0.9, 0.7]
        confidence = calculate_confidence_score(
            similarity_scores, entity_matches=2, keyword_matches=3
        )

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.8  # Should be high with good scores and matches

    def test_calculate_confidence_score_empty(self):
        """Test confidence score with empty scores."""
        confidence = calculate_confidence_score([])
        assert confidence == 0.0

    def test_calculate_confidence_score_high_matches(self):
        """Test confidence score with many matches."""
        similarity_scores = [0.5, 0.6]
        confidence = calculate_confidence_score(
            similarity_scores,
            entity_matches=10,
            keyword_matches=5,  # Many matches
        )

        assert confidence <= 1.0  # Should be capped at 1.0


class TestDateRangeParsing:
    """Test date range parsing."""

    def test_parse_date_range_valid(self):
        """Test parsing valid date range."""
        date_range = {"start": "2024-01-01T00:00:00Z", "end": "2024-12-31T23:59:59Z"}

        parsed = parse_date_range(date_range)

        assert parsed is not None
        assert "start" in parsed
        assert "end" in parsed
        assert isinstance(parsed["start"], datetime)
        assert isinstance(parsed["end"], datetime)

    def test_parse_date_range_none(self):
        """Test parsing None date range."""
        parsed = parse_date_range(None)
        assert parsed is None

    def test_parse_date_range_invalid(self):
        """Test parsing invalid date range."""
        date_range = {"start": "invalid-date", "end": "2024-12-31T23:59:59Z"}

        parsed = parse_date_range(date_range)

        # Should only have valid end date
        assert parsed is not None
        assert "start" not in parsed
        assert "end" in parsed


class TestQueryTextSanitization:
    """Test query text sanitization."""

    def test_sanitize_query_text_normal(self):
        """Test sanitization of normal text."""
        query = "What is Python programming?"
        sanitized = sanitize_query_text(query)

        assert sanitized == query

    def test_sanitize_query_text_whitespace(self):
        """Test sanitization with excessive whitespace."""
        query = "  What   is    Python?  "
        sanitized = sanitize_query_text(query)

        assert sanitized == "What is Python?"

    def test_sanitize_query_text_long(self):
        """Test sanitization of very long text."""
        query = "A" * 15000
        sanitized = sanitize_query_text(query)

        assert len(sanitized) <= 10003  # 10000 + "..."
        assert sanitized.endswith("...")

    def test_sanitize_query_text_empty(self):
        """Test sanitization of empty text."""
        sanitized = sanitize_query_text("")
        assert sanitized == ""


class TestErrorResponse:
    """Test error response creation."""

    def test_create_error_response(self):
        """Test error response creation."""
        error = "Test error"
        detail = "Test detail"
        error_code = "TEST_ERROR"

        response = create_error_response(error, detail, error_code)

        assert response["error"] == error
        assert response["detail"] == detail
        assert response["error_code"] == error_code
        assert "timestamp" in response

    def test_create_error_response_minimal(self):
        """Test error response with minimal parameters."""
        response = create_error_response("Error")

        assert response["error"] == "Error"
        assert response["detail"] is None
        assert response["error_code"] is None


class TestQueryMetricsLogging:
    """Test query metrics logging."""

    def test_log_query_metrics(self, caplog):
        """Test query metrics logging."""
        log_query_metrics(
            query="test query",
            mode="semantic_search",
            processing_time_ms=150.5,
            results_count=5,
            confidence_score=0.85,
            tenant_id="test-tenant",
        )

        # Check that log was recorded
        assert len(caplog.records) == 1
        log_record = caplog.records[0]
        assert "Query metrics" in log_record.message
        assert "semantic_search" in log_record.message
        assert "150.5" in log_record.message
        assert "test-tenant" in log_record.message


if __name__ == "__main__":
    pytest.main([__file__])
