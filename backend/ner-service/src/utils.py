"""
Utility functions for NER Service
"""

import hashlib
import logging
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def calculate_text_hash(text: str) -> str:
    """
    Calculate SHA-256 hash of text content.

    Args:
        text: Text content as string

    Returns:
        SHA-256 hash as hexadecimal string
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_text_length(text: str, max_length: int = 1000000) -> bool:
    """
    Validate text length against maximum allowed length.

    Args:
        text: Text to validate
        max_length: Maximum allowed length

    Returns:
        True if text is valid length, False otherwise
    """
    return len(text) <= max_length


def sanitize_text(text: str) -> str:
    """
    Sanitize text by removing or replacing unsafe characters.

    Args:
        text: Original text

    Returns:
        Sanitized text
    """
    # Remove null bytes and control characters (except newlines and tabs)
    sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Normalize whitespace
    sanitized = re.sub(r"\s+", " ", sanitized)

    return sanitized.strip()


def create_error_response(error: str, detail: str | None = None) -> dict[str, Any]:
    """
    Create standardized error response.

    Args:
        error: Error message
        detail: Additional error details

    Returns:
        Error response dictionary
    """
    response = {"error": error}
    if detail:
        response["detail"] = detail
    return response


def log_processing_stats(
    text_length: int,
    processing_time_ms: float,
    model_used: str,
    entity_count: int,
    entity_types: list[str],
) -> None:
    """
    Log processing statistics for monitoring.

    Args:
        text_length: Length of processed text
        processing_time_ms: Processing time in milliseconds
        model_used: Model used for processing
        entity_count: Number of entities found
        entity_types: Types of entities found
    """
    stats = {
        "text_length": text_length,
        "processing_time_ms": f"{processing_time_ms:.2f}",
        "model_used": model_used,
        "entity_count": entity_count,
        "entity_types": entity_types,
    }

    logger.info(f"NER processing stats: {stats}")


class TextValidator:
    """Utility class for text validation."""

    def __init__(self, max_length: int = 1000000):
        self.max_length = max_length

    def validate(self, text: str) -> dict[str, Any]:
        """
        Validate text against constraints.

        Args:
            text: Text to validate

        Returns:
            Validation result with 'valid' boolean and 'error' message if invalid
        """
        # Check if text is empty
        if not text or not text.strip():
            return {"valid": False, "error": "Text cannot be empty"}

        # Check text length
        if len(text) > self.max_length:
            return {
                "valid": False,
                "error": f"Text too long. Maximum length: {self.max_length:,} characters",
            }

        return {"valid": True, "error": None}


def extract_entity_patterns(text: str) -> dict[str, list[str]]:
    """
    Extract common patterns that might be entities using regex.

    This is a fallback method when spaCy models are not available.

    Args:
        text: Text to analyze

    Returns:
        Dictionary of pattern types and matches
    """
    patterns = {
        "emails": re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text
        ),
        "urls": re.findall(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            text,
        ),
        "phone_numbers": re.findall(
            r"(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})", text
        ),
        "dates": re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text),
        "money": re.findall(r"\$\d+(?:,\d{3})*(?:\.\d{2})?", text),
        "percentages": re.findall(r"\d+(?:\.\d+)?%", text),
    }

    # Clean up phone numbers (join groups)
    patterns["phone_numbers"] = ["".join(match) for match in patterns["phone_numbers"]]

    return patterns


def merge_overlapping_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge overlapping entities, keeping the longer one.

    Args:
        entities: List of entity dictionaries

    Returns:
        List of merged entities
    """
    if not entities:
        return []

    # Sort by start position
    sorted_entities = sorted(entities, key=lambda x: x["start"])

    merged = []
    current = sorted_entities[0]

    for next_entity in sorted_entities[1:]:
        # Check for overlap
        if next_entity["start"] < current["end"]:
            # Overlapping entities - keep the longer one
            current_length = current["end"] - current["start"]
            next_length = next_entity["end"] - next_entity["start"]

            if next_length > current_length:
                current = next_entity
        else:
            # No overlap - add current and move to next
            merged.append(current)
            current = next_entity

    # Add the last entity
    merged.append(current)

    return merged


def calculate_entity_density(text: str, entities: list[dict[str, Any]]) -> float:
    """
    Calculate entity density (entities per 1000 characters).

    Args:
        text: Original text
        entities: Detected entities

    Returns:
        Entity density as float
    """
    if not text or not entities:
        return 0.0

    entity_chars = sum(entity["end"] - entity["start"] for entity in entities)
    return (entity_chars / len(text)) * 1000


def get_entity_coverage(entities: list[dict[str, Any]], text_length: int) -> float:
    """
    Calculate what percentage of text is covered by entities.

    Args:
        entities: Detected entities
        text_length: Length of original text

    Returns:
        Coverage percentage (0-100)
    """
    if not entities or text_length == 0:
        return 0.0

    total_entity_length = sum(entity["end"] - entity["start"] for entity in entities)
    return (total_entity_length / text_length) * 100


def format_entity_summary(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Create a summary of detected entities.

    Args:
        entities: List of detected entities

    Returns:
        Summary dictionary
    """
    if not entities:
        return {
            "total_entities": 0,
            "unique_entities": 0,
            "entity_types": [],
            "coverage_percentage": 0.0,
        }

    # Count by type
    type_counts = Counter(entity["label"] for entity in entities)

    # Count unique entities
    unique_entities = len({entity["text"] for entity in entities})

    return {
        "total_entities": len(entities),
        "unique_entities": unique_entities,
        "entity_types": [
            {"type": label, "count": count}
            for label, count in type_counts.most_common()
        ],
        "coverage_percentage": 0.0,  # Will be calculated separately
    }
