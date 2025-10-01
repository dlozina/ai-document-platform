"""
Utility functions for Query Service
"""

import logging
import hashlib
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def calculate_text_hash(text: str) -> str:
    """Calculate SHA-256 hash of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp to ISO string."""
    return timestamp.isoformat()


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple word frequency."""
    # Simple keyword extraction - in production you'd use more sophisticated methods
    import re
    from collections import Counter
    
    # Remove punctuation and convert to lowercase
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Filter out stop words and get top keywords
    keywords = [
        word for word, count in word_counts.most_common(max_keywords * 2)
        if word not in stop_words
    ]
    
    return keywords[:max_keywords]


def validate_query_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize query parameters."""
    validated = {}
    
    # Validate top_k
    if 'top_k' in params:
        top_k = params['top_k']
        if isinstance(top_k, int) and 1 <= top_k <= 100:
            validated['top_k'] = top_k
        else:
            validated['top_k'] = 10
    
    # Validate score_threshold
    if 'score_threshold' in params:
        threshold = params['score_threshold']
        if isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0:
            validated['score_threshold'] = float(threshold)
    
    # Validate tenant_id
    if 'tenant_id' in params and params['tenant_id']:
        validated['tenant_id'] = str(params['tenant_id']).strip()
    
    return validated


def format_search_result(result: Dict[str, Any], max_text_length: int = 500) -> Dict[str, Any]:
    """Format search result for API response."""
    formatted = {
        "document_id": result.get("document_id", ""),
        "filename": result.get("filename", "Unknown"),
        "relevance_score": float(result.get("score", 0.0)),
        "metadata": result.get("metadata", {})
    }
    
    # Format text content
    text = result.get("text", "")
    if len(text) > max_text_length:
        formatted["quoted_text"] = truncate_text(text, max_text_length)
    else:
        formatted["quoted_text"] = text
    
    return formatted


def calculate_confidence_score(
    similarity_scores: List[float],
    entity_matches: int = 0,
    keyword_matches: int = 0
) -> float:
    """Calculate overall confidence score from multiple factors."""
    if not similarity_scores:
        return 0.0
    
    # Base confidence from similarity scores
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    max_similarity = max(similarity_scores)
    
    # Weighted combination
    base_confidence = (avg_similarity * 0.6) + (max_similarity * 0.4)
    
    # Boost for entity matches
    entity_boost = min(0.2, entity_matches * 0.05)
    
    # Boost for keyword matches
    keyword_boost = min(0.1, keyword_matches * 0.02)
    
    final_confidence = min(1.0, base_confidence + entity_boost + keyword_boost)
    
    return round(final_confidence, 3)


def parse_date_range(date_range: Optional[Dict[str, Any]]) -> Optional[Dict[str, datetime]]:
    """Parse date range from filter parameters."""
    if not date_range:
        return None
    
    parsed = {}
    
    if 'start' in date_range and date_range['start']:
        try:
            if isinstance(date_range['start'], str):
                parsed['start'] = datetime.fromisoformat(date_range['start'].replace('Z', '+00:00'))
            else:
                parsed['start'] = date_range['start']
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid start date: {e}")
    
    if 'end' in date_range and date_range['end']:
        try:
            if isinstance(date_range['end'], str):
                parsed['end'] = datetime.fromisoformat(date_range['end'].replace('Z', '+00:00'))
            else:
                parsed['end'] = date_range['end']
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid end date: {e}")
    
    return parsed if parsed else None


def sanitize_query_text(query: str) -> str:
    """Sanitize query text for processing."""
    if not query:
        return ""
    
    # Remove excessive whitespace
    query = ' '.join(query.split())
    
    # Limit length
    if len(query) > 10000:
        query = query[:10000] + "..."
    
    return query.strip()


def create_error_response(error: str, detail: Optional[str] = None, error_code: Optional[str] = None) -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "error": error,
        "detail": detail,
        "error_code": error_code,
        "timestamp": datetime.utcnow().isoformat()
    }


def log_query_metrics(
    query: str,
    mode: str,
    processing_time_ms: float,
    results_count: int,
    confidence_score: float,
    tenant_id: Optional[str] = None
):
    """Log query metrics for monitoring."""
    logger.info(
        f"Query metrics - "
        f"Mode: {mode}, "
        f"Time: {processing_time_ms:.2f}ms, "
        f"Results: {results_count}, "
        f"Confidence: {confidence_score:.3f}, "
        f"Tenant: {tenant_id or 'unknown'}, "
        f"Query: {query[:100]}..."
    )
