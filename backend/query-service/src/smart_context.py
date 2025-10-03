"""
Smart context selection utilities for RAG
"""

import re
from typing import List, Dict, Any, Tuple
from collections import Counter


def smart_truncate_text(text: str, max_length: int, query: str = "") -> str:
    """
    Intelligently truncate text while preserving relevance to query.
    
    Args:
        text: Text to truncate
        max_length: Maximum length in characters
        query: Query to optimize relevance for
        
    Returns:
        Truncated text that preserves meaning
    """
    if len(text) <= max_length:
        return text
    
    if not query.strip():
        # If no query, use sentence-based truncation
        return _truncate_by_sentences(text, max_length)
    
    # Find most relevant sentences
    relevant_text = _extract_relevant_sentences(text, query, max_length)
    
    if len(relevant_text) >= max_length * 0.8:  # If we got good coverage
        return relevant_text
    
    # Fallback to sentence-based truncation
    return _truncate_by_sentences(text, max_length)


def _extract_relevant_sentences(text: str, query: str, max_length: int) -> str:
    """Extract sentences most relevant to the query."""
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return text[:max_length]
    
    # Extract query keywords (remove common words)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    query_words = set(word.lower() for word in query.split() if word.lower() not in stop_words)
    
    # Score sentences by keyword overlap and position
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        sentence_words = set(word.lower() for word in sentence.split())
        
        # Keyword overlap score
        keyword_score = len(query_words.intersection(sentence_words))
        
        # Position score (slightly favor earlier sentences for context)
        position_score = 1.0 / (1.0 + i * 0.1)
        
        # Combined score
        total_score = keyword_score + position_score
        
        scored_sentences.append((total_score, i, sentence))
    
    # Sort by relevance score
    scored_sentences.sort(reverse=True)
    
    # Build context from most relevant sentences
    selected_sentences = []
    current_length = 0
    
    for score, i, sentence in scored_sentences:
        sentence_with_period = sentence + "."
        if current_length + len(sentence_with_period) <= max_length:
            selected_sentences.append((i, sentence_with_period))
            current_length += len(sentence_with_period)
        else:
            break
    
    # Sort selected sentences by original order
    selected_sentences.sort(key=lambda x: x[0])
    
    return " ".join(sentence for _, sentence in selected_sentences)


def _truncate_by_sentences(text: str, max_length: int) -> str:
    """Truncate text at sentence boundaries."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    result = ""
    for sentence in sentences:
        sentence_with_period = sentence + "."
        if len(result + sentence_with_period) <= max_length:
            result += sentence_with_period
        else:
            break
    
    return result or text[:max_length]


def extract_context_around_keywords(text: str, keywords: List[str], context_window: int = 500) -> str:
    """
    Extract context around specific keywords in the text.
    
    Args:
        text: Full text
        keywords: Keywords to find context around
        context_window: Characters before/after keyword
        
    Returns:
        Context around keywords
    """
    if not keywords:
        return text[:context_window * 2]
    
    # Find all keyword positions
    keyword_positions = []
    text_lower = text.lower()
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        start = 0
        while True:
            pos = text_lower.find(keyword_lower, start)
            if pos == -1:
                break
            keyword_positions.append(pos)
            start = pos + 1
    
    if not keyword_positions:
        return text[:context_window * 2]
    
    # Sort positions
    keyword_positions.sort()
    
    # Extract context around keywords
    contexts = []
    for pos in keyword_positions:
        start = max(0, pos - context_window)
        end = min(len(text), pos + len(keyword) + context_window)
        context = text[start:end]
        contexts.append(context)
    
    # Combine contexts and remove duplicates
    combined_context = " ... ".join(contexts)
    
    return combined_context


def chunk_document_intelligently(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split document into intelligent chunks with metadata.
    
    Args:
        text: Document text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    if len(text) <= chunk_size:
        return [{
            "text": text,
            "chunk_index": 0,
            "total_chunks": 1,
            "start_char": 0,
            "end_char": len(text),
            "chunk_type": "single"
        }]
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundaries
        if end < len(text):
            # Look for sentence boundaries in the last 200 characters
            search_start = max(start, end - 200)
            last_period = text.rfind('.', search_start, end)
            last_exclamation = text.rfind('!', search_start, end)
            last_question = text.rfind('?', search_start, end)
            
            # Find the best break point
            break_points = [last_period, last_exclamation, last_question]
            break_points = [bp for bp in break_points if bp > start + chunk_size // 2]
            
            if break_points:
                end = max(break_points) + 1
        
        # Extract chunk
        chunk_text = text[start:end].strip()
        
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "chunk_index": chunk_index,
                "total_chunks": 0,  # Will be updated later
                "start_char": start,
                "end_char": end,
                "chunk_type": "middle" if chunk_index > 0 else "start"
            })
            chunk_index += 1
        
        # Move start position with overlap
        start = end - overlap
    
    # Update total chunks count
    for chunk in chunks:
        chunk["total_chunks"] = len(chunks)
        if chunk["chunk_index"] == len(chunks) - 1:
            chunk["chunk_type"] = "end"
    
    return chunks


def merge_overlapping_chunks(chunks: List[Dict[str, Any]], max_length: int = 2000) -> List[Dict[str, Any]]:
    """
    Merge overlapping chunks if they're too small.
    
    Args:
        chunks: List of chunk dictionaries
        max_length: Maximum length for merged chunks
        
    Returns:
        List of merged chunks
    """
    if not chunks:
        return chunks
    
    merged_chunks = []
    current_chunk = chunks[0].copy()
    
    for next_chunk in chunks[1:]:
        # Check if we can merge
        combined_length = len(current_chunk["text"]) + len(next_chunk["text"])
        
        if combined_length <= max_length:
            # Merge chunks
            current_chunk["text"] += " " + next_chunk["text"]
            current_chunk["end_char"] = next_chunk["end_char"]
            current_chunk["chunk_type"] = "merged"
        else:
            # Can't merge, save current chunk and start new one
            merged_chunks.append(current_chunk)
            current_chunk = next_chunk.copy()
    
    # Add the last chunk
    merged_chunks.append(current_chunk)
    
    # Update chunk indices
    for i, chunk in enumerate(merged_chunks):
        chunk["chunk_index"] = i
        chunk["total_chunks"] = len(merged_chunks)
    
    return merged_chunks
