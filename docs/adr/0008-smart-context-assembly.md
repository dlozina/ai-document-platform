# ADR-008: Smart Context Assembly

## Status
Accepted

## Context
When answering questions using RAG (Retrieval-Augmented Generation), we need to assemble context from multiple document chunks. The challenge is to provide enough context for accurate answers while staying within token limits.

## Decision
We will implement **smart context assembly** with the following approach:

### Context Assembly Strategy
1. **Retrieve Relevant Chunks**: Use vector similarity search to find relevant chunks
2. **Rank by Relevance**: Use cross-encoder reranking for better relevance
3. **Assemble Context**: Combine chunks while respecting token limits
4. **Maintain Coherence**: Ensure context flows logically

### Context Optimization
- **Token Limit Management**: Respect maximum context length
- **Chunk Prioritization**: Prioritize most relevant chunks
- **Context Truncation**: Smart truncation when limits are reached
- **Metadata Inclusion**: Include chunk metadata for better context

### RAG Integration
- Generate embeddings for user questions
- Search for relevant document chunks
- Assemble context from top-ranked chunks
- Pass context to LLM for answer generation

## Consequences

### Positive
- **Better Answers**: More relevant context leads to better answers
- **Token Efficiency**: Optimized use of available token limits
- **Scalability**: Can handle large document collections
- **Flexibility**: Adapts to different question types

### Negative
- **Complexity**: More complex than simple context assembly
- **Performance**: Additional processing for context optimization
- **Dependencies**: Relies on embedding and reranking quality

## Implementation Timeline
- **2025-10-03**: Added smart context assembly
- **2025-10-04**: Fixed QA endpoint with improved context handling

## Related Decisions
- ADR-007: Document Chunking Strategy
- ADR-004: Vector Database Access Pattern
