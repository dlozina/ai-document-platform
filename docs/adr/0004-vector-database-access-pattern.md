# ADR-004: Vector Database Access Pattern

## Status
Accepted

## Context
We need to store and retrieve embeddings for semantic search. Initially, the embedding service was responsible for both generating embeddings and storing them in Qdrant. However, this created tight coupling and made the embedding service responsible for too many concerns.

## Decision
We will centralize vector database access through the **Ingestion Service** instead of having each service directly access Qdrant.

### Before (Distributed Access)
- Embedding service directly accessed Qdrant
- Query service directly accessed Qdrant
- Multiple services managing vector storage

### After (Centralized Access)
- Ingestion service manages all Qdrant operations
- Embedding service only generates embeddings
- Query service accesses Qdrant through ingestion service
- Single point of vector database management

## Consequences

### Positive
- **Single Responsibility**: Each service has clearer responsibilities
- **Centralized Management**: Vector database operations are centralized
- **Consistency**: Single point of truth for vector storage logic
- **Easier Maintenance**: Changes to vector storage only affect ingestion service
- **Better Testing**: Easier to mock and test vector operations

### Negative
- **Additional Network Calls**: Query service needs to go through ingestion service
- **Potential Bottleneck**: Ingestion service becomes more critical
- **Coupling**: Services are more coupled through ingestion service

## Implementation Timeline
- **2025-10-05**: Changed vector database access point
- **2025-10-05**: Moved Qdrant operations to ingestion service
- **2025-10-05**: Simplified embedding service to focus only on embedding generation

## Related Decisions
- ADR-001: Microservices Architecture
- ADR-002: Processing Pipeline Architecture
