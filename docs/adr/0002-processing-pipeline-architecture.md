# ADR-002: Processing Pipeline Architecture

## Status
Accepted

## Context
We need to process documents through multiple stages:
1. OCR text extraction
2. Named Entity Recognition
3. Embedding generation for semantic search

The processing needs to be reliable, scalable, and handle failures gracefully. We need to decide how to orchestrate these processing steps.

## Decision
We will implement a sequential processing pipeline using Celery workers with priority-based queues:

### Processing Flow
```
Document Upload → OCR Worker → NER Worker → Embedding Worker → Completion
```

### Worker Configuration
- **OCR Worker**: Priority 9 (highest), Concurrency 5, Queue: `ocr_queue`
- **NER Worker**: Priority 5 (medium), Concurrency 3, Queue: `ner_queue`  
- **Embedding Worker**: Priority 3 (medium), Concurrency 2, Queue: `embedding_queue`
- **Completion Worker**: Priority 1 (lowest), Concurrency 1, Queue: `completion_queue`

### Error Handling
- Retry policies with exponential backoff
- Dead letter queue for failed tasks
- Processing status tracking in PostgreSQL

## Consequences

### Positive
- **Sequential Processing**: Ensures OCR completes before NER, NER before embeddings
- **Priority-Based**: OCR gets highest priority as it's the bottleneck
- **Fault Tolerance**: Retry mechanisms and dead letter queues
- **Scalability**: Workers can be scaled independently
- **Monitoring**: Clear visibility into processing pipeline status

### Negative
- **Sequential Bottleneck**: Cannot parallelize processing for single document
- **Complexity**: More complex than simple sequential calls
- **Resource Usage**: Multiple worker processes required

## Implementation Timeline
- **2025-10-01**: Initial worker pipeline implementation
- **2025-10-01**: Fixed workers pipeline issues
- **2025-10-05**: Restructured worker configuration for better scaling

## Related Decisions
- ADR-003: Event-Driven vs Direct Processing
- ADR-005: Worker Scaling Strategy
