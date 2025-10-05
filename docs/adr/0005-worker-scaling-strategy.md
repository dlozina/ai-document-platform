# ADR-005: Worker Scaling Strategy

## Status
Accepted

## Context
As the system grew, we needed to optimize the Celery worker configuration for better performance and scalability. The initial setup had generic workers, but we needed specialized workers with different priorities and concurrency settings based on the type of work they perform.

## Decision
We will implement **specialized worker pools** with different configurations based on workload characteristics:

### Worker Specialization
1. **OCR Workers** (Highest Priority)
   - Priority: 9 (highest)
   - Concurrency: 5
   - Replicas: 2
   - Queue: `ocr_queue`
   - Rationale: OCR is CPU-intensive and the bottleneck

2. **NER Workers** (Medium Priority)
   - Priority: 5
   - Concurrency: 3
   - Replicas: 1
   - Queue: `ner_queue`
   - Rationale: NER is faster than OCR but still CPU-intensive

3. **Embedding Workers** (Medium Priority)
   - Priority: 3
   - Concurrency: 2
   - Replicas: 1
   - Queue: `embedding_queue`
   - Rationale: Memory-intensive, fewer concurrent workers

4. **Completion Workers** (Lowest Priority)
   - Priority: 1
   - Concurrency: 1
   - Replicas: 1
   - Queue: `completion_queue`, `dead_letter_queue`
   - Rationale: Lightweight tasks, single worker sufficient

## Consequences

### Positive
- **Optimized Resource Usage**: Each worker type optimized for its workload
- **Better Performance**: OCR workers get priority and more resources
- **Scalability**: Can scale different worker types independently
- **Fault Isolation**: Different worker types can fail independently
- **Monitoring**: Clear visibility into different worker performance

### Negative
- **Complexity**: More worker configurations to manage
- **Resource Planning**: Need to plan resources for different worker types
- **Configuration Management**: More complex deployment configuration

## Implementation Timeline
- **2025-10-05**: Changed worker structure for scale
- **2025-10-05**: Implemented specialized worker pools in docker-compose

## Related Decisions
- ADR-002: Processing Pipeline Architecture
- ADR-003: Event-Driven vs Direct Processing
